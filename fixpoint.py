# -*- coding: utf-8 -*-
"""
@Time       : 2023/2/17 11:06
@Author     : Juxin Niu (juxin.niu@outlook.com)
@FileName   : fixpoint.py
@Description: 
"""
import re
from collections import deque
from copy import deepcopy
from functools import reduce
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


class CacheSetState:
    def __init__(self, max_age: int, evicted: bool = False):
        self.max_age = max_age
        self.evicted_line = evicted
        self.state_len = max_age if not evicted else max_age + 1
        self.states = [set() for _ in range(self.state_len)]

    def where(self, b: Hashable):
        for age, state in enumerate(self.states):
            if b in state:
                return age
        return -1

    def new_access_insert(self, b: Hashable):
        age = self.where(b)
        if age != -1:
            self.states[age].discard(b)
        self.states[0].add(b)

    def all_blocks(self):
        return [b for s in self.states for b in s]

    def age_shifting(self, age: int):
        if age == self.max_age - 1:
            """ If the oldest memory block needs to be offset, then if there is an eviction line, 
            move to the eviction line, otherwise, delete it directly from the state. """
            if self.evicted_line:
                self.states[age + 1].update(self.states[age])
            self.states[age].clear()
        else:
            self.states[age + 1].update(self.states[age])
            self.states[age].clear()


class SetStateOperation:

    UpdateFuncTy = Callable[[CacheSetState, Hashable], bool]
    JoinFuncTy = Callable[[CacheSetState, CacheSetState], CacheSetState]

    @staticmethod
    def must_update(s: CacheSetState, b: Hashable) -> bool:
        b_age = s.where(b)
        is_hit = b_age != -1
        b_age = b_age if is_hit else s.max_age
        for age in reversed(range(0, b_age)):
            s.age_shifting(age)
        s.new_access_insert(b)
        return is_hit

    @staticmethod
    def may_update(s: CacheSetState, b: Hashable) -> bool:
        b_age = s.where(b)
        is_hit = b_age != -1
        b_age = b_age if is_hit else s.max_age - 1
        for age in reversed(range(0, b_age + 1)):
            s.age_shifting(age)
        s.new_access_insert(b)
        return is_hit

    @staticmethod
    def persistent_update(s: CacheSetState, b: Hashable):
        b_age = s.where(b)
        is_hit = (b_age != -1) and (b_age != s.max_age)
        b_age = b_age if b_age != -1 else s.max_age
        for age in reversed(range(0, b_age)):
            s.age_shifting(age)
        s.new_access_insert(b)
        return is_hit

    @staticmethod
    def must_join(s1: CacheSetState, s2: CacheSetState):
        new_s = CacheSetState(s1.max_age, evicted=False)
        blocks = set(s1.all_blocks()).intersection(set(s2.all_blocks()))
        for b in blocks:
            age1, age2 = s1.where(b), s2.where(b)
            new_s.states[max(age1, age2)].add(b)
        return new_s

    @staticmethod
    def may_join(s1: CacheSetState, s2: CacheSetState):
        new_s = CacheSetState(s1.max_age, evicted=False)
        blocks = set(s1.all_blocks()).union(set(s2.all_blocks()))
        for b in blocks:
            age1, age2 = s1.where(b), s2.where(b)
            target_age = min(age1, age2)
            target_age = target_age if target_age != -1 else max(age1, age2)
            new_s.states[target_age].add(b)
        return new_s

    @staticmethod
    def persistent_join(s1: CacheSetState, s2: CacheSetState):
        new_s = CacheSetState(s1.max_age, evicted=True)
        blocks = set(s1.all_blocks()).union(set(s2.all_blocks()))
        for b in blocks:
            age1, age2 = s1.where(b), s2.where(b)
            new_s.states[max(age1, age2)].add(b)
        return new_s

    @staticmethod
    def state_same(s1: CacheSetState, s2: CacheSetState):
        for l1, l2 in zip(s1.states, s2.states):
            if l1 != l2:
                return False
        return True


class MemoryBlock:
    def __init__(self, set_index: int, tag: int):
        self.set_index = set_index
        self.tag = tag

    def __str__(self):
        return "<MemBlk tag:{} set_index:{}>".format(hex(self.tag), hex(self.set_index))

    def __repr__(self):
        return self.__str__()


class CacheConfig:
    def __init__(self, offset_len: int, set_index_len: int, assoc: int):
        self.offset_len = offset_len
        self.set_index_len = set_index_len
        self.assoc = assoc

    def block_gen(self, addr_beg, addr_end):
        blocks: List[MemoryBlock] = list()

        s_cache_addr, e_cache_addr = addr_beg >> self.offset_len, addr_end >> self.offset_len
        for cache_addr in range(s_cache_addr, e_cache_addr):
            set_index = cache_addr & ((1 << self.set_index_len) - 1)
            tag = cache_addr >> self.set_index_len
            blocks.append(MemoryBlock(set_index=set_index, tag=tag))

        return blocks

    def generate_set_state(self, evicted: bool):
        return CacheSetState(max_age=self.assoc, evicted=evicted)


class FixpointNode:
    def __init__(self, ident: str, access_blocks: Sequence[MemoryBlock]):
        self.ident = ident
        self.access_blocks = tuple(access_blocks)
        self.is_hit: List[bool] = [False for _ in range(len(self.access_blocks))]

        self.incoming: List[FixpointNode] = list()
        self.outgoing: List[FixpointNode] = list()

        self.__in_state_by_set: Dict[int, CacheSetState] = dict()
        self.__out_state_by_set: Dict[int, CacheSetState] = dict()

    def clear_in_state(self):
        self.__in_state_by_set.clear()

    def clear_out_state(self):
        self.__out_state_by_set.clear()

    def set_in_state(self, idx: int, s: CacheSetState):
        self.__in_state_by_set[idx] = s

    def get_in_state(self, idx: int) -> Optional[CacheSetState]:
        return self.__in_state_by_set.get(idx, None)

    def set_out_state(self, idx: int, s: CacheSetState):
        self.__out_state_by_set[idx] = s

    def get_out_state(self, idx: int) -> Optional[CacheSetState]:
        return self.__out_state_by_set.get(idx, None)


class FixpointGraph:
    def __init__(self, nodes: Iterable[str], edges: Iterable[Tuple[str, str], ...], access_mapping: Dict[str, Tuple[MemoryBlock, ...]]):
        """"""
        """ Build nodes and ident mapping. """
        # For those nodes that do not have entry in access_mapping, the access block set is empty.
        self.all_nodes = [FixpointNode(ident,
                                       access_blocks=access_mapping.get(ident, tuple())) for ident in nodes]
        self.ident_mapping: Dict[str, FixpointNode] = {n.ident: n for n in self.all_nodes}
        """ All edges. """
        for src, dst in edges:
            try:
                n_src, n_dst = self.ident_mapping[src], self.ident_mapping[dst]
            except KeyError:
                # Ignore all unknown nodes.
                pass
            else:
                n_src.outgoing.append(n_dst)
                n_dst.incoming.append(n_src)
        """ Do topsort for all nodes. """
        self.__all_nodes: List[FixpointNode] = self.__topsort()

    def __topsort(self):
        sorted_nodes = list()
        q = deque([n for n in self.__all_nodes if len(n.incoming) == 0])
        checked = set([n.ident for n in q])
        while len(q) > 0:
            cur_node = q.popleft()
            checked.add(cur_node.ident)
            sorted_nodes.append(cur_node)
            unvisited_successor = [out_ident for out_ident in cur_node.outgoing if out_ident.ident not in checked]
            q.extend(unvisited_successor)
        return sorted_nodes

    def fixpoint_set(self, set_idx: int, join_func: SetStateOperation.JoinFuncTy,
                     update_func: SetStateOperation.UpdateFuncTy) -> bool:

        is_fixpoint = True

        for cur_node in self.all_nodes:
            to_join_states = [pred.get_out_state(set_idx) for pred in cur_node.incoming]
            if len(to_join_states) > 0:
                joined_state = reduce(join_func, to_join_states)
                cur_node.set_in_state(set_idx, joined_state)
                in_eq = SetStateOperation.state_same(cur_node.get_in_state(set_idx), joined_state)
            else:
                in_eq = True
            is_hit_eq = True
            updated_out_state = deepcopy(cur_node.get_in_state(set_idx))
            for idx, block in enumerate(cur_node.access_blocks):
                if block.set_index != set_idx:
                    continue
                hit_flag: bool = update_func(updated_out_state, block)
                is_hit_eq = is_hit_eq and (hit_flag == cur_node.is_hit[idx])
                cur_node.is_hit[idx] = hit_flag
            out_eq = SetStateOperation.state_same(cur_node.get_out_state(set_idx), updated_out_state)
            cur_node.set_out_state(set_idx, updated_out_state)
            is_fixpoint = is_fixpoint and (in_eq and is_hit_eq and out_eq)

        return is_fixpoint


def read_from_file(f: str) -> Tuple[CacheConfig, FixpointGraph]:
    node_pattern = re.compile(r"^\s*(\w+)\s*;.*$")
    edge_pattern = re.compile(r"^\s*(\w+)\s*->\s*(\w+)\s*;.*$")
    access_pattern = re.compile(r"^\s*\[\s*(\w+)\s*](.+);.*$")
    key_val_pattern = re.compile(r"^\s*(\w+)\s*:\s*(\w+)\s*;.*$")

    results = {'nodes': [], 'edges': [], 'access': dict()}

    def do_parser(line: str):
        nonlocal node_pattern, edge_pattern, access_pattern, key_val_pattern
        nonlocal results

        line = line.strip()
        if not line or line.startswith(';'):
            # Skip all blank lines and command-only lines.
            return

        """ Try to match node. """
        m = re.match(node_pattern, line)
        if m:
            results['nodes'].append(*m.groups())
            return
        """ Try to match edge. """
        m = re.match(edge_pattern, line)
        if m:
            results['edges'].append(tuple(m.groups()))
            return
        """ Try to match memory access. """
        m = re.match(access_pattern, line)
        if m:
            node, acc = m.groups()
            if node in results['access']:
                raise ValueError("There are multiple memory accesses to the node {}.".format(node))
            # A node may access many un-continuous memory ranges. Append them into a list one by one with orders.
            acc_list: List[Tuple[int, int]] = list()
            for item in [item.strip() for item in acc.split(',')]:
                try:
                    addr_list = [int(a, 16) if a.startswith(('0x', '0X')) else int(a, 10) for a in [a.strip() for a in item.split('-')]]
                    if len(addr_list) == 1:
                        acc_list.append((addr_list[0], addr_list[0] + 1))
                    elif len(addr_list) == 2:
                        acc_list.append((addr_list[0], addr_list[1]))
                    else:
                        raise RuntimeError
                except Exception:
                    raise ValueError("Unexpected access (range): {}.".format(item))
            # Add to results.
            results['access'][node] = tuple(acc_list)
            return
        """ Try to match other key-value pairs. """
        m = re.match(key_val_pattern, line)
        if m:
            k, v = m.groups()
            if k in results:
                raise KeyError("Key {} is already added.".format(k))
            results[k] = v
            return
        """ Do not matched to any one pattern. """
        raise ValueError("Unrecognized line.")

    with open(f, 'r', encoding='utf-8') as fp:
        text = fp.readlines()
    for idx, ln in enumerate(text):
        try:
            do_parser(ln)
        except Exception as e:
            raise ValueError("An error was encountered while processing line {}.\n"
                             "<:Line Content:> {}\n"
                             "<:Error Details:> {}\n".format(idx, ln, e.__str__()))

    offset_len = results.get("cache_offset", 6)
    set_index_len = results.get("cache_set_index", 8)
    assoc = results.get("cache_assoc", 4)
    config = CacheConfig(offset_len, set_index_len, assoc)

    access_rlt: Dict[str, Tuple[Tuple[int, int], ...]] = results['access']
    access_mb: Dict[str, Tuple[MemoryBlock, ...]] = dict()
    for ident, trace in access_rlt:
        all_blocks = list()
        for r in trace:
            blocks = config.block_gen(*r)
            all_blocks.extend(blocks)
        access_mb[ident] = tuple(all_blocks)

    graph = FixpointGraph(results['nodes'], results['edges'], access_mb)
    return config, graph


def fixpoint(config: CacheConfig, graph: FixpointGraph, analysis_type: str, **kwargs):
    """
    Supported analysis type:
     - must
     - may
     - persistent

    Supported keyword arguments:
     - ``considered_set``: [Sequence of int] Specify which sets to perform fixpoint iteration on. For default, all set is considered.
     - ``yield_every_iter``: [bool, default ``False``] If ``True``, function will yield set index,
       current iteration count and fixpoint flag after every iteration.
     - ``no_init``: [bool, default ``False``] Do not initialize set state.
     - ``max_iter``: [int, default 2147483648] The maximum number of iterations that can be accepted for each cache set.

    :return: If all considered set reaches fixpoint.
    """

    set_number = int(2 ** config.set_index_len)
    considered_set = set(kwargs.get('considered_set', range(set_number)))

    def state_initialization(evicted: bool):
        nonlocal set_number
        if kwargs.get('no_init', False):
            # Do not do cache set state initialization.
            return
        for n in graph.all_nodes:
            for idx in considered_set:
                n.set_in_state(idx, config.generate_set_state(evicted=evicted))
                n.set_out_state(idx, config.generate_set_state(evicted=evicted))

    if analysis_type == 'must':
        state_initialization(False)
        join_func, update_func = SetStateOperation.must_join, SetStateOperation.must_update
    elif analysis_type == 'may':
        state_initialization(False)
        join_func, update_func = SetStateOperation.may_join, SetStateOperation.may_update
    elif analysis_type == 'persistent':
        state_initialization(True)
        join_func, update_func = SetStateOperation.persistent_join, SetStateOperation.persistent_update
    else:
        raise ValueError("Unknown analysis type {}. Supported types are must, may and persistent.".format(analysis_type))

    yield_every_iter = kwargs.get('yield_every_iter', False)
    max_iter_number = kwargs.get('max_iter', int(2 ** 31))

    is_fixpoint = True
    for set_idx in considered_set:
        it = 0
        is_set_fixpoint = False
        while it < max_iter_number:
            it += 1
            is_set_fixpoint = graph.fixpoint_set(set_idx, join_func=join_func, update_func=update_func)
            if yield_every_iter:
                yield set_idx, it, is_set_fixpoint
        is_fixpoint = is_fixpoint and is_set_fixpoint

    return is_fixpoint
