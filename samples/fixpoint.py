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
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple


class CacheSetState:
    def __init__(self, max_age: int, evicted: bool = False):
        self.max_age = max_age
        self.evicted_line = evicted
        self.state_len = max_age if not evicted else max_age + 1
        self.states = [set() for _ in range(self.state_len)]

    def __str__(self):
        if self.evicted_line:
            state_str = ", ".join(['{}' if len(o) == 0 else o.__str__() for o in self.states[: -1]])
            evicted_str = self.states[-1].__str__()
            return f"line:[{state_str}] evict:{evicted_str}"
        else:
            state_str = ", ".join(['{}' if len(o) == 0 else o.__str__() for o in self.states])
            return f"line:[{state_str}]"

    def __repr__(self):
        return "<CacheSetState {}>".format(self.__str__())

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


def must_update(s: CacheSetState, b: Hashable) -> bool:
    b_age = s.where(b)
    is_hit = b_age != -1
    b_age = b_age if is_hit else s.max_age
    for age in reversed(range(0, b_age)):
        s.age_shifting(age)
    s.new_access_insert(b)
    return is_hit


def may_update(s: CacheSetState, b: Hashable) -> bool:
    b_age = s.where(b)
    is_hit = b_age != -1
    b_age = b_age if is_hit else s.max_age - 1
    for age in reversed(range(0, b_age + 1)):
        s.age_shifting(age)
    s.new_access_insert(b)
    return is_hit


def persistent_update(s: CacheSetState, b: Hashable):
    b_age = s.where(b)
    is_hit = (b_age != -1) and (b_age != s.max_age)
    b_age = b_age if b_age != -1 else s.max_age
    for age in reversed(range(0, b_age)):
        s.age_shifting(age)
    s.new_access_insert(b)
    return is_hit


def must_join(s1: CacheSetState, s2: CacheSetState):
    new_s = CacheSetState(s1.max_age, evicted=False)
    blocks = set(s1.all_blocks()).intersection(set(s2.all_blocks()))
    for b in blocks:
        age1, age2 = s1.where(b), s2.where(b)
        new_s.states[max(age1, age2)].add(b)
    return new_s


def may_join(s1: CacheSetState, s2: CacheSetState):
    new_s = CacheSetState(s1.max_age, evicted=False)
    blocks = set(s1.all_blocks()).union(set(s2.all_blocks()))
    for b in blocks:
        age1, age2 = s1.where(b), s2.where(b)
        target_age = min(age1, age2)
        target_age = target_age if target_age != -1 else max(age1, age2)
        new_s.states[target_age].add(b)
    return new_s


def persistent_join(s1: CacheSetState, s2: CacheSetState):
    new_s = CacheSetState(s1.max_age, evicted=True)
    blocks = set(s1.all_blocks()).union(set(s2.all_blocks()))
    for b in blocks:
        age1, age2 = s1.where(b), s2.where(b)
        new_s.states[max(age1, age2)].add(b)
    return new_s


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

        self.incoming: List[FixpointNode] = list()
        self.outgoing: List[FixpointNode] = list()

        self.in_state: Optional[CacheSetState] = None
        self.out_state: Optional[CacheSetState] = None
        self.is_hit: List[bool] = [False for _ in range(len(self.access_blocks))]

    def __str__(self):
        return "<{} ACC:{} IN:{}, OUT:{} IN_STAT:{}, OUT_STAT:{}>".format(
            self.ident, self.access_blocks, self.incoming, self.outgoing, self.in_state, self.out_state
        )

    def __repr__(self):
        return self.__str__()


class FixpointGraph:
    def __init__(self, nodes: Iterable[str], edges: Iterable[Tuple[str, str], ...], access_mapping: Dict[str, Tuple[MemoryBlock, ...]]):
        """"""
        """ Build nodes and ident mapping. """
        self.all_nodes = [FixpointNode(ident,
                                         access_blocks=access_mapping.get(ident, tuple())) for ident in nodes]
        self.ident_mapping: Dict[str, FixpointNode] = {n.ident: n for n in self.all_nodes}
        """ All edges. """
        for src, dst in edges:
            try:
                n_src, n_dst = self.ident_mapping[src], self.ident_mapping[dst]
            except KeyError:
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


def fixpoint_set(graph: FixpointGraph, join_func: Callable, update_func: callable, set_idx: int) -> bool:
    is_fixpoint = True
    for cur_node in graph.all_nodes:
        to_join_states = [pred.out_state for pred in cur_node.incoming]
        if len(to_join_states) > 0:
            joined_state = reduce(join_func, to_join_states)
            cur_node.in_state = joined_state
            in_eq = state_same(cur_node.in_state, joined_state)
        else:
            in_eq = True
        is_hit_eq = True
        updated_out_state = deepcopy(cur_node.in_state)
        for idx, block in enumerate(cur_node.access_blocks):
            if block.set_index != set_idx:
                continue
            


"""
=======================================================================================================================================
"""


def round(self):
    is_fixpoint = True
    for cur_node_ident in self.__sorted_idents:
        cur_node = self.__node_dict[cur_node_ident]
        to_join_states = [self.__node_dict[pred].out_state for pred in cur_node.incoming]
        if len(to_join_states) > 0:
            joined_state = reduce(self.__join_func, to_join_states)
            cur_node.in_state = joined_state
            in_eq = state_same(cur_node.in_state, joined_state)
        else:
            in_eq = True
        updated_out_state = deepcopy(cur_node.in_state)
        for block in cur_node.access_blocks:
            self.__update_func(updated_out_state, block)
        out_eq = state_same(cur_node.out_state, updated_out_state)
        cur_node.out_state = updated_out_state
        is_fixpoint = is_fixpoint and in_eq and out_eq
    return is_fixpoint

# class Fixpoint:
#     def __init__(self, ty: FixpointType, entry: Hashable, all_nodes: Sequence[FixpointNode]):
#         self.__join_func, self.__update_func = {
#             FixpointType.Must: (must_join, must_update),
#             FixpointType.May: (may_join, may_update),
#             FixpointType.Persistent: (persistent_join, persistent_update)}.get(ty, (None, None))
#         assert self.__join_func is not None, "Unknown fixpoint type {}.".format(ty)
#         self.__node_dict: Dict[Hashable, FixpointNode] = {o.ident: o for o in all_nodes}
#         self.__entry_ident = entry
#         assert self.__entry_ident in self.__node_dict, \
#             "Cannot find entry node {} in all given nodes {}.".format(self.__entry_ident, self.__node_dict)
#         self.__pse_topsort()
#
#     def __pse_topsort(self):
#         seq = list()
#         checked, q = {self.__entry_ident}, deque([self.__entry_ident])
#         while len(q) > 0:
#             cur_node = self.__node_dict[q.popleft()]
#             checked.add(cur_node.ident)
#             seq.append(cur_node.ident)
#             unvisited_successor = [out_ident for out_ident in cur_node.outgoing if out_ident not in checked]
#             q.extend(unvisited_successor)
#         self.__sorted_idents = tuple(seq)
#
#     @property
#     def pse_topsort_seq(self):
#         return self.__sorted_idents
#

#
#     def run(self):
#         is_fixpoint = False
#         while not is_fixpoint:
#             is_fixpoint = self.round()
