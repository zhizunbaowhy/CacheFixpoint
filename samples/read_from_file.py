import re
from typing import Iterable, List, Tuple


def parser(text: Iterable[str]):

    node_pattern = re.compile(r"^\s*(\w+)\s*;.*$")
    edge_pattern = re.compile(r"^\s*(\w+)\s*->\s*(\w+)\s*;.*$")
    access_pattern = re.compile(r"^\s*\[\s*(\w+)\s*](.+);.*$")
    key_val_pattern = re.compile(r"^\s*(\w+)\s*:\s*(\w+)\s*;.*$")

    results = {'nodes': [], 'edges': [], 'access': dict()}
    for idx, line in enumerate(text):
        line = line.strip()
        if not line:
            # Skip all blank lines.
            continue
        """ Match by rules, and handle errors encountered. """
        try:
            """ Try to match node. """
            m = re.match(node_pattern, line)
            if m:
                results['nodes'].append(*m.groups())
                continue

            """ Try to match edge. """
            m = re.match(edge_pattern, line)
            if m:
                results['edges'].append(tuple(m.groups()))
                continue

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
                continue

            """ Try to match other key-value pairs. """
            m = re.match(key_val_pattern, line)
            if m:
                k, v = m.groups()
                if k in results:
                    raise KeyError("Key {} is already added.".format(k))
                results[k] = v
                continue

            """ Do not matched to any one pattern. """
            raise ValueError("Unrecognized line.")
        except Exception as e:
            raise ValueError("An error was encountered while processing line {}.\n"
                             "<:Line Content:> {}\n"
                             "<:Error Details:> {}\n".format(idx, line, e.__str__()))
    return results


def read_from_file(f: str):
    with open(f, 'r', encoding='utf-8') as fp:
        text = fp.readlines()
    return parser(text)
