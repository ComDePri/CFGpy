import os
import json
from warnings import warn
import networkx as nx
from CFGpy import NAS_PATH

CFG_RESOURCES_PATH = os.path.join(NAS_PATH, "Projects", "CFG")
SHORTEST_PATHS_DICT_PATH = os.path.join(CFG_RESOURCES_PATH, "shortest_path_len.json")


def is_effective_integer(value: int) -> bool:
    return value == int(value)


class ShortestPathLengthFinder:
    SHAPE_NETWORK = nx.read_adjlist(os.path.join(CFG_RESOURCES_PATH, "all_shapes.adjlist"), nodetype=int)

    def __init__(self):
        # TODO consider: should these also be class variables?
        self.shortest_path_len = self._read_shortest_path_dict()
        self.new_shortest_path_len = {}

    @staticmethod
    def _read_shortest_path_dict():
        try:
            with open(SHORTEST_PATHS_DICT_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            warn(f"Warning: file {SHORTEST_PATHS_DICT_PATH} not found.")
        except json.JSONDecodeError:
            warn(f"Warning: {SHORTEST_PATHS_DICT_PATH} contains invalid JSON. Starting with an empty dictionary.")

        return {}

    def get_shortest_path_len(self, shape1: int, shape2: int):
        shape1, shape2 = min(shape1, shape2), max(shape1, shape2)

        if not is_effective_integer(shape1) or not is_effective_integer(shape2):
            raise TypeError(
                f"shape1 is of type {type(shape1)} and shape2 is of type {type(shape2)}. Both must be type int.")

        key = f"({int(shape1)}, {int(shape2)})"  # JSON doesn't allow tuples as keys, so they're stringified
        if key in self.shortest_path_len:
            return self.shortest_path_len[key]

        if key in self.new_shortest_path_len:
            return self.new_shortest_path_len[key]

        shortest_path_len = nx.shortest_path_length(self.SHAPE_NETWORK, source=shape1, target=shape2)
        self.new_shortest_path_len[key] = shortest_path_len

        return shortest_path_len

    def __del__(self):
        pass
        # In future versions, this can upload self.new_shortest_path_len to the lab's cloud service
        # where it can be used to update shortest_path_len.json
