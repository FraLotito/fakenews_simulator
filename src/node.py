from enum import Enum


class NodeType(Enum):
    Common = 0,
    Conspirator = 1,
    Influencer = 2,
    Debunker = 3,
    Media = 4


class Node:
    def __init__(self, node_type: NodeType):
        self.type = node_type
