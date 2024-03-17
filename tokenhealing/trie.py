"""Minimalistic trie implementation focusing on prefix search. """

class Node:
    def __init__(self):
        self.value = None
        self.children: dict[str, Node] = {}

class Trie:
    def __init__(self, *args, **kwargs):
        self.root = Node()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        """Update the trie with key-value pairs."""
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __setitem__(self, key, value):
        """Set the value at the node for the given key."""
        crawler = self.root
        for char in key:
            crawler = crawler.children.setdefault(char, Node())
        crawler.value = value

    def extensions(self, prefix: str) -> list:
        """Retrieve values starting with a given prefix."""
        crawler = self.root
        for char in prefix:
            crawler = crawler.children[char]
        return self._collect_values(crawler)

    def _collect_values(self, node: Node) -> list:
        """Recursively collect all values under the given node."""
        values = [] if node.value is None else [node.value]
        for child in node.children.values():
            values.extend(self._collect_values(child))
        return values
