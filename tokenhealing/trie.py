"""Minimalistic trie implementation focusing on prefix search."""

class Node:
    def __init__(self):
        self.children, self.value = {}, None

class Trie:
    """A trie implementation with a dict-like interface plus some extensions."""
    def __init__(self, *args):
        self.root = Node()
        for key, value in dict(*args).items():
            self._setitem__(key, value)

    def _setitem__(self, key, value):
        """Set the value at the node for the given key."""
        node = self.root
        for char in key:
            node = node.children.setdefault(char, Node())
        node.value = value

    def _collect_values(self, node):
        """Recursively collect all values under the given node."""
        values = [] if node.value is None else [node.value]
        for child in node.children.values():
            values.extend(self._collect_values(child))
        return values

    def values(self, prefix):
        """Retrieve values starting with a given prefix."""
        node = self.root
        for char in prefix:
            node = node.children[char]
        return self._collect_values(node)
