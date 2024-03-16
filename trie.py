"""Refactored Python 3.8+ implementation of a trie data structure from https://github.com/google/pygtrie
This implementation focuses on prefix node search
"""

from collections.abc import MutableMapping

class ShortKeyError(KeyError):
    """Raised when a given key is a prefix of an existing longer key but does not have a value associated with itself."""

class _NoChildren:
    """Represents lack of any children."""
    def __bool__(self):
        return False

    def add(self, parent, step):
        node = _Node()
        parent.children = _OneChild(step, node)
        return node

    require = add

_EMPTY = _NoChildren()

class _OneChild:
    """Represents a single child."""
    def __init__(self, step, node):
        self.step = step
        self.node = node

    def __bool__(self):
        return True

    def items(self):
        """Yields step-node pairs, mimicking dict.items()."""
        yield (self.step, self.node)

    def get(self, step):
        return self.node if step == self.step else None

    def add(self, parent, step):
        node = _Node()
        parent.children = _Children({self.step: self.node, step: node})
        return node

    def require(self, parent, step):
        return self.node if self.step == step else self.add(parent, step)

class _Children(dict):
    """Represents more than one child."""
    def require(self, _parent, step):
        return self.setdefault(step, _Node())

class _Node:
    """A single node of a trie."""
    def __init__(self):
        self.children = _EMPTY
        self.value = _EMPTY

    def iterate(self, path, shallow, items):
        node = self
        stack = []
        while True:
            if node.value is not _EMPTY:
                yield path, node.value
            if (not shallow or node.value is _EMPTY) and node.children:
                stack.append(iter(items(node.children)))
                path.append(None)
            while True:
                try:
                    step, node = next(stack[-1])
                    path[-1] = step
                    break
                except StopIteration:
                    stack.pop()
                    path.pop()
                except IndexError:
                    return

class Trie(MutableMapping):
    """A trie implementation with dict interface plus some extensions."""
    def __init__(self, *args, **kwargs):
        self._root = _Node()
        self.update(*args, **kwargs)

    def _get_node(self, key):
        node = self._root
        trace = [(None, node)]
        for step in self._path_from_key(key):
            node = node.children.get(step)
            if node is None:
                raise KeyError(key)
            trace.append((step, node))
        return node, trace

    def _set_node(self, key, value, only_if_missing=False):
        node = self._root
        for step in self._path_from_key(key):
            node = node.children.require(node, step)
        if node.value is _EMPTY or not only_if_missing:
            node.value = value
        return node


    def itervalues(self, prefix=_EMPTY, shallow=False):
        node, _ = self._get_node(prefix)
        for _, value in node.iterate(list(self._path_from_key(prefix)), shallow, lambda x: x.items()):
            yield value

    def values(self, prefix=_EMPTY, shallow=False):
        """Returns a list of values in the trie with the given prefix.

        Args:
            prefix: Prefix to limit the values returned.
            shallow: Perform a shallow traversal, i.e., do not yield values if
                     their prefix has been yielded.

        Returns:
            A list of all values that match the given prefix in the trie.
        """
        return list(self.itervalues(prefix=prefix, shallow=shallow))


    def _path_from_key(self, key):
        return key if key is not _EMPTY else ()

    def __setitem__(self, key, value):
        self._set_node(key, value)

    # Following methods aren't called but needed to conform with the MutableMapping base class
    def __len__(self): raise NotImplementedError
    def __iter__(self): raise NotImplementedError
    def __getitem__(self, key): raise NotImplementedError
    def __delitem__(self, key): raise NotImplementedError
