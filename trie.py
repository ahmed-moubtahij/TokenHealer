# -*- coding: utf-8 -*-
"""Pure Python implementation of a trie data structure compatible with Python
2.x and Python 3.x.

`Trie data structure <http://en.wikipedia.org/wiki/Trie>`_, also known as radix
or prefix tree, is a tree associating keys to values where all the descendants
of a node have a common prefix (associated with that node).

The trie module contains :class:`pygtrie.Trie`, :class:`pygtrie.CharTrie` and
:class:`pygtrie.StringTrie` classes each implementing a mutable mapping
interface, i.e. :class:`dict` interface.  As such, in most circumstances,
:class:`pygtrie.Trie` could be used as a drop-in replacement for
a :class:`dict`, but the prefix nature of the data structure is trie’s real
strength.

The module also contains :class:`pygtrie.PrefixSet` class which uses a trie to
store a set of prefixes such that a key is contained in the set if it or its
prefix is stored in the set.

Features
--------

- A full mutable mapping implementation.

- Supports iterating over as well as deleting of a branch of a trie
  (i.e. subtrie)

- Supports prefix checking as well as shortest and longest prefix
  look-up.

- Extensible for any kind of user-defined keys.

- A PrefixSet supports “all keys starting with given prefix” logic.

- Can store any value including None.

For a few simple examples see ``example.py`` file.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Michal Nazarewicz <mina86@mina86.com>'
__copyright__ = ('Copyright 2014-2017 Google LLC',
                 'Copyright 2018-2020 Michal Nazarewicz <mina86@mina86.com>')
__version__ = '2.5.0'


import collections.abc as _abc


class ShortKeyError(KeyError):
    """Raised when given key is a prefix of an existing longer key
    but does not have a value associated with itself."""


class _NoChildren:
    """Collection representing lack of any children.

    Also acts as an empty iterable and an empty iterator.  This isn’t the
    cleanest designs but it makes various things more concise and avoids object
    allocations in a few places.

    Don’t create objects of this type directly; instead use _EMPTY singleton.
    """
    __slots__ = ()

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    def add(self, parent, step):
        node = _Node()
        parent.children = _OneChild(step, node)
        return node

    require = add

    # delete is not implemented on purpose since it should never be called on
    # a node with no children.


_EMPTY = _NoChildren()


class _OneChild:
    """Children collection representing a single child."""

    __slots__ = ('step', 'node')

    def __init__(self, step, node):
        self.step = step
        self.node = node

    def __bool__(self):
        return True
    __nonzero__ = __bool__

    def iteritems(self):
        return iter(((self.step, self.node),))

    def get(self, step):
        return self.node if step == self.step else None

    def add(self, parent, step):
        node = _Node()
        parent.children = _Children((self.step, self.node), (step, node))
        return node

    def require(self, parent, step):
        return self.node if self.step == step else self.add(parent, step)


class _Children(dict):
    """Children collection representing more than one child."""

    __slots__ = ()

    def __init__(self, *items):
        super(_Children, self).__init__(items)

    def iteritems(self):
        return iter(self.items())

    def require(self, _parent, step):
        return self.setdefault(step, _Node())


class _Node:
    """A single node of a trie.

    Stores value associated with the node and dictionary of children.
    """
    __slots__ = ('children', 'value')

    def __init__(self):
        self.children = _EMPTY
        self.value = _EMPTY

    def iterate(self, path, shallow, iteritems):
        """Yields all the nodes with values associated to them in the trie.

        Args:
            path: Path leading to this node.  Used to construct the key when
                returning value of this node and as a prefix for children.
            shallow: Perform a shallow traversal, i.e. do not yield nodes if
                their prefix has been yielded.
            iteritems: A callable taking ``node.children`` as sole argument and
                returning an iterable of children as ``(step, node)`` pair.  The
                callable would typically call ``iteritems`` or ``sorted_items``
                method on the argument depending on whether sorted output is
                desired.

        Yields:
            ``(path, value)`` tuples.
        """
        # Use iterative function with stack on the heap so we don't hit Python's
        # recursion depth limits.
        node = self
        stack = []
        while True:
            if node.value is not _EMPTY:
                yield path, node.value

            if (not shallow or node.value is _EMPTY) and node.children:
                stack.append(iter(iteritems(node.children)))
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

    __bool__ = __nonzero__ = __hash__ = None


class Trie(_abc.MutableMapping):
    """A trie implementation with dict interface plus some extensions.

    Keys used with the :class:`pygtrie.Trie` class must be iterable which each
    component being a hashable objects.  In other words, for a given key,
    ``dict.fromkeys(key)`` must be valid expression.

    In particular, strings work well as trie keys, however when getting them
    back (for example via :func:`Trie.iterkeys` method), instead of strings,
    tuples of characters are produced.  For that reason,
    :class:`pygtrie.CharTrie` or :class:`pygtrie.StringTrie` classes may be
    preferred when using string keys.
    """

    def __init__(self, *args, **kwargs):
        """Initialises the trie.

        Arguments are interpreted the same way :func:`Trie.update` interprets
        them.
        """
        self._root = _Node()
        self._iteritems = self._ITERITEMS_CALLBACKS[0]
        self.update(*args, **kwargs)

    _ITERITEMS_CALLBACKS = (lambda x: x.iteritems(), lambda x: x.sorted_items())

    def _get_node(self, key):
        """Returns node for given key.  Creates it if requested.

        Args:
            key: A key to look for.

        Returns:
            ``(node, trace)`` tuple where ``node`` is the node for given key and
            ``trace`` is a list specifying path to reach the node including all
            the encountered nodes.  Each element of trace is a ``(step, node)``
            tuple where ``step`` is a step from parent node to given node and
            ``node`` is node on the path.  The first element of the path is
            always ``(None, self._root)``.

        Raises:
            KeyError: If there is no node for the key.
        """
        node = self._root
        trace = [(None, node)]
        for step in self.__path_from_key(key):
            # pylint thinks node.children is always _NoChildren and thus that
            # we’re assigning None here; pylint: disable=assignment-from-none
            node = node.children.get(step)
            if node is None:
                raise KeyError(key)
            trace.append((step, node))
        return node, trace

    def _set_node(self, key, value, only_if_missing=False):
        """Sets value for a given key.

        Args:
            key: Key to set value of.
            value: Value to set to.
            only_if_missing: If true, value won't be changed if the key is
                    already associated with a value.

        Returns:
            The node.
        """
        node = self._root
        for step in self.__path_from_key(key):
            node = node.children.require(node, step)
        if node.value is _EMPTY or not only_if_missing:
            node.value = value
        return node

    def __iter__(self):
        return self.iterkeys()

    # pylint: disable=arguments-differ

    def itervalues(self, prefix=_EMPTY, shallow=False):
        """Yields all values associated with keys with given prefix.

        This is equivalent to taking second element of tuples generated by
        :func:`Trie.iteritems` which see for more detailed documentation.

        Args:
            prefix: Prefix to limit iteration to.
            shallow: Perform a shallow traversal, i.e. do not yield values if
                their prefix has been yielded.

        Yields:
            All the values associated with keys (with given prefix) in the trie.

        Raises:
            KeyError: If ``prefix`` does not match any node.
        """
        node, _ = self._get_node(prefix)
        for _, value in node.iterate(list(self.__path_from_key(prefix)),
                                     shallow, self._iteritems):
            yield value

    def values(self, prefix=_EMPTY, shallow=False):
        """Returns a list of values in given subtrie.

        This is equivalent to constructing a list from generator returned by
        :func:`Trie.itervalues` which see for more detailed documentation.
        """
        return list(self.itervalues(prefix=prefix, shallow=shallow))

    def __len__(self):
        """Returns number of values in a trie.

        Note that this method is expensive as it iterates over the whole trie.
        """
        return sum(1 for _ in self.itervalues())

    __hash__ = None

    HAS_VALUE = 1
    HAS_SUBTRIE = 2

    @staticmethod
    def _slice_maybe(key_or_slice):
        """Checks whether argument is a slice or a plain key.

        Args:
            key_or_slice: A key or a slice to test.

        Returns:
            ``(key, is_slice)`` tuple.  ``is_slice`` indicates whether
            ``key_or_slice`` is a slice and ``key`` is either ``key_or_slice``
            itself (if it's not a slice) or slice's start position.

        Raises:
            TypeError: If ``key_or_slice`` is a slice whose stop or step are not
                ``None`` In other words, only ``[key:]`` slices are valid.
        """
        return key_or_slice, False

    def __getitem__(self, key_or_slice):
        """Returns value associated with given key or raises KeyError.

        When argument is a single key, value for that key is returned (or
        :class:`KeyError` exception is thrown if the node does not exist or has
        no value associated with it).

        When argument is a slice, it must be one with only `start` set in which
        case the access is identical to :func:`Trie.itervalues` invocation with
        prefix argument.

        Example:

            >>> import pygtrie
            >>> t = pygtrie.StringTrie()
            >>> t['foo/bar'] = 'Bar'
            >>> t['foo/baz'] = 'Baz'
            >>> t['qux'] = 'Qux'
            >>> t['foo/bar']
            'Bar'
            >>> sorted(t['foo':])
            ['Bar', 'Baz']
            >>> t['foo']  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ShortKeyError: 'foo'

        Args:
            key_or_slice: A key or a slice to look for.

        Returns:
            If a single key is passed, a value associated with given key.  If
            a slice is passed, a generator of values in specified subtrie.

        Raises:
            ShortKeyError: If the key has no value associated with it but is
                a prefix of some key with a value.  Note that
                :class:`ShortKeyError` is subclass of :class:`KeyError`.
            KeyError: If key has no value associated with it nor is a prefix of
                an existing key.
            TypeError: If ``key_or_slice`` is a slice but it's stop or step are
                not ``None``.
        """
        if self._slice_maybe(key_or_slice)[1]:
            return self.itervalues(key_or_slice.start)
        node, _ = self._get_node(key_or_slice)
        if node.value is _EMPTY:
            raise ShortKeyError(key_or_slice)
        return node.value

    def __setitem__(self, key_or_slice, value):
        """Sets value associated with given key.

        If `key_or_slice` is a key, simply associate it with given value.  If it
        is a slice (which must have `start` set only), it in addition clears any
        subtrie that might have been attached to particular key.  For example::

            >>> import pygtrie
            >>> t = pygtrie.StringTrie()
            >>> t['foo/bar'] = 'Bar'
            >>> t['foo/baz'] = 'Baz'
            >>> sorted(t.keys())
            ['foo/bar', 'foo/baz']
            >>> t['foo':] = 'Foo'
            >>> t.keys()
            ['foo']

        Args:
            key_or_slice: A key to look for or a slice.  If it is a slice, the
                whole subtrie (if present) will be replaced by a single node
                with given value set.
            value: Value to set.

        Raises:
            TypeError: If key is a slice whose stop or step are not None.
        """
        key, is_slice = self._slice_maybe(key_or_slice)
        node = self._set_node(key, value)
        if is_slice:
            node.children = _EMPTY

    def __delitem__(self, key_or_slice):
        """Deletes value associated with given key or raises KeyError.

        If argument is a key, value associated with it is deleted.  If the key
        is also a prefix, its descendents are not affected.  On the other hand,
        if the argument is a slice (in which case it must have only start set),
        the whole subtrie is removed.  For example::

            >>> import pygtrie
            >>> t = pygtrie.StringTrie()
            >>> t['foo'] = 'Foo'
            >>> t['foo/bar'] = 'Bar'
            >>> t['foo/bar/baz'] = 'Baz'
            >>> del t['foo/bar']
            >>> t.keys()
            ['foo', 'foo/bar/baz']
            >>> del t['foo':]
            >>> t.keys()
            []

        Args:
            key_or_slice: A key to look for or a slice.  If key is a slice, the
                    whole subtrie will be removed.

        Raises:
            ShortKeyError: If the key has no value associated with it but is
                a prefix of some key with a value.  This is not thrown if
                key_or_slice is a slice -- in such cases, the whole subtrie is
                removed.  Note that :class:`ShortKeyError` is subclass of
                :class:`KeyError`.
            KeyError: If key has no value associated with it nor is a prefix of
                an existing key.
            TypeError: If key is a slice whose stop or step are not ``None``.
        """
        key, is_slice = self._slice_maybe(key_or_slice)
        node, trace = self._get_node(key)
        if is_slice:
            node.children = _EMPTY
        elif node.value is _EMPTY:
            raise ShortKeyError(key)
        self._pop_value(trace)

    def __path_from_key(self, key):
        """Converts a user visible key object to internal path representation.

        Args:
            key: User supplied key or ``_EMPTY``.

        Returns:
            An empty tuple if ``key`` was ``_EMPTY``, otherwise whatever
            :func:`Trie._path_from_key` returns.

        Raises:
            TypeError: If ``key`` is of invalid type.
        """
        return () if key is _EMPTY else self._path_from_key(key)

    def _path_from_key(self, key):
        """Converts a user visible key object to internal path representation.

        The default implementation simply returns key.

        Args:
            key: User supplied key.

        Returns:
            A path, which is an iterable of steps.  Each step must be hashable.

        Raises:
            TypeError: If key is of invalid type.
        """
        return key
