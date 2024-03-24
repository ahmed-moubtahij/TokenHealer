import unittest
from tokenhealing import Trie

class TestTrie(unittest.TestCase):
    def setUp(self):
        self.trie = Trie()

    def test_insert_and_search(self):
        # Test insertion and search for a single key-value pair
        self.trie.add('moo')
        self.assertEqual(self.trie.extensions('moo'), ['moo'])

    def test_prefix_search(self):
        # Test searching by prefix
        self.trie.add('foo')
        self.trie.add('food')
        self.trie.add('foodie')
        self.trie.add('helium')
        self.assertEqual(
            self.trie.extensions('foo'),
            ['foo', 'food', 'foodie']
        )
        self.assertEqual(self.trie.extensions('helium'), ['helium'])

    def test_empty_prefix(self):
        # Test searching with an empty prefix returns all values
        self.trie.add('hello')
        self.trie.add('bye')
        self.assertEqual(self.trie.extensions(''), ['hello', 'bye'])

    def test_no_match(self):
        # Test searching for a prefix that doesn't match any key
        with self.assertRaises(KeyError):
            self.trie.extensions('unknown')

    def test_update_value(self):
        # Test updating the value of an existing key
        self.trie.add('hi')
        self.trie.add('hi')
        self.assertEqual(self.trie.extensions('hi'), ['hi'])

if __name__ == '__main__':
    unittest.main()
