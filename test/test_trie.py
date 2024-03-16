import unittest
from tokenhealing import Trie

class TestTrie(unittest.TestCase):
    def setUp(self):
        self.trie = Trie()

    def test_insert_and_search(self):
        # Test insertion and search for a single key-value pair
        self.trie['moo'] = 0
        self.assertEqual(self.trie.extensions('moo'), [0])

    def test_prefix_search(self):
        # Test searching by prefix
        self.trie['foo'] = 1
        self.trie['food'] = 2
        self.trie['foodie'] = 3
        self.trie['helium'] = 4
        self.assertEqual(self.trie.extensions('foo'), [1, 2, 3])
        self.assertEqual(self.trie.extensions('helium'), [4])

    def test_empty_prefix(self):
        # Test searching with an empty prefix returns all values
        self.trie['hello'] = 5
        self.trie['bye'] = 6
        self.assertEqual(self.trie.extensions(''), [5, 6])

    def test_no_match(self):
        # Test searching for a prefix that doesn't match any key
        with self.assertRaises(KeyError):
            self.trie.extensions('unknown')

    def test_update_value(self):
        # Test updating the value of an existing key
        self.trie['hi'] = 7
        self.trie['hi'] = 8
        self.assertEqual(self.trie.extensions('hi'), [8])

if __name__ == '__main__':
    unittest.main()
