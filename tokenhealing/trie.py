""" Testing/modifying HF's token-id-unaware Trie """

class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self, *args):
        self.data = {}
        self._tokens = set()
        self.update(*args)

    def update(self, *args):
        for token in tuple(*args):
            self.add(token)

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```
        """
        if not word:
            # Prevent empty string
            return

        self._tokens.add(word)
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            # ref[char] = ref.setdefault(char, {}) # NOTE: run tests with this instead
            ref = ref[char]
        ref[""] = 1

    def extensions(self, prefix: str):
        """Retrieve tokens starting with a given prefix."""
        prefix_node = self.data
        for char in prefix:
            prefix_node = prefix_node[char]
        ret = self._collect_tokens(prefix_node)
        ret = [prefix + token for token in ret]
        return ret

    def _collect_tokens(self, node: dict) -> list:
        """Recursively collect all tokens under the given node."""
        tokens = [""] if "" in node else []
        for token, subtrie_head in node.items():
            if token != "":
                tokens.extend(
                    [token + subtoken for subtoken in self._collect_tokens(subtrie_head)]
                )
        return tokens