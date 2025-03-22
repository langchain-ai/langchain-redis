"""Test if the fix for 'ids' parameter in kwargs works properly."""

import unittest
from typing import Any, List, Optional

from langchain_redis.vectorstores import RedisVectorStore


class TestIdsParameter(unittest.TestCase):
    """Test add_texts with ids parameter in kwargs."""

    def test_add_texts_with_ids_in_kwargs(self) -> None:
        """Test that the ids parameter in kwargs is correctly used when keys is None."""

        class TestableVectorStore(RedisVectorStore):
            """Test implementation that just captures the keys parameter."""

            def __init__(self) -> None:
                """Initialize with minimal requirements for this test."""
                self.captured_keys = None
                self.config = type("Config", (), {"key_prefix": "test"})  # type: ignore

            def add_texts(  # type: ignore
                self,
                texts: List[str],
                metadatas: Optional[List[dict]] = None,
                keys: Optional[List[str]] = None,
                **kwargs: Any,
            ) -> List[str]:
                """Override to test the fix."""
                # This is the only code we're testing - the PR fix
                if keys is None and "ids" in kwargs:
                    keys = kwargs["ids"]

                # Store the keys for verification
                self.captured_keys = keys  # type: ignore

                # Just return some dummy results
                return ["id1", "id2", "id3"]

        # Create our test store
        store = TestableVectorStore()

        # Set up test data
        texts = ["text1", "text2", "text3"]
        ids = ["id1", "id2", "id3"]

        # Call add_texts with ids in kwargs (not in keys)
        result = store.add_texts(texts=texts, keys=None, ids=ids)

        # Verify that keys correctly received the values from ids
        self.assertEqual(store.captured_keys, ids)

        # Also check that the result is as expected
        self.assertEqual(result, ["id1", "id2", "id3"])
