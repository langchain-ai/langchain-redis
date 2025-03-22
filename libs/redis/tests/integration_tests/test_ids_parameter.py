"""Test that ids parameter in kwargs works correctly in add_texts."""

import os

import pytest
from redis import Redis

from langchain_redis import RedisVectorStore
from tests.integration_tests.embed_patch import get_embeddings_for_tests


@pytest.fixture
def redis_url() -> str:
    """Get Redis URL from environment variable or use default."""
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def test_add_texts_with_ids_in_kwargs(redis_url: str) -> None:
    """Test that ids parameter in kwargs works correctly in add_texts."""
    # Create embeddings for tests
    embeddings = get_embeddings_for_tests()  # type: ignore

    # Create a unique index name for testing
    index_name = f"test_ids_parameter_{os.urandom(4).hex()}"
    key_prefix = "test_prefix"

    # Create Redis vector store
    vector_store = RedisVectorStore(
        embeddings=embeddings,
        index_name=index_name,
        redis_url=redis_url,
        key_prefix=key_prefix,
    )

    # Test texts and IDs
    texts = ["foo", "bar", "baz"]
    ids = ["id1", "id2", "id3"]

    # Add texts with ids parameter in kwargs
    returned_ids = vector_store.add_texts(texts=texts, ids=ids)  # type: ignore

    # Verify returned IDs match expected format
    assert len(returned_ids) == len(ids)

    # Check if keys in Redis have the expected format
    redis_client = Redis.from_url(redis_url)

    for id_val in ids:
        expected_key = f"{key_prefix}:{id_val}"
        assert redis_client.exists(
            expected_key
        ), f"Key {expected_key} not found in Redis"

    # Clean up after test
    vector_store.index.delete(drop=True)


def test_add_texts_with_both_keys_and_ids(redis_url: str) -> None:
    """Test that keys parameter takes precedence over ids in kwargs."""
    # Create embeddings for tests
    embeddings = get_embeddings_for_tests()  # type: ignore

    # Create a unique index name for testing
    index_name = f"test_ids_parameter_{os.urandom(4).hex()}"
    key_prefix = "test_prefix"

    # Create Redis vector store
    vector_store = RedisVectorStore(
        embeddings=embeddings,
        index_name=index_name,
        redis_url=redis_url,
        key_prefix=key_prefix,
    )

    # Test texts, keys and IDs
    texts = ["foo", "bar", "baz"]
    explicit_keys = ["key1", "key2", "key3"]
    ids_in_kwargs = ["id1", "id2", "id3"]

    # Add texts with both keys and ids parameters
    # keys parameter should take precedence
    returned_ids = vector_store.add_texts(
        texts=texts,
        keys=explicit_keys,
        ids=ids_in_kwargs,  # type: ignore
    )

    # Verify returned IDs match expected format from explicit keys
    assert len(returned_ids) == len(explicit_keys)

    # Check if keys in Redis have the expected format from explicit keys
    redis_client = Redis.from_url(redis_url)

    for key in explicit_keys:
        expected_key = f"{key_prefix}:{key}"
        assert redis_client.exists(
            expected_key
        ), f"Key {expected_key} not found in Redis"

    # Verify ids from kwargs were NOT used (keys takes precedence)
    for id_val in ids_in_kwargs:
        unexpected_key = f"{key_prefix}:{id_val}"
        if unexpected_key not in [f"{key_prefix}:{key}" for key in explicit_keys]:
            assert not redis_client.exists(
                unexpected_key
            ), f"Key {unexpected_key} found in Redis but should not be present"

    # Clean up after test
    vector_store.index.delete(drop=True)


def test_upsert_with_ids(redis_url: str) -> None:
    """Test updating existing documents with the same ids."""
    # Create embeddings for tests
    embeddings = get_embeddings_for_tests()  # type: ignore

    # Create a unique index name for testing
    index_name = f"test_ids_parameter_{os.urandom(4).hex()}"
    key_prefix = "test_prefix"

    # Create Redis vector store
    vector_store = RedisVectorStore(
        embeddings=embeddings,
        index_name=index_name,
        redis_url=redis_url,
        key_prefix=key_prefix,
    )

    # Test texts and IDs
    texts_original = ["foo", "bar", "baz"]
    ids = ["id1", "id2", "id3"]

    # Add texts with ids parameter in kwargs
    vector_store.add_texts(texts=texts_original, ids=ids)  # type: ignore

    # Now update the texts with the same IDs
    texts_updated = ["foo updated", "bar updated", "baz updated"]
    vector_store.add_texts(texts=texts_updated, ids=ids)  # type: ignore

    # Search for updated content
    results = vector_store.similarity_search("foo updated", k=1)
    assert len(results) == 1
    assert results[0].page_content == "foo updated"

    # Check that the number of keys hasn't increased (we're updating, not adding)
    redis_client = Redis.from_url(redis_url)
    key_count = 0
    for key in redis_client.scan_iter(f"{key_prefix}:*"):
        key_count += 1

    assert key_count == len(ids), f"Expected {len(ids)} keys but found {key_count}"

    # Clean up after test
    vector_store.index.delete(drop=True)
