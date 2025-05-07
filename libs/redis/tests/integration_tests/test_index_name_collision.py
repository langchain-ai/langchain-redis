"""Test for Issue #31: Search results from different indexes if names are similar."""

import os

import pytest
from langchain_core.documents import Document

from langchain_redis import RedisVectorStore
from tests.integration_tests.embed_patch import get_embeddings_for_tests


@pytest.fixture
def redis_url() -> str:
    """Get Redis URL from environment variable or use default."""
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def test_similar_index_names(redis_url: str) -> None:
    """Test that similar index names don't cause search results to mix."""
    # Create random suffixes for index names to ensure they're unique for this test
    suffix1 = os.urandom(4).hex()
    suffix2 = os.urandom(4).hex()

    # Create two indexes with similar names (test_index and test_index_more)
    index_name1 = f"test_index_{suffix1}"
    index_name2 = f"test_index_more_{suffix2}"  # Similar prefix

    # Get embeddings for tests
    embeddings = get_embeddings_for_tests()

    # Create documents specific to each index
    docs1 = [
        Document(
            page_content="Document A for first index", metadata={"index": "first"}
        ),
        Document(
            page_content="Document B for first index", metadata={"index": "first"}
        ),
    ]

    docs2 = [
        Document(
            page_content="Document C for second index", metadata={"index": "second"}
        ),
        Document(
            page_content="Document D for second index", metadata={"index": "second"}
        ),
    ]

    # Create the first vector store and add documents
    vector_store1 = RedisVectorStore.from_documents(
        documents=docs1,
        embedding=embeddings,
        index_name=index_name1,
        redis_url=redis_url,
    )

    # Create the second vector store and add documents
    vector_store2 = RedisVectorStore.from_documents(
        documents=docs2,
        embedding=embeddings,
        index_name=index_name2,
        redis_url=redis_url,
    )

    # Verify that documents were added to each index
    assert verify_document_count(vector_store1) == 2
    assert verify_document_count(vector_store2) == 2

    # Search in the first index - should only return documents from the first index
    results1 = vector_store1.similarity_search(query="Document", k=10)
    assert len(results1) == 2
    # Check that all results are from the first index
    for doc in results1:
        assert doc.metadata.get("index") == "first"

    # Check if the results contain the expected documents from the first index
    result_contents1 = [doc.page_content for doc in results1]
    assert any("Document A for first index" in content for content in result_contents1)
    assert any("Document B for first index" in content for content in result_contents1)
    assert not any("second index" in content for content in result_contents1)

    # Search in the second index - should only return documents from the second index
    results2 = vector_store2.similarity_search(query="Document", k=10)
    assert len(results2) == 2
    # Check that all results are from the second index
    for doc in results2:
        assert doc.metadata.get("index") == "second"

    # Check if the results contain the expected documents from the second index
    result_contents2 = [doc.page_content for doc in results2]
    assert any("Document C for second index" in content for content in result_contents2)
    assert any("Document D for second index" in content for content in result_contents2)
    assert not any("first index" in content for content in result_contents2)

    # Clean up
    vector_store1.index.delete(drop=True)
    vector_store2.index.delete(drop=True)


def verify_document_count(vector_store: RedisVectorStore) -> int:
    """Count the number of documents in the vector store by doing a wide search."""
    # Use "Document" as query since it appears in most test documents
    results = vector_store.similarity_search(query="Document", k=100)
    return len(results)


def test_index_namespace_isolation(redis_url: str) -> None:
    """Test that indexes with same prefixes but different names are isolated."""
    # Create random suffixes for index names to ensure they're unique for this test
    suffix = os.urandom(4).hex()

    # Create two indexes with the same prefix but different names
    index_prefix = f"test_prefix_{suffix}"
    index_name1 = f"test_index1_{suffix}"
    index_name2 = f"test_index2_{suffix}"

    # Get embeddings for tests
    embeddings = get_embeddings_for_tests()

    # Create documents specific to each index
    docs1 = [
        Document(
            page_content="Unique document X for first index",
            metadata={"index": "first"},
        ),
    ]

    docs2 = [
        Document(
            page_content="Unique document Y for second index",
            metadata={"index": "second"},
        ),
    ]

    # Create the first vector store and add documents
    vector_store1 = RedisVectorStore.from_documents(
        documents=docs1,
        embedding=embeddings,
        index_name=index_name1,
        key_prefix=index_prefix,  # Same prefix for both
        redis_url=redis_url,
    )

    # Create the second vector store and add documents
    vector_store2 = RedisVectorStore.from_documents(
        documents=docs2,
        embedding=embeddings,
        index_name=index_name2,
        key_prefix=index_prefix,  # Same prefix for both
        redis_url=redis_url,
    )

    # Verify each index has its own document
    results1 = vector_store1.similarity_search(query="Unique", k=10)
    assert len(results1) == 1
    assert results1[0].metadata.get("index") == "first"
    assert "document X" in results1[0].page_content

    results2 = vector_store2.similarity_search(query="Unique", k=10)
    assert len(results2) == 1
    assert results2[0].metadata.get("index") == "second"
    assert "document Y" in results2[0].page_content

    # Clean up
    vector_store1.index.delete(drop=True)
    vector_store2.index.delete(drop=True)
