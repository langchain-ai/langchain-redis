"""Test key prefix handling consistency."""

from unittest.mock import Mock, patch

from langchain_redis.config import RedisConfig


def test_config_key_prefix_behavior():
    """Test that RedisConfig preserves key_prefix exactly as provided."""
    
    # Test case 1: key_prefix without trailing colon
    config1 = RedisConfig(
        index_name="test_index",
        key_prefix="my_prefix",
        embedding_dimensions=3
    )
    schema1 = config1.to_index_schema()
    assert schema1.index.prefix == "my_prefix"
    
    # Test case 2: key_prefix with trailing colon
    config2 = RedisConfig(
        index_name="test_index", 
        key_prefix="my_prefix:",
        embedding_dimensions=3
    )
    schema2 = config2.to_index_schema()
    assert schema2.index.prefix == "my_prefix:"
    
    # Test case 3: default key_prefix (should be index_name)
    config3 = RedisConfig(
        index_name="test_index",
        embedding_dimensions=3
    )
    schema3 = config3.to_index_schema()
    assert config3.key_prefix == "test_index"
    assert schema3.index.prefix == "test_index"


def test_vectorstore_prefix_issue():
    """Test that demonstrates the inconsistency in RedisVectorStore prefix handling."""
    from langchain_redis.vectorstores import RedisVectorStore
    
    # Mock embeddings and Redis to avoid external dependencies
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    
    config = RedisConfig(
        index_name="test_index",
        key_prefix="my_prefix",
        embedding_dimensions=3,
        redis_url="redis://localhost:6379"
    )
    
    # Check what RedisConfig thinks the prefix should be
    schema_from_config = config.to_index_schema()
    assert schema_from_config.index.prefix == "my_prefix"
    
    # Mock the SearchIndex to capture what gets passed to it
    with patch('langchain_redis.vectorstores.SearchIndex') as mock_search_index:
        with patch('redis.Redis') as mock_redis:
            # Mock Redis connection
            mock_redis_instance = Mock()
            mock_redis.from_url.return_value = mock_redis_instance
            
            # Create the vector store
            RedisVectorStore(embeddings=mock_embeddings, config=config)
            
            # Check what was passed to SearchIndex.from_dict
            mock_search_index.from_dict.assert_called_once()
            call_args = mock_search_index.from_dict.call_args[0][0]
            
            # The issue: RedisVectorStore adds an extra ':'
            actual_prefix = call_args["index"]["prefix"]
            expected_prefix_from_config = config.key_prefix
            
            # This test should now pass - the issue is fixed
            # The vectorstore should use the prefix exactly as provided
            assert actual_prefix == expected_prefix_from_config, (
                f"Expected '{expected_prefix_from_config}', got '{actual_prefix}'"
            )


def test_vectorstore_prefix_with_colon():
    """Test behavior when key_prefix already has a colon."""
    from langchain_redis.vectorstores import RedisVectorStore
    
    # Mock embeddings and Redis to avoid external dependencies  
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    
    config = RedisConfig(
        index_name="test_index",
        key_prefix="my_prefix:",  # Already has colon
        embedding_dimensions=3,
        redis_url="redis://localhost:6379"
    )
    
    with patch('langchain_redis.vectorstores.SearchIndex') as mock_search_index:
        with patch('redis.Redis') as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.from_url.return_value = mock_redis_instance
            
            RedisVectorStore(embeddings=mock_embeddings, config=config)
            
            call_args = mock_search_index.from_dict.call_args[0][0]
            actual_prefix = call_args["index"]["prefix"] 
            
            # This should now be fixed - no double colon
            assert actual_prefix == "my_prefix:", f"Got: '{actual_prefix}'"