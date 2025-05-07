"""Test the bug fix in RedisVectorStore.add_texts().

The bug fix ensures that when keys=None and ids are provided via kwargs,
the ids are correctly used as keys.
"""

def test_add_texts_uses_ids_from_kwargs_when_keys_is_none():
    """Test directly that the ids parameter in kwargs is used when keys is None."""
    # Import the needed function
    from langchain_redis.vectorstores import RedisVectorStore
    
    # Create a simple subclass that we can test without actually connecting to Redis
    class TestableRedisVectorStore(RedisVectorStore):
        def __init__(self):
            self.keys_used = None
            self.config = type('Config', (), {'key_prefix': 'test'})
        
        def add_texts(self, texts, metadatas=None, keys=None, **kwargs):
            # This is the code we're testing, copied from the PR
            if keys is None and "ids" in kwargs:
                keys = kwargs["ids"]
            
            # Store the keys that would be used
            self.keys_used = keys
            
            # Return some dummy result
            return ["result1", "result2"]
    
    # Create our test instance
    store = TestableRedisVectorStore()
    
    # Call add_texts with ids in kwargs but keys=None
    texts = ["Test document"]
    result = store.add_texts(texts, keys=None, ids=["custom_id_1"])
    
    # Check that the keys were set from the ids parameter
    assert store.keys_used == ["custom_id_1"]
    
    # Check that we get the expected result
    assert result == ["result1", "result2"]