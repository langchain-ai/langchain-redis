from redis import Redis

from langchain_redis import RedisConfig


def test_from_existing_index_marks_config_as_existing() -> None:
    """The config must route the vector store to the existing index."""
    # A real client, not a mock: `redis_client` is validated as a `Redis`
    # instance, and `Redis.from_url` does not connect until first use.
    config = RedisConfig.from_existing_index(
        "my_existing_index", Redis.from_url("redis://localhost:6379")
    )

    assert config.index_name == "my_existing_index"
    assert config.from_existing is True


def test_from_existing_index_keeps_the_supplied_client() -> None:
    """The client the caller passed must be the one the config hands back.

    Dropping it silently fell back to the default ``redis_url``, so a store
    built from the config talked to ``localhost:6379`` instead of the server
    the caller supplied.
    """
    client = Redis.from_url("redis://prod-redis.internal:6380")

    config = RedisConfig.from_existing_index("my_existing_index", client)

    assert config.redis_client is client
    assert config.redis() is client


def test_default_config_is_not_from_existing() -> None:
    assert RedisConfig(index_name="fresh").from_existing is False
