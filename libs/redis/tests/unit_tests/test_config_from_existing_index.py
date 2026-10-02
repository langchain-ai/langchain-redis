from unittest.mock import MagicMock

from redis import Redis

from langchain_redis import RedisConfig


def test_from_existing_index_marks_config_as_existing() -> None:
    """The config must route the vector store to the existing index."""
    config = RedisConfig.from_existing_index("my_existing_index", MagicMock(spec=Redis))

    assert config.index_name == "my_existing_index"
    assert config.from_existing is True


def test_default_config_is_not_from_existing() -> None:
    assert RedisConfig(index_name="fresh").from_existing is False


def test_from_existing_index_keeps_the_redis_client() -> None:
    """The client passed in must be the one the config hands back."""
    client = MagicMock(spec=Redis)

    config = RedisConfig.from_existing_index("my_existing_index", client)

    assert config.redis_client is client
    assert config.redis() is client
