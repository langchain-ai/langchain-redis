from unittest.mock import MagicMock

from langchain_redis import RedisConfig


def test_from_existing_index_marks_config_as_existing() -> None:
    """The config must route the vector store to the existing index."""
    config = RedisConfig.from_existing_index("my_existing_index", MagicMock())

    assert config.index_name == "my_existing_index"
    assert config.from_existing is True


def test_default_config_is_not_from_existing() -> None:
    assert RedisConfig(index_name="fresh").from_existing is False
