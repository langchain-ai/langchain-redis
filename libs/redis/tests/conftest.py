import os
from typing import Generator

import pytest

try:
    from testcontainers.core.container import DockerContainer  # type: ignore[import]
    from testcontainers.core.wait_strategies import (  # type: ignore[import]
        LogMessageWaitStrategy,
    )

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

if TESTCONTAINERS_AVAILABLE:

    @pytest.fixture(scope="session", autouse=True)
    def redis_container() -> Generator[DockerContainer, None, None]:
        # Set the default Redis version if not already set
        redis_version = os.environ.get("REDIS_VERSION", "edge")
        redis_image = f"redis/redis-stack:{redis_version}"

        # Use DockerContainer with explicit wait strategy instead of RedisContainer
        # to avoid deprecated @wait_container_is_ready decorator
        container = (
            DockerContainer(redis_image)
            .with_exposed_ports(6379)
            .with_env("REDIS_ARGS", "--save '' --appendonly no")
            .waiting_for(
                LogMessageWaitStrategy(
                    "Ready to accept connections"
                ).with_startup_timeout(30)
            )
        )
        container.start()

        redis_host = container.get_container_host_ip()
        redis_port = container.get_exposed_port(6379)
        redis_url = f"redis://{redis_host}:{redis_port}"
        os.environ["REDIS_URL"] = redis_url

        yield container

        container.stop()


@pytest.fixture(scope="session")
def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379")


@pytest.fixture(scope="session", autouse=True)
def setup_openai_api_key() -> None:
    """Set up the OpenAI API key for tests if not already set."""
    # If not already set in the environment, use a default value for testing
    if "OPENAI_API_KEY" not in os.environ:
        # This would ideally come from a secrets store or CI/CD environment
        api_key = os.getenv("OPENAI_API_KEY_FOR_TESTS")

        # If we have a key in the alternate env var, use it
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
