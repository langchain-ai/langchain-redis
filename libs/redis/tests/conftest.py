import os
import subprocess

import pytest

try:
    from testcontainers.compose import DockerCompose  # type: ignore[import]

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

if TESTCONTAINERS_AVAILABLE:

    @pytest.fixture(scope="session", autouse=True)
    def redis_container() -> DockerCompose:
        # Set the default Redis version if not already set
        os.environ.setdefault("REDIS_VERSION", "edge")

        try:
            compose = DockerCompose(
                "tests", compose_file_name="docker-compose.yml", pull=True
            )
            compose.start()

            redis_host, redis_port = compose.get_service_host_and_port("redis", 6379)
            redis_url = f"redis://{redis_host}:{redis_port}"
            os.environ["REDIS_URL"] = redis_url

            yield compose

            compose.stop()
        except subprocess.CalledProcessError:
            yield None


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
