import os
import subprocess

import pytest

try:
    from testcontainers.compose import DockerCompose  # type: ignore[import]

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

if 0 and TESTCONTAINERS_AVAILABLE:

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
