"""Test for Issue #62: Logs reset to INFO when importing RedisChatMessageHistory."""

import io
import logging
import os
from contextlib import redirect_stdout


def test_logging_reset() -> None:
    """Test that demonstrates logs being reset to INFO level when importing
    RedisChatMessageHistory."""
    # Reset logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up a root logger with debug level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Create a custom logger that would be affected
    test_logger = logging.getLogger("test_app_logger")
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = True

    # Create a stream handler to capture output
    console_output = io.StringIO()
    handler = logging.StreamHandler(console_output)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log a debug message - this should work
    test_logger.debug("Debug message before import")

    # Check if a push_response logger exists yet
    [name for name in logging.root.manager.loggerDict.keys() if "push" in name.lower()]

    # Check the level
    before_level = root_logger.level
    test_logger_before_level = test_logger.level

    # Import Redis module to see if it creates the push_response logger
    with redirect_stdout(io.StringIO()):  # Suppress any print statements during import
        import redis

    # Check if a push_response logger exists after importing Redis
    [name for name in logging.root.manager.loggerDict.keys() if "push" in name.lower()]

    # Now import RedisChatMessageHistory, which internally uses Redis
    with redirect_stdout(io.StringIO()):  # Suppress any print statements during import
        from langchain_redis import RedisChatMessageHistory

    # Check if a push_response logger exists now
    [name for name in logging.root.manager.loggerDict.keys() if "push" in name.lower()]

    # Let's actually create an instance and use it
    with redirect_stdout(io.StringIO()):  # Suppress any print statements during usage
        try:
            # Get Redis URL from environment variable or use default
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

            # Creating a RedisChatMessageHistory instance
            history = RedisChatMessageHistory(
                session_id="test_session",
                redis_url=redis_url,
                ttl=60,
            )

            # This should trigger Redis client operations
            from langchain_core.messages import HumanMessage

            history.add_message(HumanMessage(content="Hello"))

            # Now let's try PubSub which should trigger _set_info_logger
            import redis

            client = redis.Redis.from_url(redis_url)
            client.pubsub()
        except Exception:
            pass

    # Check if push_response was created after using RedisChatMessageHistory
    has_push_response_after_use = "push_response" in logging.root.manager.loggerDict
    [name for name in logging.root.manager.loggerDict.keys() if "push" in name.lower()]

    # If push_response logger exists, check its level
    push_response_level = None
    if has_push_response_after_use:
        push_response_logger = logging.getLogger("push_response")
        push_response_level = push_response_logger.level

    # Check the level again
    after_level = root_logger.level
    test_logger_after_level = test_logger.level

    # Examine all loggers and their levels
    {
        name: logging.getLogger(name).level
        for name in logging.root.manager.loggerDict.keys()
    }

    # Log another debug message
    test_logger.debug("Debug message after import")

    # Get the captured output
    output = console_output.getvalue()

    # The actual test - verify that debug messages are still working
    assert (
        "DEBUG:test_app_logger:Debug message before import" in output
    ), "First debug message not logged"
    assert (
        "DEBUG:test_app_logger:Debug message after import" in output
    ), "Second debug message not logged"

    # Directly call the function that creates the push_response logger
    from redis.utils import _set_info_logger

    _set_info_logger()

    # Check if push_response logger was created after direct call
    has_push_response_after_direct = "push_response" in logging.root.manager.loggerDict
    assert (
        has_push_response_after_direct
    ), "The push_response logger was not created even after direct call"

    # Get the push_response logger and check its level
    push_response_logger = logging.getLogger("push_response")
    push_response_level = push_response_logger.level

    # If it was created, check that it's at INFO level
    assert (
        push_response_level == logging.INFO
    ), f"Push response logger level is {push_response_level}, expected {logging.INFO}"

    # Make sure the root logger wasn't changed
    assert (
        before_level == after_level
    ), f"Root logger level changed from {before_level} to {after_level}"

    # Make sure our test logger wasn't changed
    assert test_logger_before_level == test_logger_after_level, (
        f"Test logger level changed from {test_logger_before_level} "
        f"to {test_logger_after_level}"
    )

    # The crucial part - check for debug log after setting push_response logger to INFO
    test_logger.debug("Debug message after push_response created")
    output_after = console_output.getvalue()
    assert (
        "Debug message after push_response created" in output_after
    ), "Debug messages stopped working after push_response logger created"
