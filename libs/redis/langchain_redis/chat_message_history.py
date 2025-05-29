import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from redis import Redis
from redis.exceptions import ResponseError
from redisvl.index import SearchIndex  # type: ignore
from redisvl.query import CountQuery, FilterQuery, TextQuery  # type: ignore
from redisvl.query.filter import Tag  # type: ignore
from ulid import ULID

from langchain_redis.version import __full_lib_name__


def _noop_push_handler(response: Any) -> None:
    """
    No-op push response handler to prevent _set_info_logger from being called.

    Redis's PubSub functionality creates a special INFO level logger called
    'push_response' when no handler is provided. This affects global logging config
    by creating a new INFO level logger while an app might be using DEBUG level.

    This handler simply does nothing with the push responses, preventing Redis from
    creating its own INFO logger. If an app needs to process push responses,
    it should provide its own custom handler when instantiating RedisChatMessageHistory.

    Args:
        response: The push response from Redis that we're ignoring.

    Returns:
        None
    """
    # Explicitly do nothing with the response
    pass


class RedisChatMessageHistory(BaseChatMessageHistory):
    """Redis-based implementation of chat message history using RedisVL.

    This class provides a way to store and retrieve chat message history using Redis
    with RedisVL for efficient indexing, querying, and document management.

    Attributes:
        redis_client (Redis): The Redis client instance.
        session_id (str): A unique identifier for the chat session.
        key_prefix (str): Prefix for Redis keys to namespace the messages.
        ttl (Optional[int]): Time-to-live for message entries in seconds.
        index_name (str): Name of the Redis search index for message retrieval.

    Args:
        session_id (str): A unique identifier for the chat session.
        redis_url (str, optional): URL of the Redis instance. Defaults to "redis://localhost:6379".
        key_prefix (str, optional): Prefix for Redis keys. Defaults to "chat:".
        ttl (Optional[int], optional): Time-to-live for entries in seconds.
            Defaults to None (no expiration).
        index_name (str, optional): Name of the Redis search index.
            Defaults to "idx:chat_history".
        redis_client (Optional[Redis], optional): Existing Redis client instance.
            If provided, redis_url is ignored.
        **kwargs: Additional keyword arguments to pass to the Redis client.

    Raises:
        ValueError: If session_id is empty or None.
        ResponseError: If Redis connection fails or RedisVL operations fail.

    Example:
        .. code-block:: python

            from langchain_redis import RedisChatMessageHistory
            from langchain_core.messages import HumanMessage, AIMessage

            history = RedisChatMessageHistory(
                session_id="user123",
                redis_url="redis://localhost:6379",
                ttl=3600  # Expire chat history after 1 hour
            )

            # Add messages to the history
            history.add_message(HumanMessage(content="Hello, AI!"))
            history.add_message(
              AIMessage(content="Hello, human! How can I assist you today?")
            )

            # Retrieve all messages
            messages = history.messages
            for message in messages:
                print(f"{message.type}: {message.content}")

            # Clear the history for the session
            history.clear()

    Note:
        - This class uses RedisVL for managing Redis JSON storage and search indexes,
          providing efficient querying and retrieval.
        - A Redis search index is created to enable fast lookups and search
          capabilities over the chat history.
        - If TTL is set, message entries will automatically expire after the
          specified duration.
        - The session_id is used to group messages belonging to the same conversation
          or user session.
        - RedisVL automatically handles tokenization and escaping for search queries.
    """

    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "chat:",
        ttl: Optional[int] = None,
        index_name: str = "idx:chat_history",
        redis_client: Optional[Redis] = None,
        **kwargs: Any,
    ) -> None:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty, valid string")

        self.redis_client = redis_client or Redis.from_url(redis_url, **kwargs)

        # Configure Redis client to use a no-op push handler when PubSub is initialized
        if hasattr(self.redis_client, "pubsub_configs"):
            # In newer Redis-py versions, we can set a default pubsub_configs
            self.redis_client.pubsub_configs = {"push_handler_func": _noop_push_handler}
        try:
            self.redis_client.client_setinfo("LIB-NAME", __full_lib_name__)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            self.redis_client.echo(__full_lib_name__)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.index_name = index_name

        # Create RedisVL SearchIndex
        self._create_search_index()

    def _create_search_index(self) -> None:
        """Create and configure the RedisVL SearchIndex.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.
        """
        schema = {
            "index": {
                "name": self.index_name,
                "prefix": self.key_prefix,
                "storage_type": "json",
            },
            "fields": [
                {"name": "session_id", "type": "tag", "path": "$.session_id"},
                {"name": "content", "type": "text", "path": "$.data.content"},
                {"name": "type", "type": "tag", "path": "$.type"},
                {"name": "timestamp", "type": "numeric", "path": "$.timestamp"},
            ],
        }

        self.index = SearchIndex.from_dict(schema, redis_client=self.redis_client)
        self.index.create(overwrite=False)

    @property
    def id(self) -> str:
        """Return the session ID.

        Returns:
            str: The session ID.
        """
        return self.session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages for the current session, sorted by timestamp.

        Returns:
            List[BaseMessage]: A list of messages in chronological order.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.
        """
        messages_query = FilterQuery(
            filter_expression=Tag("session_id") == self.session_id,
            return_fields=["type", "$.data"],
            num_results=10000,
        ).sort_by("timestamp", asc=True)

        messages = self.index.query(messages_query)

        # Unpack message results and load from dict
        return messages_from_dict(
            [
                {"type": msg["type"], "data": json.loads(msg["$.data"])}
                for msg in messages
            ]
        )

    def _message_key(self, message_id: Optional[str] = None) -> str:
        """Construct message key based on key prefix, session, and unique message ID.

        Args:
            message_id (Optional[str]): Optional message ID. If None, a new ULID is
                generated.

        Returns:
            str: The constructed Redis key.
        """
        if message_id is None:
            message_id = str(ULID())
        return f"{self.key_prefix}{self.session_id}:{message_id}"

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history using RedisVL.

        This method adds a new message to the Redis store for the current session
        using RedisVL's document loading capabilities.

        Args:
            message (BaseMessage): The message to add to the history. This should be an
                                   instance of a class derived from BaseMessage, such as
                                   HumanMessage, AIMessage, or SystemMessage.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.
            ValueError: If message is None or invalid.

        Example:
            .. code-block:: python

                from langchain_redis import RedisChatMessageHistory
                from langchain_core.messages import HumanMessage, AIMessage

                history = RedisChatMessageHistory(
                    session_id="user123",
                    redis_url="redis://localhost:6379",
                    ttl=3600  # optional: set TTL to 1 hour
                )

                # Add a human message
                history.add_message(HumanMessage(content="Hello, AI!"))

                # Add an AI message
                history.add_message(
                  AIMessage(content="Hello! How can I assist you today?")
                )

                # Verify messages were added
                print(f"Number of messages: {len(history.messages)}")

        Note:
            - Each message is stored as a separate entry in Redis, associated
              with the current session_id.
            - Messages are stored using RedisVL's JSON capabilities for efficient
              storage and retrieval.
            - If a TTL (Time To Live) was specified when initializing the history,
              it will be applied to each message.
            - The message's content, type, and any additional data (like timestamp)
              are stored.
            - This method is thread-safe and can be used in concurrent environments.
            - The Redis search index is automatically updated to include the new
              message, enabling future searches.
            - Large message contents may impact performance and storage usage.
              Consider implementing size limits if dealing with potentially
              large messages.
        """
        if message is None:
            raise ValueError("Message cannot be None")

        timestamp = datetime.now().timestamp()
        message_id = str(ULID())
        redis_msg = {
            "type": message.type,
            "message_id": message_id,
            "data": {
                "content": message.content,
                "additional_kwargs": message.additional_kwargs,
                "type": message.type,
            },
            "session_id": self.session_id,
            "timestamp": timestamp,
        }

        # Use RedisVL to load the data
        self.index.load(
            data=[redis_msg], keys=[self._message_key(message_id)], ttl=self.ttl
        )

    def clear(self) -> None:
        """Clear all messages from the chat history for the current session.

        This method removes all messages associated with the current session_id from
        the Redis store using RedisVL queries.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.

        Example:
            .. code-block:: python

                from langchain_redis import RedisChatMessageHistory
                from langchain_core.messages import HumanMessage, AIMessage

                history = RedisChatMessageHistory(session_id="user123", redis_url="redis://localhost:6379")

                # Add some messages
                history.add_message(HumanMessage(content="Hello, AI!"))
                history.add_message(AIMessage(content="Hello, human!"))

                # Clear the history
                history.clear()

                # Verify that the history is empty
                assert len(history.messages) == 0

        Note:
            - This method only clears messages for the current session_id.
            - It uses RedisVL's FilterQuery to find all relevant messages and then
              deletes them individually using the Redis client.
            - The operation removes all messages for the current session only.
            - After clearing, the Redis search index is still maintained, allowing
              for immediate use of the same session_id for new messages if needed.
            - This operation is irreversible. Make sure you want to remove all messages
              before calling this method.
        """
        # Get total count of records to delete
        session_filter = Tag("session_id") == self.session_id
        count_query = CountQuery(filter_expression=session_filter)
        total_count = self.index.query(count_query)

        if total_count > 0:
            # Collect all keys first to avoid pagination issues during deletion
            all_keys = []
            filter_query = FilterQuery(
                filter_expression=session_filter, num_results=total_count
            )

            # Use pagination to collect all keys without deleting during iteration
            for results in self.index.paginate(filter_query, page_size=50):
                all_keys.extend([res["id"] for res in results])

            # Now delete all keys at once
            if all_keys:
                self.index.drop_keys(all_keys)

    def delete(self) -> None:
        """Delete all sessions and the chat history index from Redis.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.
        """
        self.index.delete(drop=True)

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for messages in the chat history that match the given query.

        This method performs a full-text search on the content of messages in the
        current session using RedisVL's TextQuery capabilities.

        Args:
            query (str): The search query string to match against message content.
            limit (int, optional): The maximum number of results to return.
                Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a
                                  matching message.
            Each dictionary contains the message content and metadata.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.

        Example:
            .. code-block:: python

                from langchain_redis import RedisChatMessageHistory
                from langchain_core.messages import HumanMessage, AIMessage

                history = RedisChatMessageHistory(session_id="user123", redis_url="redis://localhost:6379")

                # Add some messages
                history.add_message(
                  HumanMessage(content="Tell me about Machine Learning")
                )
                history.add_message(
                  AIMessage(content="Machine Learning is a subset of AI...")
                )
                history.add_message(
                  HumanMessage(content="What are neural networks?")
                )
                history.add_message(
                  AIMessage(
                    content="Neural networks are a key component of deep learning..."
                  )
                )

                # Search for messages containing "learning"
                results = history.search_messages("learning", limit=5)

                for result in results:
                    print(f"Content: {result['content']}")
                    print(f"Type: {result['type']}")
                    print("---")

        Note:
            - The search is performed using RedisVL's TextQuery capabilities, which
              allows for efficient full-text search.
            - The search is case-insensitive and uses Redis' default tokenization
              and stemming.
            - Only messages from the current session (as defined by session_id)
              are searched.
            - The returned dictionaries include all stored fields, which typically
              include 'content', 'type', and any additional metadata stored
              with the message.
            - This method is useful for quickly finding relevant parts of a
              conversation without having to iterate through all messages.
        """
        if not query or not isinstance(query, str):
            return []

        text_query = TextQuery(
            text=query,
            text_field_name="content",
            filter_expression=Tag("session_id") == self.session_id,
            return_fields=["type", "$.data"],
            num_results=limit,
            stopwords=None,  # Disable stopwords to avoid NLTK dependency
        )

        messages = self.index.query(text_query)

        search_data = []
        for msg in messages:
            search_data.append(json.loads(msg["$.data"]))

        return search_data

    def __len__(self) -> int:
        """Return the number of messages in the chat history for the current session.

        Returns:
            int: The number of messages in the current session.

        Raises:
            ResponseError: If Redis connection fails or RedisVL operations fail.
        """
        return self.index.query(
            CountQuery(filter_expression=Tag("session_id") == self.session_id)
        )
