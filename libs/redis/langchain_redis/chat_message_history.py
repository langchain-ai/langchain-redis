import json  # noqa: I001
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.json.path import Path
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from langchain_redis.version import __full_lib_name__


class RedisChatMessageHistory(BaseChatMessageHistory):
    """Redis-based implementation of chat message history.

    This class provides a way to store and retrieve chat message history using Redis.
    It implements the BaseChatMessageHistory interface and uses Redis JSON capabilities
    for efficient storage and retrieval of messages.

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
        redis (Optional[Redis], optional): Existing Redis client instance. If provided,
                                           redis_url is ignored.
        **kwargs: Additional keyword arguments to pass to the Redis client.

    Example:
        .. code-block:: python

            from langchain_redis import RedisChatMessageHistory
            from langchain_core.messages import HumanMessage, AIMessage

            history = RedisChatMessageHistory(
                session_id="user123",
                redis_url="redis://localhost:6379",
                ttl=3600  # Messages expire after 1 hour
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

            # Clear the history
            history.clear()

    Note:
        - This class uses Redis JSON for storing messages, allowing for efficient
          querying and retrieval.
        - A Redis search index is created to enable fast lookups and potential future
          search needs over the chat history.
        - If TTL is set, message entries will automatically expire after the
          specified duration.
        - The session_id is used to group messages belonging to the same conversation
          or user session.
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
    ):
        self.redis_client = redis_client or Redis.from_url(redis_url, **kwargs)
        try:
            self.redis_client.client_setinfo("LIB-NAME", __full_lib_name__)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            self.redis_client.echo(__full_lib_name__)
        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.index_name = index_name
        self._ensure_index()

    @property
    def id(self) -> str:
        return self.session_id

    def _ensure_index(self) -> None:
        try:
            self.redis_client.ft(self.index_name).info()
        except ResponseError as e:
            if str(e).lower() == "unknown index name":
                schema = (
                    TagField("$.session_id", as_name="session_id"),
                    TextField("$.data.content", as_name="content"),
                    TagField("$.type", as_name="type"),
                    NumericField("$.timestamp", as_name="timestamp"),
                )
                definition = IndexDefinition(
                    prefix=[self.key_prefix], index_type=IndexType.JSON
                )
                self.redis_client.ft(self.index_name).create_index(
                    schema, definition=definition
                )
            else:
                raise

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        query = (
            Query(f"@session_id:{{{self.session_id}}}")
            .sort_by("timestamp", asc=True)
            .paging(0, 10000)
        )
        results = self.redis_client.ft(self.index_name).search(query)
        return messages_from_dict(
            [
                {
                    "type": json.loads(doc.json)["type"],
                    "data": json.loads(doc.json)["data"],
                }
                for doc in results.docs
            ]
        )

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history.

        This method adds a new message to the Redis store for the current session.

        Args:
            message (BaseMessage): The message to add to the history. This should be an
                                   instance of a class derived from BaseMessage, such as
                                   HumanMessage, AIMessage, or SystemMessage.

        Returns:
            None

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
            - Messages are stored using Redis JSON capabilities for efficient storage
              and retrieval.
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
        data_to_store = {
            "type": message.type,
            "data": {
                "content": message.content,
                "additional_kwargs": message.additional_kwargs,
                "type": message.type,
            },
            "session_id": self.session_id,
            "timestamp": datetime.now().timestamp(),
        }

        key = f"{self.key_prefix}{self.session_id}:{data_to_store['timestamp']}"
        self.redis_client.json().set(key, Path.root_path(), data_to_store)

        if self.ttl:
            self.redis_client.expire(key, self.ttl)

    def clear(self) -> None:
        """Clear all messages from the chat history for the current session.

        This method removes all messages associated with the current session_id from
        the Redis store.

        Returns:
            None

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
            - It uses a Redis search query to find all relevant messages and then
              deletes them.
            - The operation is atomic - either all messages are deleted, or none are.
            - After clearing, the Redis search index is still maintained, allowing
              for immediate use of the same session_id for new messages if needed.
            - This operation is irreversible. Make sure you want to remove all messages
              before calling this method.
        """
        query = Query(f"@session_id:{{{self.session_id}}}").paging(0, 10000)
        results = self.redis_client.ft(self.index_name).search(query)
        for doc in results.docs:
            self.redis_client.delete(doc.id)

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for messages in the chat history that match the given query.

        This method performs a full-text search on the content of messages in the
        current session.

        Args:
            query (str): The search query string to match against message content.
            limit (int, optional): The maximum number of results to return.
            Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a
                                  matching message.
            Each dictionary contains the message content and metadata.

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
            - The search is performed using the Redis search capabilities, which allows
              for efficient full-text search.
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
        search_query = (
            Query(f"(@session_id:{{{self.session_id}}}) (@content:{query})")
            .sort_by("timestamp", asc=True)
            .paging(0, limit)
        )
        results = self.redis_client.ft(self.index_name).search(search_query)

        return [json.loads(doc.json)["data"] for doc in results.docs]

    def __len__(self) -> int:
        query = Query(f"@session_id:{{{self.session_id}}}").no_content()
        return self.redis_client.ft(self.index_name).search(query).total
