from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator
from redis import Redis
from redisvl.schema import IndexSchema, StorageType  # type: ignore[import]
from typing_extensions import Annotated, Self
from ulid import ULID


def generate_ulid() -> str:
    return str(ULID())


class RedisConfig(BaseModel):
    """Configuration class for Redis vector store settings.

    This class defines the configuration parameters for setting up and interacting with
    a Redis vector store. It uses Pydantic for data validation and settings management.

    Attributes:
        index_name (str): Name of the index in Redis. Defaults to a generated ULID.
        from_existing (bool): Whether to use an existing index. Defaults to False.
        key_prefix (Optional[str]): Prefix for Redis keys. Defaults to index_name
                                    if not set.
        redis_url (str): URL of the Redis instance. Defaults to "redis://localhost:6379".
        redis_client (Optional[Redis]): Pre-existing Redis client instance.
        connection_args (Optional[Dict[str, Any]]): Additional Redis
                                                    connection arguments.
        distance_metric (str): Distance metric for vector similarity.
                               Defaults to "COSINE".
        indexing_algorithm (str): Algorithm used for indexing. Defaults to "FLAT".
        vector_datatype (str): Data type of the vector. Defaults to "FLOAT32".
        storage_type (str): Storage type in Redis. Defaults to "hash".
        id_field (str): Field name for document ID. Defaults to "id".
        content_field (str): Field name for document content. Defaults to "text".
        embedding_field (str): Field name for embedding vector. Defaults to "embedding".
        default_tag_separator (str): Separator for tag fields. Defaults to "|".
        metadata_schema (Optional[List[Dict[str, Any]]]): Schema for metadata fields.
        index_schema (Optional[IndexSchema]): Full index schema definition.
        schema_path (Optional[str]): Path to a YAML file containing the index schema.
        return_keys (bool): Whether to return keys after adding documents.
                            Defaults to False.
        custom_keys (Optional[List[str]]): Custom keys for documents.
        embedding_dimensions (Optional[int]): Dimensionality of embedding vectors.

    Example:
        .. code-block:: python

            from langchain_redis import RedisConfig

            config = RedisConfig(
                index_name="my_index",
                redis_url="redis://localhost:6379",
                distance_metric="COSINE",
                embedding_dimensions=1536
            )

            # Use this config to initialize a RedisVectorStore
            vector_store = RedisVectorStore(embeddings=my_embeddings, config=config)

    Note:
        - Only one of 'index_schema', 'schema_path', or 'metadata_schema'
          should be specified.
        - The 'key_prefix' is automatically set to 'index_name' if not provided.
        - When 'from_existing' is True, it connects to an existing index instead
          of creating a new one.
        - Custom validation ensures that incompatible options are not
          simultaneously specified.
    """

    index_name: str = Field(default_factory=lambda: generate_ulid())
    from_existing: bool = False
    key_prefix: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    redis_client: Optional[Redis] = Field(default=None)
    connection_args: Optional[Dict[str, Any]] = Field(default={})
    distance_metric: str = "COSINE"
    indexing_algorithm: str = "FLAT"
    vector_datatype: str = "FLOAT32"
    storage_type: str = "hash"
    id_field: str = "id"
    content_field: str = "text"
    embedding_field: str = "embedding"
    default_tag_separator: str = "|"
    metadata_schema: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    index_schema: Annotated[Optional[IndexSchema], SkipValidation()] = Field(
        default=None, alias="schema"
    )
    schema_path: Optional[str] = None
    return_keys: bool = False
    custom_keys: Optional[List[str]] = None
    embedding_dimensions: Optional[int] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def check_schema_options(cls, values: Dict) -> Dict:
        options = [
            values.get("index_schema"),
            values.get("schema_path"),
            values.get("metadata_schema"),
        ]
        if sum(option is not None for option in options) > 1:
            raise ValueError(
                "Only one of 'index_schema', 'schema_path', "
                "or 'metadata_schema' can be specified."
            )
        if "schema" in values:
            schema = values.pop("schema")
            values["index_name"] = schema.index.name
            values["key_prefix"] = schema.index.prefix
            values["storage_type"] = schema.index.storage_type.value
            values["index_schema"] = schema

        return values

    @model_validator(mode="after")
    def set_key_prefix(self) -> Self:
        if self.key_prefix is None:
            self.key_prefix = self.index_name
        return self

    @classmethod
    def from_kwargs(cls: Type["RedisConfig"], **kwargs: Any) -> "RedisConfig":
        """Create a RedisConfig object with default values,
           overwritten by provided kwargs.

        This class method allows for flexible creation of a RedisConfig object,
        using default values where not specified and overriding with any provided
        keyword arguments.

        Args:
            **kwargs: Keyword arguments that match RedisConfig attributes. These will
                      override default values.
                Common kwargs include:
                - index_name (str): Name of the index in Redis.
                - redis_url (str): URL of the Redis instance.
                - distance_metric (str): Distance metric for vector similarity.
                - indexing_algorithm (str): Algorithm used for indexing.
                - vector_datatype (str): Data type of the vector.
                - embedding_dimensions (int): Dimensionality of embedding vectors.

        Returns:
            RedisConfig: A new instance of RedisConfig with applied settings.

        Example:
            .. code-block:: python

                from langchain_redis import RedisConfig

                config = RedisConfig.from_kwargs(
                    index_name="my_custom_index",
                    redis_url="redis://custom-host:6379",
                    distance_metric="COSINE",
                    embedding_dimensions=768
                )

                print(config.index_name)  # Output: my_custom_index
                print(config.distance_metric)  # Output: COSINE

        Note:
            - This method first sets all attributes to their default values and
              then overwrites them with provided kwargs.
            - If a 'schema' argument is provided, it will be set as 'index_schema'
              in the config.
            - This method is particularly useful when you want to create a config
              with mostly default values but need to customize a few specific
              attributes.
            - Any attribute of RedisConfig can be set through kwargs, providing full
              flexibility in configuration.
        """
        # Get the default values from the class attributes
        default_config = {}
        for field_name, field in cls.model_fields.items():
            if field.default is not None:
                default_config[field_name] = field.default
            elif field.default_factory is not None:
                default_config[field_name] = field.default_factory()

        # Handle special case for 'schema' argument
        if "schema" in kwargs:
            kwargs["index_schema"] = kwargs.pop("schema")

        # Update default_config with any provided kwargs
        default_config.update(kwargs)

        # Create and return the RedisConfig object
        return cls(**default_config)

    @classmethod
    def from_schema(cls, schema: IndexSchema, **kwargs: Any) -> "RedisConfig":
        """Create a RedisConfig object from an IndexSchema.

        This class method creates a RedisConfig instance using the provided IndexSchema,
        which defines the structure of the Redis index.

        Args:
            schema (IndexSchema): An IndexSchema object defining the structure of
                                  the Redis index.
            **kwargs: Additional keyword arguments to override or supplement the
                      schema-derived settings.
                Common kwargs include:
                - redis_url (str): URL of the Redis instance.
                - distance_metric (str): Distance metric for vector similarity.
                - embedding_dimensions (int): Dimensionality of embedding vectors.

        Returns:
            RedisConfig: A new instance of RedisConfig configured based on the provided
                         schema and kwargs.

        Example:
            .. code-block:: python

                from redisvl.schema import IndexSchema
                from langchain_redis import RedisConfig

                schema = IndexSchema.from_dict({
                    "index": {"name": "my_index", "storage_type": "hash"},
                    "fields": [
                        {"name": "text", "type": "text"},
                        {
                            "name": "embedding",
                            "type": "vector",
                            "attrs": {"dims": 1536, "distance_metric": "cosine"}
                        }
                    ]
                })

                config = RedisConfig.from_schema(
                    schema,
                    redis_url="redis://localhost:6379"
                )

                print(config.index_name)  # Output: my_index
                print(config.storage_type)  # Output: hash

        Note:
            - The method extracts index name, key prefix, and storage type from the
              schema.
            - If the schema specifies a vector field, its attributes (like dimensions
              and distance metric) are used.
            - Additional kwargs can override settings derived from the schema.
            - This method is useful when you have a pre-defined index structure and want
              to create a matching config.
            - The resulting config can be used to ensure that a RedisVectorStore matches
              an existing index structure.
        """
        if schema.index.storage_type == StorageType.HASH:
            storage_type = "hash"
        else:
            storage_type = "json"
        return cls(
            schema=schema,
            index_name=schema.index.name,
            key_prefix=schema.index.prefix,
            storage_type=storage_type,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, schema_path: str, **kwargs: Any) -> "RedisConfig":
        """Create a RedisConfig object from a YAML file containing the index schema.

        This class method creates a RedisConfig instance using a YAML file that defines
        the structure of the Redis index.

        Args:
            schema_path (str): Path to the YAML file containing the index schema
                               definition.
            **kwargs: Additional keyword arguments to override or supplement the
                      schema-derived settings.
                Common kwargs include:
                - redis_url (str): URL of the Redis instance.
                - distance_metric (str): Distance metric for vector similarity.
                - embedding_dimensions (int): Dimensionality of embedding vectors.

        Returns:
            RedisConfig: A new instance of RedisConfig configured based on the YAML
                         schema and kwargs.

        Example:
            .. code-block:: python

                from langchain_redis import RedisConfig

                # Assuming 'index_schema.yaml' contains a valid index schema
                config = RedisConfig.from_yaml(
                    schema_path="path/to/index_schema.yaml",
                    redis_url="redis://localhost:6379"
                )

                print(config.index_name)  # Output: Name defined in YAML
                print(config.storage_type)  # Output: Storage type defined in YAML

        Note:
            - The YAML file should contain a valid index schema definition
              compatible with RedisVL.
            - This method internally uses IndexSchema.from_yaml() to parse
              the YAML file.
            - Settings derived from the YAML schema can be overridden by
              additional kwargs.
            - This method is particularly useful when index structures are defined
              externally in YAML files.
            - Ensure that the YAML file is correctly formatted and accessible at
              the given path.
            - Any errors in reading or parsing the YAML file will be propagated
              as exceptions.

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
            YAMLError: If there are issues parsing the YAML file.
            ValueError: If the YAML content is not a valid index schema.
        """
        return cls(schema_path=schema_path, **kwargs)

    @classmethod
    def with_metadata_schema(
        cls, metadata_schema: List[Dict[str, Any]], **kwargs: Any
    ) -> "RedisConfig":
        """Create a RedisConfig object with a specified metadata schema.

        This class method creates a RedisConfig instance using a provided
        metadata schema, which defines the structure of additional metadata
        fields in the Redis index.

        Args:
            metadata_schema (List[Dict[str, Any]]): A list of dictionaries defining
                the metadata fields. Each dictionary should contain at least 'name'
                and 'type' keys.
            **kwargs: Additional keyword arguments to configure the RedisConfig
                      instance.
                Common kwargs include:
                - index_name (str): Name of the index in Redis.
                - redis_url (str): URL of the Redis instance.
                - distance_metric (str): Distance metric for vector similarity.
                - embedding_dimensions (int): Dimensionality of embedding vectors.

        Returns:
            RedisConfig: A new instance of RedisConfig configured with the specified
                         metadata schema.

        Example:
            .. code-block:: python

                from langchain_redis import RedisConfig

                metadata_schema = [
                    {"name": "author", "type": "text"},
                    {"name": "publication_date", "type": "numeric"},
                    {"name": "tags", "type": "tag", "separator": ","}
                ]

                config = RedisConfig.with_metadata_schema(
                    metadata_schema,
                    index_name="book_index",
                    redis_url="redis://localhost:6379",
                    embedding_dimensions=1536
                )

                print(config.metadata_schema)  # Output: The metadata schema list
                print(config.index_name)  # Output: book_index

        Note:
            - The metadata_schema defines additional fields beyond the default content
              and embedding fields.
            - Common metadata field types include 'text', 'numeric', and 'tag'.
            - For 'tag' fields, you can specify a custom separator using the
              'separator' key.
            - This method is useful when you need to index and search on specific
              metadata attributes.
            - The resulting config ensures that the RedisVectorStore will create
              an index with the specified metadata fields.
            - Make sure the metadata schema aligns with the actual metadata you
              plan to store with your documents.
            - This method sets only the metadata_schema; other configurations
              should be provided via kwargs.

        Raises:
            ValueError: If the metadata_schema is not a list of dictionaries or
                        if required keys are missing.
        """
        return cls(metadata_schema=metadata_schema, **kwargs)

    @classmethod
    def from_existing_index(cls, index_name: str, redis: Redis) -> "RedisConfig":
        """Create a RedisConfig object from an existing Redis index.

        This class method creates a RedisConfig instance based on the configuration
        of an existing index in Redis. It's useful for connecting to and working with
        pre-existing Redis vector store indexes.

        Args:
            index_name (str): The name of the existing index in Redis.
            redis (Redis): An active Redis client instance connected to the Redis server
                        where the index exists.

        Returns:
            RedisConfig: A new instance of RedisConfig configured to match the existing
                         index.

        Example:
            .. code-block:: python

                from redis import Redis
                from langchain_redis import RedisConfig

                # Assuming an existing Redis connection
                redis_client = Redis.from_url("redis://localhost:6379")

                config = RedisConfig.from_existing_index(
                    index_name="my_existing_index",
                    redis_client=redis_client
                )

                print(config.index_name)  # Output: my_existing_index
                print(config.from_existing)  # Output: True

        Note:
            - This method sets the 'from_existing' attribute to True, indicating that
              the configuration is based on an existing index.
            - The method doesn't fetch the full schema or configuration of the
              existing index. It only sets up the basic parameters needed to connect
              to the index.
            - Additional index details (like field configurations) are not retrieved and
              should be known or discovered separately if needed.
            - This method is particularly useful when you need to work with or extend an
              existing Redis vector store index.
            - Ensure that the provided Redis client has the necessary permissions to
              access the specified index.
            - If the index doesn't exist, this method will still create a config, but
              operations using this config may fail until the index is created.

        Raises:
            ValueError: If the index_name is empty or None.
            ConnectionError: If there's an issue connecting to Redis using the
                             provided client.
        """
        return cls(index_name=index_name)

    def to_index_schema(self) -> IndexSchema:
        """Convert the RedisConfig to an IndexSchema.

        This method creates an IndexSchema object based on the current configuration.
        It's useful for generating a schema that can be used to create or update
        a Redis index.

        Returns:
            IndexSchema: An IndexSchema object representing the current configuration.

        Example:
            .. code-block:: python

                from langchain_redis import RedisConfig

                config = RedisConfig(
                    index_name="my_index",
                    embedding_dimensions=1536,
                    distance_metric="COSINE",
                    metadata_schema=[
                        {"name": "author", "type": "text"},
                        {"name": "year", "type": "numeric"}
                    ]
                )

                schema = config.to_index_schema()
                print(schema.index.name)
                # Output: my_index
                print(len(schema.fields))
                # Output: 4 (id, content, embedding, author, year)

        Note:
            - If an index_schema is already set, it will be returned directly.
            - If a schema_path is set, the schema will be loaded from the YAML file.
            - Otherwise, a new IndexSchema is created based on the current
              configuration.
            - The resulting schema includes fields for id, content, and embedding
              vector, as well as any additional fields specified in metadata_schema.
            - The embedding field is configured with the specified dimensions,
              distance metric, and other relevant attributes.
            - This method is particularly useful when you need to create a new index or
              validate the structure of an existing one.
            - The generated schema can be used with RedisVL operations that require
              an IndexSchema.

        Raises:
            ValueError: If essential configuration elements (like embedding_dimensions)
                        are missing.
        """
        if self.index_schema:
            return self.index_schema
        elif self.schema_path:
            return IndexSchema.from_yaml(self.schema_path)
        else:
            index_info = {
                "name": self.index_name,
                "prefix": self.key_prefix,
                "storage_type": self.storage_type,
            }

            fields = [
                {"name": self.id_field, "type": "tag"},
                {"name": self.content_field, "type": "text"},
                {
                    "name": self.embedding_field,
                    "type": "vector",
                    "attrs": {
                        "dims": self.embedding_dimensions,
                        "distance_metric": self.distance_metric.lower(),
                        "algorithm": self.indexing_algorithm.lower(),
                        "datatype": self.vector_datatype.lower(),
                    },
                },
            ]

            if self.metadata_schema:
                fields.extend(self.metadata_schema)

            return IndexSchema.from_dict({"index": index_info, "fields": fields})

    def redis(self) -> Redis:
        if self.redis_client is not None:
            return self.redis_client
        elif self.redis_url is not None:
            if self.connection_args is not None:
                return Redis.from_url(self.redis_url, **self.connection_args)
            else:
                return Redis.from_url(self.redis_url)
        else:
            raise ValueError("Either redis_client or redis_url must be provided")
