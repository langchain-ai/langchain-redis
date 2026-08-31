# langchain-redis

This package contains the LangChain integration with Redis, providing powerful tools for vector storage, semantic caching, and chat history management.

## Installation

```bash
pip install -U langchain-redis
```

This will install the package along with its dependencies, including `redis`, `redisvl`, and `ulid`.

## Quickstart

Start Redis, install the package, and run a similarity search:

```bash
docker run -d --name redis -p 6379:6379 redis
```

```python
from langchain_openai import OpenAIEmbeddings  # any LangChain Embeddings works
from langchain_redis import RedisVectorStore

vector_store = RedisVectorStore(
    OpenAIEmbeddings(),
    index_name="quickstart",
    redis_url="redis://localhost:6379",
)

vector_store.add_texts(["Redis is a fast in-memory database"])
docs = vector_store.similarity_search("quick key-value store", k=1)
print(docs[0].page_content)
```

## Redis version compatibility

`langchain-redis` is tested against Redis 8.2 and newer. Most functionality works on any Redis with the Query Engine, with a few features gated by server version:

| Feature | Redis 8.0 | Redis 8.2 | Redis 8.4+ |
|---|:---:|:---:|:---:|
| Vector search (KNN / range / MMR) | ✓ | ✓ | ✓ |
| Full-text search | ✓ | ✓ | ✓ |
| Hybrid search — aggregate method (`FT.AGGREGATE`) | ✓ | ✓ | ✓ |
| Hybrid search — native method (`FT.HYBRID`) | ✗ | ✗ | ✓ |
| SVS-VAMANA vector indexing (with compression) | ✗ | ✓ | ✓ |

`hybrid_search()` detects the server version automatically: it uses `FT.HYBRID` on Redis 8.4+ and falls back to the aggregate method on older servers.

## Configuration

Every component takes the connection as a `redis_url` argument (or a
pre-existing client via `redis_client`; see Advanced Configuration for the
`RedisConfig` class). The URL scheme selects the deployment mode:

```python
redis_url = "redis://localhost:6379"                # standard
redis_url = "redis://username:password@host:6379"   # with authentication
redis_url = "rediss://host:6380"                    # SSL/TLS

# Redis Sentinel (high availability):
# redis+sentinel://[username:password@]host1:port1[,host2:port2,...]/service_name
redis_url = "redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
```

The same URL works for every component:

```python
from langchain_redis import RedisCache, RedisChatMessageHistory, RedisVectorStore
from langchain_openai import OpenAIEmbeddings

url = "redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"

vector_store = RedisVectorStore(
    OpenAIEmbeddings(), index_name="my_index", redis_url=url
)
cache = RedisCache(redis_url=url, ttl=3600)
history = RedisChatMessageHistory(session_id="user_123", redis_url=url)
```

## Features

### 1. Vector Store

The `RedisVectorStore` class provides a vector database implementation using Redis.

#### Usage

```python
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_openai import OpenAIEmbeddings
from redisvl.query.filter import Tag

embeddings = OpenAIEmbeddings()  # any LangChain Embeddings implementation works

config = RedisConfig(
    index_name="my_vectors",
    redis_url="redis://localhost:6379",
    distance_metric="COSINE",  # Options: COSINE, L2, IP
    metadata_schema=[
        {"name": "category", "type": "tag"},
    ],
)

vector_store = RedisVectorStore(embeddings, config=config)

# Adding documents
texts = ["Document 1 content", "Document 2 content"]
metadatas = [{"category": "science"}, {"category": "history"}]
vector_store.add_texts(texts, metadatas=metadatas)

# Adding documents with custom keys
custom_keys = ["doc1", "doc2"]
vector_store.add_texts(texts, metadatas=metadatas, keys=custom_keys)

# Similarity search
query = "Sample query"
docs = vector_store.similarity_search(query, k=2)

# Similarity search with score
docs_and_scores = vector_store.similarity_search_with_score(query, k=2)

# Similarity search with filtering
from redisvl.query.filter import Tag

filter_expr = Tag("category") == "science"
filtered_docs = vector_store.similarity_search(query, k=2, filter=filter_expr)

# Tag filters support wildcard patterns
filtered_docs = vector_store.similarity_search(
    query, k=2, filter=Tag("category") % "scien*"
)

# Maximum marginal relevance search
docs = vector_store.max_marginal_relevance_search(query, k=2, fetch_k=10)
```

Metadata is always stored and returned with documents, but only fields
declared in `metadata_schema` are indexed and filterable. Filtering on an
undeclared field raises an `Unknown field` error.

#### Hybrid and full-text search

`hybrid_search` combines BM25 text scoring with vector similarity in a single
query, and `full_text_search` exposes BM25-only retrieval. The examples below
share this product catalog:

```python
from redisvl.query.filter import Num, Tag

store = RedisVectorStore(
    embeddings,
    index_name="products",
    redis_url="redis://localhost:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "title", "type": "text"},
        {"name": "price", "type": "numeric"},
    ],
)

store.add_texts(
    [
        "Lightweight running shoes with cushioned soles",
        "Waterproof hiking boots for rough mountain trails",
        "Ergonomic office chair with lumbar support",
    ],
    metadatas=[
        {"category": "footwear", "title": "running shoes", "price": 89},
        {"category": "footwear", "title": "hiking boots", "price": 129},
        {"category": "furniture", "title": "office chair", "price": 249},
    ],
)
```

```python
# Hybrid search: text relevance + vector similarity, one ranked list.
# Uses FT.HYBRID (RRF rank fusion) on Redis 8.4+ and transparently falls
# back to an FT.AGGREGATE-based combination on older servers.
docs = store.hybrid_search("running shoes", k=2)

# With scores, an explicit engine, and linear weighting:
# score = alpha * vector_score + (1 - alpha) * text_score
results = store.hybrid_search_with_score(
    "running shoes",
    k=2,
    method="aggregate",       # "auto" (default), "ft_hybrid", or "aggregate"
    alpha=0.5,
    filter=Tag("category") == "footwear",
)

# Full-text (BM25) search, weighting title matches higher than body text
docs = store.full_text_search(
    "running shoes", k=2, text_fields={"title": 5.0, "text": 1.0}
)
```

Hybrid scores are combined scores (rank-based for RRF), not comparable with
the cosine distances returned by `similarity_search_with_score`.

#### Filter-based deletion and metadata updates

Documents can be deleted or updated by what they are, not just by id — using
the same `FilterExpression` type as search. Useful for re-syncing a changed
source, tenant offboarding, retention policies, or bulk retagging. Continuing
with the product catalog above:

```python
# Delete every document matching a filter
store.delete(filter=Tag("category") == "furniture")

# Preview a purge before running it, and get exact counts
would_delete = store.delete_by_filter(Num("price") > 100, dry_run=True)
deleted = store.delete_by_filter(Num("price") > 100)

# Bulk-update metadata in place — no re-embedding
updated = store.update_metadata_by_filter(
    Tag("category") == "footwear", {"category": "outdoor"}
)
```

Both operations are automatically scoped to the store's own index, so
indexes sharing a `key_prefix` cannot delete or modify each other's
documents.

#### Vector index tuning

`vector_attrs` exposes algorithm-specific index attributes without writing a
full custom schema:

```python
# HNSW with tuned build/query parameters
config = RedisConfig(
    index_name="tuned_hnsw",
    indexing_algorithm="HNSW",
    vector_attrs={"m": 16, "ef_construction": 200, "ef_runtime": 20},
)

# SVS-VAMANA (Redis 8.2+) with vector compression for reduced memory
config = RedisConfig(
    index_name="compressed",
    indexing_algorithm="SVS-VAMANA",
    vector_attrs={"compression": "LeanVec4x8", "reduce": 256},
)
```

#### Using an existing index or schema

To connect to an index that already exists in Redis (created by another
process, or by redisvl directly):

```python
vector_store = RedisVectorStore.from_existing_index(
    index_name="products",
    embedding=embeddings,
    redis_url="redis://localhost:6379",
)
```

A redisvl `IndexSchema` or schema YAML file can also define the index
instead of the individual config fields:

```python
from redisvl.schema import IndexSchema

schema = IndexSchema.from_yaml("index.yaml")   # or IndexSchema.from_dict(...)
config = RedisConfig(schema=schema, redis_url="redis://localhost:6379")
vector_store = RedisVectorStore(embeddings, config=config)
```

#### Features
- Efficient vector storage and retrieval
- Hybrid search combining BM25 text scoring and vector similarity (RRF or
  linear fusion), with automatic engine selection by server version
- Full-text (BM25) search with per-field weighting
- Support for metadata filtering, including wildcard tag patterns
- Filter-based bulk deletion (with dry-run) and in-place metadata updates
- Multiple distance metrics: Cosine similarity, L2, and Inner Product
- FLAT, HNSW, and SVS-VAMANA indexing algorithms with tunable
  algorithm-specific attributes (including SVS-VAMANA vector compression)
- Maximum marginal relevance search
- Custom key support for document indexing
- Multi-prefix indexes: one searchable index spanning several key namespaces
- Index-level stopwords configuration

### 2. Cache

The `RedisCache`, `RedisSemanticCache`, and `LangCacheSemanticCache` classes provide caching mechanisms for LLM calls.

#### Usage

```python
from langchain_redis import RedisCache, RedisSemanticCache, LangCacheSemanticCache
from langchain_core.language_models import LLM
from langchain_openai import OpenAIEmbeddings

# Standard cache
cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)

# Semantic cache
embeddings = OpenAIEmbeddings()  # any LangChain Embeddings implementation works
semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embeddings,
    distance_threshold=0.1
)

# LangChain cache - manages embeddings for you
langchain_cache = LangCacheSemanticCache(
    cache_id="your-cache-id",
    api_key="your-api-key",
    distance_threshold=0.1
)

# Using cache with an LLM
llm = LLM(cache=cache)  # or LLM(cache=semantic_cache) or LLM(cache=langchain_cache)

# Async cache operations
await cache.aupdate("prompt", "llm_string", [Generation(text="cached_response")])
cached_result = await cache.alookup("prompt", "llm_string")
```

#### Features
- Efficient caching of LLM responses
- TTL support for automatic cache expiration
- Semantic caching for similarity-based retrieval
- Asynchronous cache operations

#### What is Redis LangCache?
- LangCache is a fully managed, cloud-based service that provides a semantic cache for LLM applications.
- It manages embeddings and vector search for you, allowing you to focus on your application logic.
- See [our docs](https://redis.io/docs/latest/develop/ai/langcache/) to learn more, or [try LangCache on Redis Cloud today](https://redis.io/docs/latest/operate/rc/langcache/#get-started-with-langcache-on-redis-cloud).

### 3. Chat History

The `RedisChatMessageHistory` class provides a Redis-based storage for chat message history with efficient search capabilities.

#### Usage

```python
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize with optional TTL (time-to-live) in seconds
history = RedisChatMessageHistory(
    session_id="user_123",
    redis_url="redis://localhost:6379",
    ttl=3600,  # Messages will expire after 1 hour
)

# Adding messages
history.add_message(HumanMessage(content="Hello, AI!"))
history.add_message(AIMessage(content="Hello, human! How can I assist you today?"))
history.add_message(SystemMessage(content="This is a system message"))

# Retrieving all messages in chronological order
messages = history.messages

# Searching messages with full-text search
results = history.search_messages("assist", limit=5)  # Returns matching messages

# Get message count
message_count = len(history)

# Clear history for current session
history.clear()

# Delete all sessions and index (use with caution)
history.delete()
```

#### Features
- Fast storage of chat messages with automatic expiration (TTL)
- Support for different message types (Human, AI, System)
- Full-text search capabilities across message content
- Chronological message retrieval
- Session-based message organization
- Customizable key prefixing
- Thread-safe operations
- Efficient RedisVL-based indexing and querying

## Advanced Configuration

The `RedisConfig` class allows for detailed configuration of the Redis integration:

```python
from langchain_redis import RedisConfig

config = RedisConfig(
    index_name="my_index",
    redis_url="redis://localhost:6379",
    distance_metric="COSINE",
    key_prefix="my_prefix",
    vector_datatype="FLOAT32",
    storage_type="hash",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "price", "type": "numeric"}
    ]
)
```

Available options include:

- `key_prefix`: a single prefix, or a list of prefixes (e.g.
  `key_prefix=["kb:team_acme", "kb:global"]`). With a list, searches cover
  every namespace and new documents are written under the first prefix.
- `stopwords`: index-level stopwords. `None` (default) uses the Redis
  defaults, `[]` disables them (`STOPWORDS 0`), and a list replaces them.
- `indexing_algorithm`: `"FLAT"`, `"HNSW"`, or `"SVS-VAMANA"`.
  Algorithm-specific parameters go in `vector_attrs` (see Vector index
  tuning above).

Refer to the inline documentation for detailed information on these configuration options.

## Error Handling and Logging

The package uses Python's standard logging module. You can configure logging to get more information about the package's operations:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Error handling is done through custom exceptions. Make sure to handle these exceptions in your application code.

## Performance Considerations

- For large datasets, consider using batched operations when adding documents to the vector store.
- Adjust the `k` and `fetch_k` parameters in similarity searches to balance between accuracy and performance.
- Use appropriate indexing algorithms (FLAT, HNSW) based on your dataset size and query requirements.

## Examples

For more detailed examples and use cases, please refer to the `docs/` directory in this repository.

## Contributing / Development

The library is rooted at `libs/redis`, for all the commands below, CD to `libs/redis`:

### Unit Tests

To install dependencies for unit tests:

```bash
poetry install --with test
```

To run unit tests:

```bash
make test
```

To run a specific test:

```bash
TEST_FILE=tests/unit_tests/test_imports.py make test
```

## Integration Tests

You would need an OpenAI API Key to run the integration tests:

```bash
export OPENAI_API_KEY=sk-J3nnYJ3nnYWh0Can1Turnt0Ug1VeMe50mth1n1cAnH0ld0n2
```

To install dependencies for integration tests:

```bash
poetry install --with test,test_integration
```

To run integration tests:

```bash
make integration_tests
```

## Local Development

Install langchain-redis development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
poetry install --with lint,typing,test,test_integration
```

Then verify dependency installation:

```bash
make lint
```

## License

This project is licensed under the MIT License (LICENSE).