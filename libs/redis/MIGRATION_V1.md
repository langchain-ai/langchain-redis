# Migration Guide: LangChain v1.0

## Overview

`langchain-redis` v0.3.0 adds support for LangChain v1.0, which was released in January 2025. This guide helps you migrate your applications to use the latest version.

## What Changed

### Breaking Changes

#### 1. Python Version Requirement

### Minimum Python version is now 3.10+

- **Python 3.9 is no longer supported** (reaches end-of-life October 2025)
- **Supported versions**: Python 3.10, 3.11, 3.12, 3.13
- **Not yet supported**: Python 3.14 (awaiting dependency updates)

**Action Required:**

- If you're on Python 3.9, upgrade to Python 3.10 or higher
- If you're on Python 3.10-3.13, no action needed
- If you're on Python 3.14, please use Python 3.10-3.13 until future release

#### 2. Dependency Updates

### Updated to LangChain v1.0

```toml
langchain-core = "^1.0"  # was ^0.3
```

**Action Required:**

```bash
# Update your requirements.txt or pyproject.toml
pip install --upgrade langchain-redis langchain-core

# Or with poetry
poetry update langchain-redis langchain-core
```

### Recommended Changes (Optional but Encouraged)

#### Import Path Updates

While old imports still work, we recommend updating to the new paths for future compatibility:

**Cache and Globals:**

```python
# Old (still works but deprecated):
from langchain.globals import set_llm_cache

# New (recommended):
from langchain_core.globals import set_llm_cache
```

**Output Types:**

```python
# Old (still works but deprecated):
from langchain.schema import Generation

# New (recommended):
from langchain_core.outputs import Generation
```

**Documents:**

```python
# Old (still works but deprecated):
from langchain.docstore.document import Document

# New (recommended):
from langchain_core.documents import Document
```

**Text Splitters:**

```python
# Old (still works but deprecated):
from langchain.text_splitter import RecursiveCharacterTextSplitter

# New (recommended):
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

## What Did NOT Change

**Good news**: The `langchain-redis` API remains completely unchanged!

All three main components work seamlessly without any code changes:

- ✅ **`RedisVectorStore`** - No changes required
- ✅ **`RedisCache` / `RedisSemanticCache`** - No changes required
- ✅ **`RedisChatMessageHistory`** - No changes required
- ✅ **`RedisConfig`** - No changes required

Your existing code will continue to work as-is after updating dependencies.

## Migration Steps

### Step 1: Check Your Python Version

```bash
python --version
```

If you're on Python 3.9, upgrade to 3.10+:

```bash
# Using pyenv (recommended)
pyenv install 3.10.15  # or 3.11, 3.12, 3.13
pyenv global 3.10.15

# Recreate your virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Update Dependencies

### Option A: Using pip

```bash
pip install --upgrade langchain-redis langchain-core
```

### Option B: Using poetry

```bash
# Update pyproject.toml
# python = ">=3.10,<3.14"
# langchain-core = "^1.0"

poetry lock
poetry install
```

### Option C: Using requirements.txt

```txt
langchain-redis>=0.3.0
langchain-core>=1.0
```

### Step 3: Update Imports (Optional)

Update deprecated import paths in your code:

```bash
# Find deprecated imports
grep -r "from langchain\\.globals" .
grep -r "from langchain\\.schema" .
grep -r "from langchain\\.docstore" .
grep -r "from langchain\\.text_splitter" .
```

Replace with new paths as shown in the "Recommended Changes" section above.

### Step 4: Test Your Application

```bash
# Run your test suite
pytest tests/

# Verify your application works
python your_app.py
```

## Example Migration

### Before (works with both v0.2.x and v0.3.0)

```python
from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_openai import OpenAI
from langchain_redis import RedisCache

# Initialize cache
cache = RedisCache(redis_url="redis://localhost:6379")
set_llm_cache(cache)

# Use as normal
llm = OpenAI()
result = llm.invoke("Hello!")
```

### After (recommended for v0.3.0+)

```python
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation
from langchain_openai import OpenAI
from langchain_redis import RedisCache

# Initialize cache (no changes needed!)
cache = RedisCache(redis_url="redis://localhost:6379")
set_llm_cache(cache)

# Use as normal (no changes needed!)
llm = OpenAI()
result = llm.invoke("Hello!")
```

## Troubleshooting

### Issue: Import errors after upgrade

**Symptom:**

```python
ImportError: cannot import name 'set_llm_cache' from 'langchain.globals'
```

**Solution:**
Update your imports to use `langchain_core.globals`:

```python
from langchain_core.globals import set_llm_cache
```

### Issue: Python version conflict

**Symptom:**

```bash
ERROR: Package 'langchain-redis' requires a different Python: 3.9.x not in '>=3.10,<3.14'
```

**Solution:**
Upgrade to Python 3.10 or higher (see Step 1 above).

### Issue: Dependency resolver conflicts

**Symptom:**

```bash
ERROR: Cannot install langchain-redis and langchain-core because these package versions have conflicting dependencies.
```

**Solution:**

```bash
# Clear dependency cache
pip cache purge

# Install with updated resolver
pip install --upgrade --force-reinstall langchain-redis langchain-core
```

## Version Compatibility Matrix

| langchain-redis | langchain-core | Python | Status |
|-----------------|----------------|--------|--------|
| 0.2.x | ^0.3 | >=3.9,<3.14 | Legacy |
| 0.3.x | ^1.0 | >=3.10,<3.14 | ✅ Current |

## Python Version Support Matrix

| Python Version | langchain-redis v0.3.0 | Notes |
|----------------|------------------------|-------|
| 3.9 | ❌ Not supported | EOL October 2025 |
| 3.10 | ✅ Supported | Minimum version |
| 3.11 | ✅ Supported | Recommended |
| 3.12 | ✅ Supported | Recommended |
| 3.13 | ✅ Supported | Latest |
| 3.14 | ⏳ Coming soon | Awaiting redisvl update |

## Need Help?

- 📖 **Documentation**: [https://python.langchain.com/docs/integrations/providers/redis](https://python.langchain.com/docs/integrations/providers/redis)
- 🐛 **Issues**: [https://github.com/langchain-ai/langchain-redis/issues](https://github.com/langchain-ai/langchain-redis/issues)
- 💬 **Discord**: [LangChain Community Discord](https://discord.gg/langchain)
- 📚 **LangChain v1.0 Migration Guide**: [https://docs.langchain.com/oss/python/migrate/langchain-v1](https://docs.langchain.com/oss/python/migrate/langchain-v1)

## FAQ

### Q: Do I need to change my application code?

**A:** No! The `langchain-redis` API is unchanged. You only need to update dependencies and optionally update import paths for future compatibility.

### Q: What if I can't upgrade from Python 3.9?

**A:** Stay on `langchain-redis` v0.2.x until you can upgrade Python. Python 3.9 reaches end-of-life in October 2025, so we recommend planning your upgrade soon.

### Q: Will my existing Redis data still work?

**A:** Yes! There are no changes to how data is stored in Redis. All existing indices, caches, and chat histories will continue to work.

### Q: When will Python 3.14 be supported?

**A:** Python 3.14 support will be added in a future release (v0.4.0+) once our dependency `redisvl` adds support for Python 3.14.

### Q: Can I use langchain-redis v0.3.0 with langchain-core 0.3.x?

**A:** No, v0.3.0 requires langchain-core ^1.0. If you need to stay on langchain-core 0.3.x, use langchain-redis v0.2.x.

### Q: Are there any performance improvements in v0.3.0?

**A:** The migration to LangChain v1.0 includes upstream performance improvements and bug fixes. `langchain-redis` itself has no performance-related changes in this release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## Next Steps

After migration:

1. ✅ Update your CI/CD pipelines to use Python 3.10+
2. ✅ Update import paths in your codebase (optional but recommended)
3. ✅ Review your dependencies for other LangChain v1.0-compatible versions
4. ✅ Test thoroughly in a staging environment before production deployment

---

**Last Updated**: 2025-01-13
**Version**: langchain-redis v0.3.0
