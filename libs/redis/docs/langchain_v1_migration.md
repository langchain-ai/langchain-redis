# Migration Guide: LangChain v1.0

## Overview

`langchain-redis` v0.3.0 adds support for LangChain v1.0. This guide helps you migrate your applications to use the latest version.

## What Changed

### Breaking Changes

#### 1. Python Version Requirement

**Minimum Python version is now 3.10+**

- **Python 3.9 is no longer supported** (reaches end-of-life October 2025)
- **Supported versions**: Python 3.10, 3.11, 3.12, 3.13

**Action Required:**
- If you're on Python 3.9, upgrade to Python 3.10 or higher
- If you're on Python 3.10-3.13, no action needed

#### 2. Dependency Updates

**Updated to LangChain v1.0**

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
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Update Dependencies

**Option A: Using pip**
```bash
pip install --upgrade langchain-redis langchain-core
```

**Option B: Using poetry**
```bash
# Update pyproject.toml
# python = ">=3.10,<3.14"
# langchain-core = "^1.0"

poetry lock
poetry install
```

**Option C: Using requirements.txt**
```txt
langchain-redis>=0.3.0
langchain-core>=1.0
```

### Step 4: Test Your Application

```bash
# Run your test suite
pytest tests/

# Verify your application works
python your_app.py
```

## Example Migration

### Before (works with both v0.2.x and v0.3.x):

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

### After (recommended for v0.3.0+):

```python
# Note that `langchain` became `langchain_core`
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation
from langchain_openai import OpenAI
from langchain_redis import RedisCache

# Initialize cache (no changes needed)
cache = RedisCache(redis_url="redis://localhost:6379")
set_llm_cache(cache)

# Use as normal (no changes needed)
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
```
ERROR: Package 'langchain-redis' requires a different Python: 3.9.x not in '>=3.10,<3.14'
```

**Solution:**
Upgrade to Python 3.10 or higher (see Step 1 above).

### Issue: Dependency resolver conflicts

**Symptom:**
```
ERROR: Cannot install langchain-redis and langchain-core because these package versions have conflicting dependencies.
```

**Solution:**
```bash
# Clear dependency cache
pip cache purge

# Install with updated resolver
pip install --upgrade --force-reinstall langchain-redis langchain-core
```

## FAQ

### Q: Do I need to change my application code?

**A:** No! The `langchain-redis` API is unchanged. You only need to update dependencies and optionally update import paths for future compatibility.

### Q: What if I can't upgrade from Python 3.9?

**A:** Stay on `langchain-redis` v0.3.x until you can upgrade Python. Python 3.9 reaches end-of-life in October 2025, so we recommend planning your upgrade soon.

### Q: Will my existing Redis data still work?

**A:** Yes! There are no changes to how data is stored in Redis. All existing indices, caches, and chat histories will continue to work.

### Q: Can I use langchain-redis v0.3.0 with langchain-core 0.3.x?

**A:** No, v0.3.0 requires langchain-core ^1.0. If you need to stay on langchain-core 0.3.x, use langchain-redis v0.2.x.

### Q: Are there any performance improvements in v0.3.0?

**A:** The migration to LangChain v1.0 includes upstream performance improvements and bug fixes. `langchain-redis` itself has no performance-related changes in this release.

