from redisvl.version import __version__ as __redisvl_version__  # type: ignore

__version__ = "0.0.4"
__lib_name__ = f"langchain-redis_v{__version__}"
__full_lib_name__ = f"redis-py(redisvl_v{__redisvl_version__};{__lib_name__})"
