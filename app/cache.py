"""
Redis caching module for MaZe application.
Provides caching for geocoding results and OSM queries with 1-year TTL.
Falls back to in-memory cache if Redis is unavailable.
"""

import json
from hashlib import md5
from typing import Optional, Any
from functools import lru_cache

# Cache TTL: 1 year in seconds
CACHE_TTL = 3600 * 24 * 365

# Try to import Redis, fall back to in-memory cache
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis cache connected")
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False
    print("⚠️ Redis unavailable, using in-memory LRU cache")

# In-memory fallback cache
_memory_cache = {}


def cache_key(prefix: str, *args) -> str:
    """Generate a cache key from prefix and arguments."""
    data = json.dumps(args, sort_keys=True)
    return f"{prefix}:{md5(data.encode()).hexdigest()}"


def get_cached(key: str) -> Optional[Any]:
    """Get a value from cache."""
    if REDIS_AVAILABLE:
        try:
            value = redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"Redis get error: {e}")
    else:
        return _memory_cache.get(key)
    return None


def set_cached(key: str, value: Any) -> bool:
    """Set a value in cache with TTL."""
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(key, CACHE_TTL, json.dumps(value))
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    else:
        _memory_cache[key] = value
        # Limit in-memory cache size
        if len(_memory_cache) > 1000:
            # Remove oldest entries (simple LRU)
            keys_to_remove = list(_memory_cache.keys())[:100]
            for k in keys_to_remove:
                del _memory_cache[k]
        return True


def delete_cached(key: str) -> bool:
    """Delete a value from cache."""
    if REDIS_AVAILABLE:
        try:
            redis_client.delete(key)
            return True
        except Exception:
            return False
    else:
        _memory_cache.pop(key, None)
        return True


def cache_geocode(address: str) -> Optional[tuple]:
    """Get cached geocoding result for an address."""
    key = cache_key("geocode", address.lower().strip())
    return get_cached(key)


def set_cache_geocode(address: str, coords: tuple) -> bool:
    """Cache a geocoding result."""
    key = cache_key("geocode", address.lower().strip())
    return set_cached(key, coords)


def cache_amenities(lat: float, lon: float, dist: int, feature_weights: dict) -> Optional[dict]:
    """Get cached amenities result."""
    # Round coordinates to reduce cache fragmentation
    lat_rounded = round(lat, 4)
    lon_rounded = round(lon, 4)
    key = cache_key("amenities", lat_rounded, lon_rounded, dist, feature_weights)
    return get_cached(key)


def set_cache_amenities(lat: float, lon: float, dist: int, feature_weights: dict, results: dict) -> bool:
    """Cache amenities result."""
    lat_rounded = round(lat, 4)
    lon_rounded = round(lon, 4)
    key = cache_key("amenities", lat_rounded, lon_rounded, dist, feature_weights)
    return set_cached(key, results)
