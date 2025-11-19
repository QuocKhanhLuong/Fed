"""
Redis Client Manager Module
Manages client state and metadata using Redis for stateless server architecture.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
import redis

logger = logging.getLogger(__name__)

class RedisClientManager:
    """
    Manages client state in Redis.
    Keys:
        fl:clients:active -> Set of active client IDs
        fl:client:{id}:meta -> Hash of client metadata (ip, battery, score, last_seen)
        fl:client:{id}:weights -> (Optional) Temporary storage for weights if not using S3
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis DB index
        """
        try:
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError:
            logger.warning("Redis connection failed! Using MockRedis (in-memory) for testing.")
            from unittest.mock import MagicMock
            self.redis = MagicMock()
            # Simple dict mock for basic operations
            self._mock_storage = {}
            
            # Mock methods
            def mock_hset(name, mapping):
                self._mock_storage[name] = mapping
            self.redis.hset = mock_hset
            
            def mock_hgetall(name):
                return self._mock_storage.get(name, {})
            self.redis.hgetall = mock_hgetall
            
            def mock_sadd(name, *values):
                if name not in self._mock_storage:
                    self._mock_storage[name] = set()
                for v in values:
                    self._mock_storage[name].add(v)
            self.redis.sadd = mock_sadd
            
            def mock_smembers(name):
                return self._mock_storage.get(name, set())
            self.redis.smembers = mock_smembers
            
            def mock_srem(name, *values):
                if name in self._mock_storage:
                    for v in values:
                        self._mock_storage[name].discard(v)
            self.redis.srem = mock_srem

    def register_client(self, client_id: str, metadata: Dict[str, Any]):
        """
        Register a client or update its metadata.
        
        Args:
            client_id: Unique client ID
            metadata: Client metadata (battery, cpu, etc.)
        """
        key = f"fl:client:{client_id}:meta"
        metadata['last_seen'] = time.time()
        
        # Store metadata
        # Convert non-string values to string for Redis
        safe_meta = {k: str(v) for k, v in metadata.items()}
        self.redis.hset(key, mapping=safe_meta)
        
        # Add to active set
        self.redis.sadd("fl:clients:active", client_id)
        
        logger.info(f"Registered client {client_id} in Redis")

    def update_heartbeat(self, client_id: str):
        """Update client last_seen timestamp."""
        key = f"fl:client:{client_id}:meta"
        self.redis.hset(key, mapping={'last_seen': str(time.time())})

    def get_active_clients(self, timeout: float = 30.0) -> List[str]:
        """
        Get list of currently active clients (seen within timeout).
        Also performs lazy cleanup of dead clients.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of active client IDs
        """
        all_clients = self.redis.smembers("fl:clients:active")
        active_clients = []
        dead_clients = []
        
        now = time.time()
        
        for client_id in all_clients:
            key = f"fl:client:{client_id}:meta"
            meta = self.redis.hgetall(key)
            
            if not meta:
                dead_clients.append(client_id)
                continue
                
            last_seen = float(meta.get('last_seen', 0))
            if now - last_seen < timeout:
                active_clients.append(client_id)
            else:
                dead_clients.append(client_id)
        
        # Cleanup
        if dead_clients:
            self.redis.srem("fl:clients:active", *dead_clients)
            logger.info(f"Removed {len(dead_clients)} dead clients")
            
        return active_clients

    def get_client_metadata(self, client_id: str) -> Dict[str, Any]:
        """Get metadata for a specific client."""
        key = f"fl:client:{client_id}:meta"
        return self.redis.hgetall(key)
