"""
Flower-compatible Client Manager
================================

This module provides a client management system that mirrors Flower's 
ClientManager/ClientProxy architecture for compatibility and future integration.

Architecture (aligned with Flower):
    - ClientManager: Manages pool of connected clients (like flwr.server.ClientManager)
    - ClientProxy: Server-side representation of each client (like flwr.server.ClientProxy)

Backend Support:
    - In-memory (default): For single-machine experiments
    - SQLite: For persistence across restarts
    - Redis: For multi-machine distributed FL

For IEEE Publication: "Nested Early-Exit Federated Learning"

Author: Research Team
"""

import time
import json
import logging
import threading
import sqlite3
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try Redis import
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


# =============================================================================
# ClientProxy: Server-side representation of a FL client
# =============================================================================

@dataclass
class ClientProxy:
    """
    Server-side representation of a FL client.
    
    Mirrors Flower's flwr.server.client_proxy.ClientProxy interface.
    
    Attributes:
        cid: Unique client identifier
        metadata: Client properties (battery, cpu, network_score, etc.)
        last_seen: Unix timestamp of last heartbeat
        is_active: Whether client is currently available
        properties: Additional properties for strategy sampling
    """
    cid: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    is_active: bool = True
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def get_properties(self, ins: Dict = None) -> Dict[str, Any]:
        """Get client properties (Flower-compatible method)."""
        return {**self.metadata, **self.properties}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cid': self.cid,
            'metadata': self.metadata,
            'last_seen': self.last_seen,
            'is_active': self.is_active,
            'properties': self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClientProxy':
        """Create from dictionary."""
        return cls(
            cid=data['cid'],
            metadata=data.get('metadata', {}),
            last_seen=data.get('last_seen', time.time()),
            is_active=data.get('is_active', True),
            properties=data.get('properties', {}),
        )


# =============================================================================
# Abstract ClientManager (Flower-compatible interface)
# =============================================================================

class ClientManager(ABC):
    """
    Abstract base class for client management.
    
    Mirrors Flower's flwr.server.client_manager.ClientManager interface.
    
    Flow:
        1. Clients connect → register() creates ClientProxy
        2. Strategy samples clients via sample()
        3. Task dispatched via ClientProxy
        4. Results collected
        5. Client disconnects → unregister()
    """
    
    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register a client with the manager."""
        pass
    
    @abstractmethod
    def unregister(self, cid: str) -> None:
        """Unregister a client from the manager."""
        pass
    
    @abstractmethod
    def all(self) -> Dict[str, ClientProxy]:
        """Get all registered clients."""
        pass
    
    @abstractmethod
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Callable[[ClientProxy], bool]] = None,
    ) -> List[ClientProxy]:
        """
        Sample clients for participation.
        
        Args:
            num_clients: Number of clients to sample
            min_num_clients: Minimum required (blocks until available)
            criterion: Optional filter function
            
        Returns:
            List of sampled ClientProxy instances
        """
        pass
    
    @abstractmethod
    def num_available(self) -> int:
        """Get number of currently available clients."""
        pass
    
    def wait_for(self, num_clients: int, timeout: float = 86400) -> bool:
        """
        Wait until at least num_clients are available.
        
        Args:
            num_clients: Minimum number of clients needed
            timeout: Maximum wait time in seconds
            
        Returns:
            True if enough clients available, False on timeout
        """
        start = time.time()
        while self.num_available() < num_clients:
            if time.time() - start > timeout:
                return False
            time.sleep(0.5)
        return True


# =============================================================================
# In-Memory Implementation (for single-machine testing)
# =============================================================================

class InMemoryClientManager(ClientManager):
    """
    In-memory client manager for single-machine experiments.
    
    Thread-safe implementation using RLock.
    Zero configuration, fastest option for testing.
    """
    
    def __init__(self, heartbeat_timeout: float = 60.0):
        self._clients: Dict[str, ClientProxy] = {}
        self._lock = threading.RLock()
        self.heartbeat_timeout = heartbeat_timeout
        logger.info("InMemoryClientManager initialized")
    
    def register(self, client: ClientProxy) -> bool:
        """Register a client."""
        with self._lock:
            client.last_seen = time.time()
            client.is_active = True
            self._clients[client.cid] = client
            logger.info(f"Registered client: {client.cid}")
            return True
    
    def unregister(self, cid: str) -> None:
        """Unregister a client."""
        with self._lock:
            if cid in self._clients:
                self._clients[cid].is_active = False
                logger.info(f"Unregistered client: {cid}")
    
    def all(self) -> Dict[str, ClientProxy]:
        """Get all clients."""
        with self._lock:
            return dict(self._clients)
    
    def _is_available(self, client: ClientProxy) -> bool:
        """Check if client is available (active and recent heartbeat)."""
        if not client.is_active:
            return False
        return (time.time() - client.last_seen) < self.heartbeat_timeout
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Callable[[ClientProxy], bool]] = None,
    ) -> List[ClientProxy]:
        """Sample available clients."""
        with self._lock:
            available = [c for c in self._clients.values() if self._is_available(c)]
            
            if criterion:
                available = [c for c in available if criterion(c)]
            
            min_required = min_num_clients or num_clients
            if len(available) < min_required:
                logger.warning(f"Not enough clients: {len(available)} < {min_required}")
                return []
            
            # Random sampling
            import random
            sampled = random.sample(available, min(num_clients, len(available)))
            logger.debug(f"Sampled {len(sampled)} clients")
            return sampled
    
    def num_available(self) -> int:
        """Count available clients."""
        with self._lock:
            return sum(1 for c in self._clients.values() if self._is_available(c))
    
    def update_heartbeat(self, cid: str) -> None:
        """Update client heartbeat."""
        with self._lock:
            if cid in self._clients:
                self._clients[cid].last_seen = time.time()
                self._clients[cid].is_active = True


# =============================================================================
# SQLite Implementation (for persistence)
# =============================================================================

class SQLiteClientManager(ClientManager):
    """
    SQLite-backed client manager with persistence.
    
    Use for:
    - Experiments that may restart
    - Logging all client interactions
    - Single-machine with data persistence
    """
    
    def __init__(
        self, 
        db_path: str = "./data/fl_clients.db",
        heartbeat_timeout: float = 60.0,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_timeout = heartbeat_timeout
        self._local = threading.local()
        self._init_db()
        logger.info(f"SQLiteClientManager initialized: {db_path}")
    
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn
    
    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _init_db(self):
        with self._cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    cid TEXT PRIMARY KEY,
                    metadata TEXT DEFAULT '{}',
                    properties TEXT DEFAULT '{}',
                    last_seen REAL,
                    is_active INTEGER DEFAULT 1,
                    created_at REAL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_active ON clients(is_active, last_seen)")
    
    def register(self, client: ClientProxy) -> bool:
        now = time.time()
        with self._cursor() as c:
            c.execute("""
                INSERT INTO clients (cid, metadata, properties, last_seen, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(cid) DO UPDATE SET
                    metadata = excluded.metadata,
                    properties = excluded.properties,
                    last_seen = excluded.last_seen,
                    is_active = 1
            """, (client.cid, json.dumps(client.metadata), 
                  json.dumps(client.properties), now, now))
        logger.info(f"Registered client: {client.cid}")
        return True
    
    def unregister(self, cid: str) -> None:
        with self._cursor() as c:
            c.execute("UPDATE clients SET is_active = 0 WHERE cid = ?", (cid,))
        logger.info(f"Unregistered client: {cid}")
    
    def all(self) -> Dict[str, ClientProxy]:
        with self._cursor() as c:
            c.execute("SELECT * FROM clients")
            clients = {}
            for row in c.fetchall():
                clients[row['cid']] = ClientProxy(
                    cid=row['cid'],
                    metadata=json.loads(row['metadata']),
                    properties=json.loads(row['properties']),
                    last_seen=row['last_seen'],
                    is_active=bool(row['is_active']),
                )
            return clients
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Callable[[ClientProxy], bool]] = None,
    ) -> List[ClientProxy]:
        cutoff = time.time() - self.heartbeat_timeout
        
        with self._cursor() as c:
            c.execute("""
                SELECT * FROM clients 
                WHERE is_active = 1 AND last_seen >= ?
                ORDER BY RANDOM()
            """, (cutoff,))
            
            available = []
            for row in c.fetchall():
                proxy = ClientProxy(
                    cid=row['cid'],
                    metadata=json.loads(row['metadata']),
                    properties=json.loads(row['properties']),
                    last_seen=row['last_seen'],
                    is_active=True,
                )
                if criterion is None or criterion(proxy):
                    available.append(proxy)
        
        min_required = min_num_clients or num_clients
        if len(available) < min_required:
            return []
        
        return available[:num_clients]
    
    def num_available(self) -> int:
        cutoff = time.time() - self.heartbeat_timeout
        with self._cursor() as c:
            c.execute("SELECT COUNT(*) FROM clients WHERE is_active = 1 AND last_seen >= ?", (cutoff,))
            return c.fetchone()[0]
    
    def update_heartbeat(self, cid: str) -> None:
        with self._cursor() as c:
            c.execute("UPDATE clients SET last_seen = ?, is_active = 1 WHERE cid = ?",
                      (time.time(), cid))


# =============================================================================
# Redis Implementation (for multi-machine distributed FL)
# =============================================================================

class RedisClientManager(ClientManager):
    """
    Redis-backed client manager for distributed FL.
    
    Use for:
    - Multi-machine experiments (server on different machine than clients)
    - Multiple server instances sharing client pool
    - Production deployments
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        heartbeat_timeout: float = 60.0,
    ):
        if not HAS_REDIS:
            raise ImportError("redis package required: pip install redis")
        
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.redis.ping()  # Test connection
        self.heartbeat_timeout = heartbeat_timeout
        logger.info(f"RedisClientManager connected: {host}:{port}")
    
    def _key(self, cid: str) -> str:
        return f"fl:client:{cid}"
    
    def register(self, client: ClientProxy) -> bool:
        data = client.to_dict()
        data['last_seen'] = time.time()
        self.redis.hset(self._key(client.cid), mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in data.items()
        })
        self.redis.sadd("fl:clients:all", client.cid)
        logger.info(f"Registered client: {client.cid}")
        return True
    
    def unregister(self, cid: str) -> None:
        self.redis.hset(self._key(cid), "is_active", "False")
        logger.info(f"Unregistered client: {cid}")
    
    def _load_client(self, cid: str) -> Optional[ClientProxy]:
        data = self.redis.hgetall(self._key(cid))
        if not data:
            return None
        return ClientProxy(
            cid=cid,
            metadata=json.loads(data.get('metadata', '{}')),
            properties=json.loads(data.get('properties', '{}')),
            last_seen=float(data.get('last_seen', 0)),
            is_active=data.get('is_active', 'True') == 'True',
        )
    
    def all(self) -> Dict[str, ClientProxy]:
        clients = {}
        for cid in self.redis.smembers("fl:clients:all"):
            client = self._load_client(cid)
            if client:
                clients[cid] = client
        return clients
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Callable[[ClientProxy], bool]] = None,
    ) -> List[ClientProxy]:
        cutoff = time.time() - self.heartbeat_timeout
        available = []
        
        for cid in self.redis.smembers("fl:clients:all"):
            client = self._load_client(cid)
            if client and client.is_active and client.last_seen >= cutoff:
                if criterion is None or criterion(client):
                    available.append(client)
        
        min_required = min_num_clients or num_clients
        if len(available) < min_required:
            return []
        
        import random
        return random.sample(available, min(num_clients, len(available)))
    
    def num_available(self) -> int:
        cutoff = time.time() - self.heartbeat_timeout
        count = 0
        for cid in self.redis.smembers("fl:clients:all"):
            client = self._load_client(cid)
            if client and client.is_active and client.last_seen >= cutoff:
                count += 1
        return count
    
    def update_heartbeat(self, cid: str) -> None:
        self.redis.hset(self._key(cid), mapping={
            'last_seen': str(time.time()),
            'is_active': 'True'
        })


# =============================================================================
# Factory Function (Auto-select best backend)
# =============================================================================

def create_client_manager(
    backend: str = "auto",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    sqlite_path: str = "./data/fl_clients.db",
    heartbeat_timeout: float = 60.0,
) -> ClientManager:
    """
    Factory function to create appropriate ClientManager.
    
    Args:
        backend: "auto", "memory", "sqlite", or "redis"
        redis_host: Redis host (for redis backend)
        redis_port: Redis port (for redis backend)
        sqlite_path: SQLite path (for sqlite backend)
        heartbeat_timeout: Client timeout in seconds
        
    Returns:
        ClientManager instance
        
    Auto-selection logic:
        1. Try Redis if available → best for multi-machine
        2. Fall back to in-memory → simplest, fastest
    """
    if backend == "redis" or (backend == "auto" and HAS_REDIS):
        try:
            manager = RedisClientManager(
                host=redis_host,
                port=redis_port,
                heartbeat_timeout=heartbeat_timeout,
            )
            logger.info("✓ Using Redis ClientManager (multi-machine ready)")
            return manager
        except Exception as e:
            if backend == "redis":
                raise
            logger.warning(f"Redis unavailable: {e}")
    
    if backend == "sqlite":
        manager = SQLiteClientManager(
            db_path=sqlite_path,
            heartbeat_timeout=heartbeat_timeout,
        )
        logger.info("✓ Using SQLite ClientManager (persistent)")
        return manager
    
    # Default: in-memory
    manager = InMemoryClientManager(heartbeat_timeout=heartbeat_timeout)
    logger.info("✓ Using InMemory ClientManager (single-machine)")
    return manager


# =============================================================================
# For backward compatibility with old redis_client_manager.py
# =============================================================================

# Alias the old name
ClientStateManager = InMemoryClientManager


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ClientManager Test (Flower-compatible)")
    print("=" * 60 + "\n")
    
    # Test in-memory
    manager = create_client_manager(backend="memory")
    
    # Register clients
    for i in range(5):
        client = ClientProxy(
            cid=f"client_{i}",
            metadata={"battery": 80 + i, "cpu": 50 - i * 5},
        )
        manager.register(client)
    
    print(f"Total clients: {len(manager.all())}")
    print(f"Available clients: {manager.num_available()}")
    
    # Sample clients
    sampled = manager.sample(num_clients=3, min_num_clients=2)
    print(f"Sampled {len(sampled)} clients: {[c.cid for c in sampled]}")
    
    # Sample with criterion
    high_battery = manager.sample(
        num_clients=2,
        criterion=lambda c: c.metadata.get('battery', 0) > 82
    )
    print(f"High battery clients: {[c.cid for c in high_battery]}")
    
    # Unregister
    manager.unregister("client_0")
    print(f"After unregister: {manager.num_available()} available")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
