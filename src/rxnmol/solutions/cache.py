"""
Persistent Reaction Cache with Multi-Process Support.

Architecture (2-tier lookup, 2-tier write):

    READS:  Memory Cache -> Shared DB (read-only, immutable, copied to local storage)
    WRITES: Memory Cache -> Local DB (for post-run merge)

This design eliminates SQLite locking issues when running 100+ concurrent experiments:
- Shared DB is copied to local node storage (/tmp) to avoid NFS contention
- Shared DB is opened read-only with immutable=1 (no WAL/SHM access, no Bus errors)
- Each process writes to its own local DB (no contention)
- Memory cache provides fast access to hot data
- Local DB provides durability for post-run merge
- After experiments complete, merge local DBs into shared DB

Note: Local DB is NOT checked during reads (it's empty at start).
      It only serves as durable storage for merging after the run.

Usage:
    # During experiment
    cache = PersistentReactionCache(
        shared_db_path="./cache/reaction_cache_shared.db",
        run_name="seh_mf2-5_r1"  # Used for local DB naming
    )

    # At end of experiment (optional but recommended)
    cache.close()

    # After ALL experiments complete, run merge script
    merge_local_caches_to_shared("./cache/local", "./cache/reaction_cache_shared.db")
"""

import sqlite3
import os
import atexit
import hashlib
import shutil
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging
import time
import fcntl
import tempfile
import threading

# Try to import filelock, fall back to fcntl-based locking if not available
try:
    import filelock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False

logger = logging.getLogger(__name__)


class PersistentReactionCache:
    """
    A persistent cache for reaction predictions using SQLite.

    Designed for high-concurrency scenarios (100+ simultaneous experiments):
    - Shared DB is read-only (no locking issues)
    - Each process has its own local DB for writes
    - Memory cache for fast repeated lookups

    Attributes:
        shared_db_path: Path to the shared (read-only) database
        local_db_path: Path to the process-local database
        _memory_cache: In-memory dict for fastest access
        _shared_conn: Read-only connection to shared DB
        _local_conn: Read-write connection to local DB
    """

    def __init__(
        self,
        shared_db_path: str = "./cache/reaction_cache_shared.db",
        local_db_dir: str = "./cache/local",
        run_name: str = None,
        preload_shared: bool = False,
        copy_shared_to_local: bool = True,
        local_storage_dir: str = "/tmp",
    ):
        """
        Initialize the cache.

        Args:
            shared_db_path: Path to shared database (read-only access)
            local_db_dir: Directory for process-local databases
            run_name: Name for local DB file (e.g., "seh_mf2-5_r1").
                      If None, uses PID_timestamp for uniqueness.
            preload_shared: If True, load entire shared DB into memory at startup
            copy_shared_to_local: If True, copy shared DB to local node storage
                                  before opening (avoids NFS contention)
            local_storage_dir: Directory for local copy (default: /tmp)
        """
        self._memory_cache: Dict[str, str] = {}
        self._closed = False
        # Runtime-local DB path (lives on node-local storage for safety/speed)
        self._local_db_persist_path: Optional[str] = None

        # Shared DB (read-only)
        env_override = os.environ.get("RXN_SHARED_CACHE_PATH") or os.environ.get("RXNMOL_SHARED_CACHE_PATH")
        self.shared_db_path = env_override or shared_db_path
        if env_override:
            logger.info(f"Using shared cache path from env: {self.shared_db_path}")
        self._shared_db_local_copy: Optional[str] = None  # Path to local copy if used
        self._shared_conn: Optional[sqlite3.Connection] = None
        self._init_shared_db(copy_shared_to_local, local_storage_dir)

        # Local DB (per-process, write-only for durability)
        os.makedirs(local_db_dir, exist_ok=True)
        if run_name:
            # Use meaningful name based on run (allows potential resume)
            # Sanitize run_name to be filesystem-safe
            safe_name = run_name.replace("/", "_").replace("\\", "_")
            self.local_db_path = os.path.join(local_db_dir, f"cache_{safe_name}.db")
        else:
            # Fallback: use PID + timestamp for uniqueness
            pid = os.getpid()
            timestamp = int(time.time() * 1000)
            self.local_db_path = os.path.join(local_db_dir, f"cache_{pid}_{timestamp}.db")

        # Run-time location: prefer node-local tmp for safety (no WAL on NFS)
        runtime_local_root = (
            os.environ.get("RXNMOL_CACHE_RUNTIME_DIR")
            or os.environ.get("SLURM_TMPDIR")
            or tempfile.gettempdir()
        )
        os.makedirs(runtime_local_root, exist_ok=True)
        runtime_local_path = os.path.join(runtime_local_root, os.path.basename(self.local_db_path))
        self._local_db_persist_path = self.local_db_path  # original target under ./cache/local
        self.local_db_path = runtime_local_path  # actual live DB path during the run

        self._local_conn: Optional[sqlite3.Connection] = None
        self._init_local_db()

        # Optionally preload shared DB into memory
        if preload_shared and self._shared_conn:
            self._preload_shared_to_memory()

        # Register cleanup on exit
        atexit.register(self.close)

        logger.info(
            f"ReactionCache initialized: "
            f"shared={self.shared_db_path} (exists={os.path.exists(self.shared_db_path)}), "
            f"local={self.local_db_path}"
        )

    def _init_shared_db(self, copy_to_local: bool = True, local_storage_dir: str = "/tmp"):
        """Open shared DB in read-only mode (no locking).

        Args:
            copy_to_local: If True, copy shared DB to local storage first
            local_storage_dir: Directory for local copy (default: /tmp)
        """
        logger.info(f"Initializing shared reaction cache from: {self.shared_db_path}")
        if not os.path.exists(self.shared_db_path):
            logger.info(f"Shared cache not found at {self.shared_db_path}, will use local only")
            return

        # Determine which path to use for reading
        if copy_to_local:
            db_path = self._copy_shared_to_local_storage(local_storage_dir)
            if db_path is None:
                # Copy failed, fall back to original path
                db_path = self.shared_db_path
                # Note: detailed error already logged by _copy_shared_to_local_storage
                logger.warning(f"Using NFS path directly (see error above): {db_path}")
        else:
            db_path = self.shared_db_path

        # Timeout for NFS connection (in seconds)
        # NFS can be slow under heavy load, don't wait forever
        NFS_CONNECT_TIMEOUT = 120  # 2 minutes max

        def _connect_with_timeout():
            """Connect to shared DB with timeout using a worker thread."""

            def do_connect():
                """Inner connect helper (no timeouts here)."""
                # SQLite URI mode with read-only + immutable flags
                # immutable=1: Skips WAL/SHM file access entirely (avoids Bus errors on NFS)
                uri = f"file:{db_path}?mode=ro&immutable=1"
                # check_same_thread=False allows connection to be used from any thread
                # This is safe because the DB is opened read-only and immutable
                conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
                # Disable memory-mapped I/O to avoid Bus errors on NFS/network storage
                conn.execute("PRAGMA mmap_size=0")
                # Verify table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='reactions'"
                )
                if cursor.fetchone() is None:
                    logger.warning(f"Shared DB exists but has no 'reactions' table from {uri}")
                    conn.close()
                    return None

                source = "local copy" if db_path != self.shared_db_path else "NFS"
                logger.info(f"Connected to shared cache ({source}): {db_path}")
                return conn

            result_holder = {}

            def worker():
                try:
                    result_holder["conn"] = do_connect()
                except Exception as e:
                    result_holder["error"] = e

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(NFS_CONNECT_TIMEOUT)

            if t.is_alive():
                raise TimeoutError(
                    f"Connection to shared DB timed out after {NFS_CONNECT_TIMEOUT}s"
                )
            if "error" in result_holder:
                raise result_holder["error"]
            return result_holder.get("conn")

        try:
            self._shared_conn = _connect_with_timeout()
        except TimeoutError as e:
            logger.warning(f"Shared DB connection timeout: {e} - will continue without shared cache")
            self._shared_conn = None
        except Exception as e:
            logger.warning(f"Could not open shared cache (will continue without): {e}")
            self._shared_conn = None

    def _copy_shared_to_local_storage(self, local_storage_dir: str) -> Optional[str]:
        """Copy shared DB to local node storage to avoid NFS contention.

        Uses file locking to ensure only one process per node copies the file.
        Other processes wait for the copy to complete.

        Args:
            local_storage_dir: Directory for local copy (e.g., /tmp)

        Returns:
            Path to local copy, or None if copy failed
        """
        # Create a unique filename based on the original path and file modification time
        # This ensures we get a fresh copy if the source file changes
        try:
            source_stat = os.stat(self.shared_db_path)
            source_mtime = int(source_stat.st_mtime)
            source_size = source_stat.st_size
        except OSError as e:
            logger.warning(f"Cannot stat shared DB: {e}")
            return None

        # Hash the source path to create a unique but consistent filename
        path_hash = hashlib.md5(os.path.abspath(self.shared_db_path).encode()).hexdigest()[:12]
        local_filename = f"reaction_cache_{path_hash}_{source_mtime}.db"
        local_path = os.path.join(local_storage_dir, local_filename)
        lock_path = local_path + ".lock"

        self._shared_db_local_copy = local_path

        # Check if local copy already exists and is valid
        if os.path.exists(local_path):
            try:
                local_size = os.path.getsize(local_path)
                if local_size == source_size:
                    logger.info(f"Using existing local copy: {local_path}")
                    return local_path
                else:
                    logger.warning(f"Local copy size mismatch ({local_size} vs {source_size}), will re-copy")
            except OSError:
                pass

        # Need to copy - use file locking to prevent race conditions
        logger.info(f"Copying shared DB to local storage: {self.shared_db_path} -> {local_path}")

        def do_copy_with_lock():
            """Perform the copy while holding a lock."""
            # Double-check after acquiring lock (another process may have copied)
            if os.path.exists(local_path):
                try:
                    local_size = os.path.getsize(local_path)
                    if local_size == source_size:
                        logger.info(f"Another process completed copy: {local_path}")
                        return local_path
                except OSError:
                    pass

            # Perform the copy
            t0 = time.time()
            temp_path = local_path + f".tmp.{os.getpid()}"

            try:
                # Check available space before copying
                try:
                    stat_info = os.statvfs(local_storage_dir)
                    available_bytes = stat_info.f_bavail * stat_info.f_frsize
                    if available_bytes < source_size * 1.1:  # Need 10% buffer
                        raise OSError(
                            28,  # ENOSPC
                            f"Insufficient space in {local_storage_dir}: "
                            f"need {source_size/(1024**3):.1f}GB, have {available_bytes/(1024**3):.1f}GB"
                        )
                except OSError as space_err:
                    if space_err.errno == 28 or "Insufficient space" in str(space_err):
                        raise
                    # statvfs might fail on some filesystems, continue anyway
                    pass

                shutil.copy2(self.shared_db_path, temp_path)
                os.rename(temp_path, local_path)  # Atomic rename
                elapsed = time.time() - t0
                size_mb = source_size / (1024 * 1024)
                logger.info(f"Copied shared DB to local storage in {elapsed:.1f}s ({size_mb:.1f} MB)")
                return local_path
            except Exception as e:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise

        # Lock timeout: don't wait too long - NFS fallback works fine
        LOCK_TIMEOUT_SECONDS = 120  # 2 minutes max before falling back

        try:
            if HAS_FILELOCK:
                # Use filelock for cross-process synchronization
                lock = filelock.FileLock(lock_path, timeout=LOCK_TIMEOUT_SECONDS)
                with lock:
                    return do_copy_with_lock()
            else:
                # Fallback to fcntl-based locking
                lock_fd = open(lock_path, 'w')
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                    return do_copy_with_lock()
                finally:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                    lock_fd.close()

        except Exception as e:
            if HAS_FILELOCK and isinstance(e, filelock.Timeout):
                # Lock timeout - another process is still copying
                # Check if the file was completed by another process during our wait
                if os.path.exists(local_path):
                    try:
                        local_size = os.path.getsize(local_path)
                        if local_size == source_size:
                            logger.info(f"Lock timeout but file ready (copied by another process): {local_path}")
                            return local_path
                    except OSError:
                        pass
                logger.info(
                    f"Copy still in progress after {LOCK_TIMEOUT_SECONDS//60} min, using NFS directly"
                )
            elif isinstance(e, OSError):
                # Common OS errors: disk full, permission denied, etc.
                logger.warning(
                    f"OS error copying to local storage: {type(e).__name__}: {e} "
                    f"(errno={getattr(e, 'errno', 'N/A')}, path={local_storage_dir})"
                )
            else:
                # Unexpected error - include more details
                import traceback
                logger.warning(
                    f"Failed to copy shared DB to local storage: {type(e).__name__}: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
            return None

    def _init_local_db(self):
        """Initialize local (per-process) DB for writes.

        If local DB already exists (from a crashed run with same run_name),
        load its contents into memory for resume capability.
        """
        local_exists = os.path.exists(self.local_db_path)

        try:
            self._local_conn = sqlite3.connect(self.local_db_path, timeout=30.0)
            # Use DELETE journal (NFS-safe, no mmap’d WAL/SHM files)
            self._local_conn.execute("PRAGMA journal_mode=DELETE;")
            self._local_conn.execute("PRAGMA synchronous=NORMAL;")
            self._local_conn.execute("PRAGMA mmap_size=0;")
            self._local_conn.execute("""
                CREATE TABLE IF NOT EXISTS reactions (
                    reactants TEXT PRIMARY KEY,
                    product TEXT
                )
            """)
            # Index for faster lookups
            self._local_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reactants ON reactions(reactants)"
            )
            self._local_conn.commit()

            # If local DB existed (resume scenario), load into memory
            if local_exists:
                self._load_local_to_memory()
            else:
                logger.debug(f"Local cache created: {self.local_db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize local cache: {e}")
            raise

    def _load_local_to_memory(self):
        """Load existing local DB entries into memory (for resume)."""
        if self._local_conn is None:
            return

        try:
            cursor = self._local_conn.execute("SELECT COUNT(*) FROM reactions")
            count = cursor.fetchone()[0]

            if count > 0:
                t0 = time.time()
                cursor = self._local_conn.execute("SELECT reactants, product FROM reactions")
                for row in cursor:
                    self._memory_cache[row[0]] = row[1]
                logger.info(f"Resumed from local cache: loaded {count:,} entries in {time.time()-t0:.2f}s")
            else:
                logger.debug(f"Local cache exists but empty: {self.local_db_path}")
        except Exception as e:
            logger.warning(f"Failed to load local cache for resume: {e}")

    def _preload_shared_to_memory(self):
        """Load entire shared DB into memory for fastest access."""
        if self._shared_conn is None:
            return

        try:
            t0 = time.time()
            cursor = self._shared_conn.execute("SELECT reactants, product FROM reactions")
            count = 0
            for row in cursor:
                self._memory_cache[row[0]] = row[1]
                count += 1
            logger.info(f"Preloaded {count:,} entries from shared DB in {time.time()-t0:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to preload shared DB: {e}")

    # =========================================================================
    # Core API
    # =========================================================================

    def get(self, reactants: str) -> Optional[str]:
        """
        Retrieve a product from the cache.

        Lookup order: Memory -> Shared DB

        Note: Local DB is not checked during runtime (it was loaded to memory
        at startup if it existed from a previous run).
        """
        if self._closed:
            raise RuntimeError("Cache is closed")

        # 1. Memory cache (fastest)
        # Contains: entries from shared DB + entries from resumed local DB + new predictions
        if reactants in self._memory_cache:
            return self._memory_cache[reactants]

        # 2. Shared DB (read-only, immutable - no WAL/SHM access)
        if self._shared_conn:
            try:
                cursor = self._shared_conn.execute(
                    "SELECT product FROM reactions WHERE reactants = ?", (reactants,)
                )
                row = cursor.fetchone()
                if row:
                    self._memory_cache[reactants] = row[0]
                    return row[0]
            except Exception as e:
                logger.debug(f"Shared DB read error: {e}")

        return None

    def set(self, reactants: str, product: str):
        """
        Store a reaction result.

        Writes to: Memory + Local DB (no shared DB writes during experiment)
        """
        if self._closed:
            raise RuntimeError("Cache is closed")

        # 1. Memory cache
        self._memory_cache[reactants] = product

        # 2. Local DB
        if self._local_conn:
            try:
                self._local_conn.execute(
                    "INSERT OR REPLACE INTO reactions (reactants, product) VALUES (?, ?)",
                    (reactants, product)
                )
                self._local_conn.commit()
            except Exception as e:
                logger.warning(f"Local DB write error: {e}")

    def get_batch(self, reactants_list: List[str]) -> List[Optional[str]]:
        """
        Retrieve a batch of products.

        Lookup order: Memory -> Shared DB
        (Local DB was loaded to memory at startup if resuming)
        """
        if self._closed:
            raise RuntimeError("Cache is closed")

        if not reactants_list:
            return []

        results = [None] * len(reactants_list)
        missing = []  # (original_index, reactants) - not in memory

        # Tier 1: Memory cache
        for i, r in enumerate(reactants_list):
            if r in self._memory_cache:
                results[i] = self._memory_cache[r]
            else:
                missing.append((i, r))

        if not missing:
            return results

        # Tier 2: Shared DB (read-only, immutable)
        if self._shared_conn:
            try:
                for idx, r in missing:
                    cursor = self._shared_conn.execute(
                        "SELECT product FROM reactions WHERE reactants = ?", (r,)
                    )
                    row = cursor.fetchone()
                    if row:
                        results[idx] = row[0]
                        self._memory_cache[r] = row[0]
            except Exception as e:
                logger.debug(f"Shared DB batch read error: {e}")

        return results

    def set_batch(self, reactants_list: List[str], products_list: List[str]):
        """
        Store a batch of reaction results.

        Writes to: Memory + Local DB
        """
        if self._closed:
            raise RuntimeError("Cache is closed")

        if not reactants_list:
            return

        # 1. Memory cache
        for r, p in zip(reactants_list, products_list):
            self._memory_cache[r] = p

        # 2. Local DB (batch insert)
        if self._local_conn:
            try:
                self._local_conn.executemany(
                    "INSERT OR REPLACE INTO reactions (reactants, product) VALUES (?, ?)",
                    list(zip(reactants_list, products_list))
                )
                self._local_conn.commit()
            except Exception as e:
                logger.warning(f"Local DB batch write error: {e}")

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def close(self):
        """Close all database connections."""
        if self._closed:
            return

        self._closed = True

        if self._local_conn:
            try:
                self._local_conn.close()
                logger.debug(f"Closed local cache: {self.local_db_path}")
            except Exception as e:
                logger.warning(f"Error closing local cache: {e}")
            self._local_conn = None

        # Copy runtime local DB back to persistent location (for merge)
        if (
            self.local_db_path
            and self._local_db_persist_path
            and os.path.abspath(self.local_db_path) != os.path.abspath(self._local_db_persist_path)
            and os.path.exists(self.local_db_path)
        ):
            try:
                os.makedirs(os.path.dirname(self._local_db_persist_path) or ".", exist_ok=True)
                shutil.copy2(self.local_db_path, self._local_db_persist_path)
                logger.info(
                    f"Copied local cache to persistent dir: {self.local_db_path} -> {self._local_db_persist_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to copy local cache to persistent dir: {e}")

        if self._shared_conn:
            try:
                self._shared_conn.close()
            except Exception as e:
                logger.warning(f"Error closing shared cache: {e}")
            self._shared_conn = None

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self._memory_cache),
            "shared_db_connected": self._shared_conn is not None,
            "shared_db_path": self.shared_db_path,
            "local_db_path": self.local_db_path,
        }

        if self._local_conn:
            try:
                cursor = self._local_conn.execute("SELECT COUNT(*) FROM reactions")
                stats["local_db_entries"] = cursor.fetchone()[0]
            except:
                stats["local_db_entries"] = "error"

        if self._shared_conn:
            try:
                cursor = self._shared_conn.execute("SELECT COUNT(*) FROM reactions")
                stats["shared_db_entries"] = cursor.fetchone()[0]
            except:
                stats["shared_db_entries"] = "error"

        return stats

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Merge Utilities (Run AFTER all experiments complete)
# =============================================================================

def merge_local_caches_to_shared(
    local_db_dir: str = "./cache/local",
    shared_db_path: str = "./cache/reaction_cache_shared.db",
    delete_after_merge: bool = False,
    batch_size: int = 10000,
) -> Dict:
    """
    Merge all local cache databases into the shared database.

    IMPORTANT: Run this AFTER all experiments have completed to avoid
    conflicts with running processes.

    Args:
        local_db_dir: Directory containing local cache files (cache_*.db)
        shared_db_path: Path to the shared database
        delete_after_merge: If True, delete local DBs after successful merge
        batch_size: Number of rows to insert per transaction

    Returns:
        Dict with merge statistics

    Edge cases handled:
        - Duplicate entries (keeps existing in shared DB - first write wins)
        - Corrupted local DBs (skipped with warning)
        - Empty local DBs (skipped)
        - Concurrent merge attempts (uses file locking)
        - Partial failures (transaction-safe)
    """
    import glob

    stats = {
        "local_files_found": 0,
        "local_files_merged": 0,
        "local_files_skipped": 0,
        "local_files_deleted": 0,
        "total_entries_processed": 0,
        "new_entries_added": 0,
        "duplicate_entries_skipped": 0,
        "errors": [],
    }

    # Find all local cache files
    pattern = os.path.join(local_db_dir, "cache_*.db")
    local_files = glob.glob(pattern)
    stats["local_files_found"] = len(local_files)

    if not local_files:
        logger.info(f"No local cache files found in {local_db_dir}")
        return stats

    logger.info(f"Found {len(local_files)} local cache files to merge")

    # Create lock file for exclusive access during merge
    lock_file_path = shared_db_path + ".merge.lock"

    try:
        # Acquire exclusive lock
        lock_file = open(lock_file_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.error("Another merge process is running. Aborting.")
            stats["errors"].append("Merge lock held by another process")
            return stats

        # Open/create shared DB
        os.makedirs(os.path.dirname(shared_db_path) or ".", exist_ok=True)
        shared_conn = sqlite3.connect(shared_db_path, timeout=60.0)
        shared_conn.execute("PRAGMA journal_mode=WAL;")
        shared_conn.execute("PRAGMA synchronous=NORMAL;")
        shared_conn.execute("""
            CREATE TABLE IF NOT EXISTS reactions (
                reactants TEXT PRIMARY KEY,
                product TEXT
            )
        """)
        shared_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reactants ON reactions(reactants)"
        )
        shared_conn.commit()

        # Get existing keys in shared DB for deduplication
        logger.info("Loading existing keys from shared DB...")
        existing_keys = set()
        cursor = shared_conn.execute("SELECT reactants FROM reactions")
        for row in cursor:
            existing_keys.add(row[0])
        logger.info(f"Shared DB has {len(existing_keys):,} existing entries")

        # Process each local file
        for local_path in local_files:
            try:
                # Skip if file is too small (likely empty/corrupted)
                if os.path.getsize(local_path) < 1024:
                    logger.debug(f"Skipping small file: {local_path}")
                    stats["local_files_skipped"] += 1
                    continue

                # Check if file is still in use (by checking for -wal file with recent mtime)
                wal_path = local_path + "-wal"
                if os.path.exists(wal_path):
                    wal_mtime = os.path.getmtime(wal_path)
                    if time.time() - wal_mtime < 60:  # Modified in last 60 seconds
                        logger.warning(f"Skipping in-use file: {local_path}")
                        stats["local_files_skipped"] += 1
                        stats["errors"].append(f"In-use: {local_path}")
                        continue

                # Open local DB
                local_conn = sqlite3.connect(local_path, timeout=10.0)

                # Verify table exists
                cursor = local_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='reactions'"
                )
                if cursor.fetchone() is None:
                    logger.warning(f"No reactions table in {local_path}, skipping")
                    local_conn.close()
                    stats["local_files_skipped"] += 1
                    continue

                # Read entries from local DB
                cursor = local_conn.execute("SELECT reactants, product FROM reactions")
                rows = cursor.fetchall()
                local_conn.close()

                if not rows:
                    logger.debug(f"Empty local cache: {local_path}")
                    stats["local_files_skipped"] += 1
                    if delete_after_merge:
                        _safe_delete_db_files(local_path)
                        stats["local_files_deleted"] += 1
                    continue

                # Filter out duplicates
                new_rows = []
                for reactants, product in rows:
                    stats["total_entries_processed"] += 1
                    if reactants not in existing_keys:
                        new_rows.append((reactants, product))
                        existing_keys.add(reactants)  # Track for subsequent files
                    else:
                        stats["duplicate_entries_skipped"] += 1

                # Insert new entries in batches
                if new_rows:
                    for i in range(0, len(new_rows), batch_size):
                        batch = new_rows[i:i+batch_size]
                        shared_conn.executemany(
                            "INSERT OR IGNORE INTO reactions (reactants, product) VALUES (?, ?)",
                            batch
                        )
                    shared_conn.commit()
                    stats["new_entries_added"] += len(new_rows)

                stats["local_files_merged"] += 1
                logger.info(
                    f"Merged {local_path}: {len(new_rows)} new / {len(rows)} total entries"
                )

                # Delete local file if requested
                if delete_after_merge:
                    _safe_delete_db_files(local_path)
                    stats["local_files_deleted"] += 1

            except Exception as e:
                logger.error(f"Error processing {local_path}: {e}")
                stats["errors"].append(f"{local_path}: {str(e)}")
                stats["local_files_skipped"] += 1

        # Final commit and cleanup
        shared_conn.commit()
        shared_conn.execute("PRAGMA optimize;")
        shared_conn.close()

    finally:
        # Release lock
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            os.remove(lock_file_path)
        except:
            pass

    logger.info(
        f"Merge complete: {stats['local_files_merged']} files merged, "
        f"{stats['new_entries_added']:,} new entries, "
        f"{stats['duplicate_entries_skipped']:,} duplicates skipped"
    )

    return stats


def _safe_delete_db_files(db_path: str):
    """Safely delete SQLite database and associated files (-wal, -shm)."""
    for suffix in ["", "-wal", "-shm"]:
        path = db_path + suffix
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Could not delete {path}: {e}")


def verify_shared_cache(shared_db_path: str = "./cache/reaction_cache_shared.db") -> Dict:
    """
    Verify integrity of the shared cache.

    Returns:
        Dict with verification results
    """
    results = {
        "exists": False,
        "readable": False,
        "entry_count": 0,
        "sample_entries": [],
        "errors": [],
    }

    if not os.path.exists(shared_db_path):
        results["errors"].append("File does not exist")
        return results

    results["exists"] = True

    try:
        conn = sqlite3.connect(shared_db_path, timeout=10.0)

        # Check table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reactions'"
        )
        if cursor.fetchone() is None:
            results["errors"].append("No 'reactions' table found")
            conn.close()
            return results

        results["readable"] = True

        # Count entries
        cursor = conn.execute("SELECT COUNT(*) FROM reactions")
        results["entry_count"] = cursor.fetchone()[0]

        # Sample entries
        cursor = conn.execute("SELECT reactants, product FROM reactions LIMIT 5")
        results["sample_entries"] = cursor.fetchall()

        # Check for corruption
        cursor = conn.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]
        if integrity != "ok":
            results["errors"].append(f"Integrity check failed: {integrity}")

        conn.close()

    except Exception as e:
        results["errors"].append(str(e))

    return results


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    # set logger to debug level
    logging.basicConfig(level=logging.DEBUG)
    
    import argparse

    parser = argparse.ArgumentParser(description="Reaction Cache Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge local caches to shared")
    merge_parser.add_argument(
        "--local-dir", default="./cache/local",
        help="Directory containing local cache files"
    )
    merge_parser.add_argument(
        "--shared-db", default="./cache/reaction_cache_shared.db",
        help="Path to shared database"
    )
    merge_parser.add_argument(
        "--delete", action="store_true",
        help="Delete local files after successful merge"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify shared cache integrity")
    verify_parser.add_argument(
        "--shared-db", default="./cache/reaction_cache_shared.db",
        help="Path to shared database"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument(
        "--local-dir", default="./cache/local",
        help="Directory containing local cache files"
    )
    stats_parser.add_argument(
        "--shared-db", default="./cache/reaction_cache_shared.db",
        help="Path to shared database"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "merge":
        stats = merge_local_caches_to_shared(
            local_db_dir=args.local_dir,
            shared_db_path=args.shared_db,
            delete_after_merge=args.delete,
        )
        print(f"\nMerge Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.command == "verify":
        results = verify_shared_cache(args.shared_db)
        print(f"\nVerification Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    elif args.command == "stats":
        import glob

        # Shared DB stats
        print("\n=== Shared Cache ===")
        results = verify_shared_cache(args.shared_db)
        print(f"  Path: {args.shared_db}")
        print(f"  Exists: {results['exists']}")
        print(f"  Entries: {results['entry_count']:,}")
        if results['errors']:
            print(f"  Errors: {results['errors']}")

        # Local DB stats
        print("\n=== Local Caches ===")
        pattern = os.path.join(args.local_dir, "cache_*.db")
        local_files = glob.glob(pattern)
        print(f"  Directory: {args.local_dir}")
        print(f"  Files: {len(local_files)}")

        total_local = 0
        for lf in local_files[:5]:  # Show first 5
            try:
                conn = sqlite3.connect(lf, timeout=5.0)
                cursor = conn.execute("SELECT COUNT(*) FROM reactions")
                count = cursor.fetchone()[0]
                conn.close()
                total_local += count
                print(f"    {os.path.basename(lf)}: {count:,} entries")
            except Exception as e:
                print(f"    {os.path.basename(lf)}: error - {e}")

        if len(local_files) > 5:
            print(f"    ... and {len(local_files) - 5} more files")

        print(f"\n  Total in local caches (sampled): ~{total_local:,} entries")

    else:
        parser.print_help()
