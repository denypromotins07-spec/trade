"""Nautilus/Ray FFI Bridge - Stage 54
PyO3/ctypes bridge for Ray workers to call Rust matching engine.
Enforces 4GB Python RAM quota during FFI marshalling."""

from __future__ import annotations
import ctypes, logging, os, threading, weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nautilus_bridge")

PYTHON_RAM_QUOTA_BYTES = 4 * 1024**3
GLOBAL_RAM_CEILING_BYTES = 8 * 1024**3

class OrderType(IntEnum):
    LIMIT = 0; MARKET = 1; STOP_LOSS = 2; TAKE_PROFIT = 3

class OrderSide(IntEnum):
    BUY = 0; SELL = 1

@dataclass
class OrderFFI:
    order_id: int; symbol: bytes; side: int; order_type: int
    price: float; quantity: float; timestamp_ns: int; client_order_id: bytes = b""
    
    def to_bytes(self) -> bytes:
        return (ctypes.c_uint64(self.order_id).value.to_bytes(8, 'little') +
                self.symbol[:32].ljust(32, b'\x00') +
                ctypes.c_uint8(self.side).value.to_bytes(1, 'little') +
                ctypes.c_uint8(self.order_type).value.to_bytes(1, 'little') +
                ctypes.c_double(self.price).value +
                ctypes.c_double(self.quantity).value +
                ctypes.c_uint64(self.timestamp_ns).value.to_bytes(8, 'little'))

class MemoryTracker:
    """Enforces 4GB Python RAM quota."""
    _instance: Optional["MemoryTracker"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "MemoryTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'): return
        self._quota = PYTHON_RAM_QUOTA_BYTES
        self._ceiling = GLOBAL_RAM_CEILING_BYTES
        self._callbacks: List = []
        self._initialized = True
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None
    
    def get_usage(self) -> int:
        if self._process: return self._process.memory_info().rss
        return 0
    
    def check_quota(self) -> bool:
        usage = self.get_usage()
        ratio = usage / self._quota
        for cb in self._callbacks: cb(ratio)
        if usage > self._ceiling:
            logger.critical(f"GLOBAL RAM BREACH: {usage/1e9:.2f}GB / 8GB")
            return False
        if usage > self._quota:
            logger.error(f"PYTHON QUOTA EXCEEDED: {usage/1e9:.2f}GB / 4GB")
            return False
        if ratio > 0.9: logger.warning(f"RAM at {ratio*100:.1f}% of quota")
        return True
    
    def register_callback(self, cb) -> None:
        self._callbacks.append(cb)

memory_tracker = MemoryTracker()

class NautilusRayBridge:
    """Main bridge for Ray workers to interact with Rust engine."""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or self._find_lib()
        self._lib: Optional[ctypes.CDLL] = None
        self._handle: Optional[int] = None
        self._active = False
        self._lock = threading.RLock()
        memory_tracker.register_callback(self._on_pressure)
    
    def _find_lib(self) -> str:
        candidates = ["./target/release/nautilus_hft.dll", "./target/release/libnautilus_hft.so"]
        for c in candidates:
            if os.path.exists(c): return c
        raise FileNotFoundError("Rust FFI library not found. Run: cargo build --release")
    
    def _on_pressure(self, ratio: float) -> None:
        if ratio > 0.95:
            logger.critical("Memory pressure > 95%, triggering GC")
            import gc; gc.collect()
    
    def connect(self) -> bool:
        with self._lock:
            if self._active: return True
            try:
                self._lib = ctypes.CDLL(self.lib_path)
                self._setup_ffi()
                self._handle = self._lib.nautilus_engine_create()
                self._active = self._handle != 0
                logger.info(f"Connected (handle={self._handle})")
                return self._active
            except Exception as e:
                logger.error(f"Connect failed: {e}")
                return False
    
    def _setup_ffi(self) -> None:
        lib = self._lib
        lib.nautilus_engine_create.argtypes = []; lib.nautilus_engine_create.restype = ctypes.c_void_p
        lib.nautilus_engine_destroy.argtypes = [ctypes.c_void_p]; lib.nautilus_engine_destroy.restype = None
        lib.nautilus_submit_order.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.nautilus_submit_order.restype = ctypes.c_int64
        lib.nautilus_cancel_order.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.nautilus_cancel_order.restype = ctypes.c_bool
        lib.nautilus_get_position.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nautilus_get_position.restype = ctypes.c_double
        lib.nautilus_get_pnl.argtypes = [ctypes.c_void_p]; lib.nautilus_get_pnl.restype = ctypes.c_double
    
    def disconnect(self) -> None:
        with self._lock:
            if self._active and self._lib:
                self._lib.nautilus_engine_destroy(ctypes.c_void_p(self._handle))
                self._active = False; self._handle = None
                logger.info("Disconnected")
    
    def submit_order(self, order: OrderFFI, timeout: float = 5.0) -> Optional[int]:
        if not self._active: raise RuntimeError("Not connected")
        if not memory_tracker.check_quota(): raise MemoryError("Python RAM quota exceeded")
        with self._lock:
            result = self._lib.nautilus_submit_order(
                ctypes.c_void_p(self._handle), order.to_bytes(), len(order.to_bytes()))
            return result if result >= 0 else None
    
    def cancel_order(self, order_id: int) -> bool:
        if not self._active: raise RuntimeError("Not connected")
        with self._lock:
            return bool(self._lib.nautilus_cancel_order(ctypes.c_void_p(self._handle), ctypes.c_uint64(order_id)))
    
    def get_position(self, symbol: str) -> float:
        if not self._active: raise RuntimeError("Not connected")
        with self._lock:
            return float(self._lib.nautilus_get_position(ctypes.c_void_p(self._handle), symbol.encode()))
    
    def get_pnl(self) -> float:
        if not self._active: raise RuntimeError("Not connected")
        with self._lock:
            return float(self._lib.nautilus_get_pnl(ctypes.c_void_p(self._handle)))
    
    def __enter__(self): self.connect(); return self
    def __exit__(self, *args): self.disconnect()
    def __del__(self): self.disconnect()

try:
    import ray
    @ray.remote
    class NautilusRayActor:
        def __init__(self, worker_id: int):
            self.worker_id = worker_id; self.bridge = NautilusRayBridge(); self.connected = False
        def initialize(self) -> bool:
            self.connected = self.bridge.connect(); return self.connected
        def submit_order_remote(self, data: dict) -> Optional[int]:
            if not self.connected: return None
            order = OrderFFI(data["order_id"], data["symbol"].encode(), data["side"],
                           data["order_type"], data["price"], data["quantity"], data["timestamp_ns"])
            return self.bridge.submit_order(order)
        def cancel_order_remote(self, order_id: int) -> bool:
            return self.bridge.cancel_order(order_id)
        def get_position_remote(self, symbol: str) -> float:
            return self.bridge.get_position(symbol)
        def get_pnl_remote(self) -> float:
            return self.bridge.get_pnl()
        def shutdown(self): self.bridge.disconnect()
except ImportError: pass

def create_bridge() -> NautilusRayBridge: return NautilusRayBridge()

if __name__ == "__main__":
    print("Testing Nautilus FFI Bridge...")
    bridge = create_bridge()
    if bridge.connect():
        print(f"Connected, PnL: ${bridge.get_pnl():.2f}")
        bridge.disconnect()
    else:
        print("Failed to connect (expected if library not built)")
