import grpc
import time
import logging
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)

# Maximum gRPC message size for activation tensors in pipeline training
MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64MB for large batches/sequences

# Default gRPC channel options for P2P network
_DEFAULT_CHANNEL_OPTIONS = [
    ('grpc.keepalive_time_ms', 30000),  # Ping every 30 seconds
    ('grpc.keepalive_timeout_ms', 10000),  # 10 second timeout
    ('grpc.keepalive_permit_without_calls', True),  # Ping even when idle
    ('grpc.http2.max_pings_without_data', 0),  # Unlimited pings
    ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),
    ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),
]


class ConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConnectionPool, cls).__new__(cls)
                    cls._instance.channels = {}  # url -> channel
                    cls._instance.last_used = {}  # url -> timestamp
                    cls._instance._use_tls = False
                    cls._instance._tls_credentials = None
        return cls._instance

    def configure_tls(self, use_tls: bool = True,
                      root_certificates: Optional[bytes] = None,
                      private_key: Optional[bytes] = None,
                      certificate_chain: Optional[bytes] = None):
        """Configure TLS for all new channels.
        
        Args:
            use_tls: Whether to use TLS (True for production, False for local dev)
            root_certificates: PEM-encoded root CA certificates (None = system defaults)
            private_key: PEM-encoded client private key for mutual TLS (optional)
            certificate_chain: PEM-encoded client certificate for mutual TLS (optional)
        """
        self._use_tls = use_tls
        if use_tls:
            self._tls_credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain,
            )
            logger.info("[TLS] gRPC channels configured with TLS encryption")
        else:
            self._tls_credentials = None
            logger.info("[TLS] gRPC channels configured WITHOUT TLS (insecure)")

    def get_channel(self, address: str):
        """
        Get an existing channel or create a new one.
        address format: "ip:port"
        
        Uses TLS if configure_tls() was called with use_tls=True.
        """
        # Normalize address (remove http:// if present)
        if address.startswith("http://"):
            address = address.replace("http://", "")
        if address.startswith("https://"):
            address = address.replace("https://", "")

        with self._lock:
            if address in self.channels:
                # Check if channel is active (simplified check)
                self.last_used[address] = time.time()
                return self.channels[address]
            
            options = list(_DEFAULT_CHANNEL_OPTIONS)
            
            if self._use_tls and self._tls_credentials is not None:
                channel = grpc.secure_channel(
                    address, self._tls_credentials, options=options
                )
                logger.debug(f"[TLS] Created secure channel to {address}")
            else:
                channel = grpc.insecure_channel(address, options=options)
            
            self.channels[address] = channel
            self.last_used[address] = time.time()
            return channel

    def cleanup(self, max_idle_seconds=300):
        """Close channels idle for too long"""
        now = time.time()
        to_remove = []
        with self._lock:
            for addr, last_time in self.last_used.items():
                if now - last_time > max_idle_seconds:
                    to_remove.append(addr)
            
            for addr in to_remove:
                logger.debug(f"Closing idle connection to {addr}")
                self.channels[addr].close()
                del self.channels[addr]
                del self.last_used[addr]


# Global accessor
def get_channel(address: str):
    pool = ConnectionPool()
    return pool.get_channel(address)
