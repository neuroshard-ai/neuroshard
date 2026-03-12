"""
NeuroShard SDK Error Classes

Standard exception classes for SDK error handling.
"""

from typing import Optional


class NeuroShardError(Exception):
    """Base exception for all NeuroShard SDK errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def __str__(self):
        return f"[{self.code}] {self.message}"


class AuthenticationError(NeuroShardError):
    """Raised when API token is invalid or missing."""
    
    def __init__(self, message: str = "Invalid or missing API token"):
        super().__init__(message, code="UNAUTHORIZED")


class InsufficientBalanceError(NeuroShardError):
    """Raised when wallet balance is insufficient for an operation."""
    
    def __init__(self, required: float, available: float, message: Optional[str] = None):
        self.required = required
        self.available = available
        msg = message or f"Insufficient balance: need {required} NEURO, have {available}"
        super().__init__(msg, code="INSUFFICIENT_BALANCE", details={
            "required": required,
            "available": available
        })


class RateLimitError(NeuroShardError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int, message: Optional[str] = None):
        self.retry_after = retry_after
        msg = message or f"Rate limited, retry after {retry_after} seconds"
        super().__init__(msg, code="RATE_LIMITED", details={
            "retry_after": retry_after
        })


class NodeOfflineError(NeuroShardError):
    """Raised when the target node is offline or unreachable."""
    
    def __init__(self, url: str, message: Optional[str] = None):
        self.url = url
        msg = message or f"Node is offline or unreachable: {url}"
        super().__init__(msg, code="NODE_OFFLINE", details={
            "url": url
        })


class InvalidRequestError(NeuroShardError):
    """Raised when the request is malformed or invalid."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        details = {"field": field} if field else {}
        super().__init__(message, code="INVALID_REQUEST", details=details)


class NotFoundError(NeuroShardError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource: str, identifier: Optional[str] = None):
        self.resource = resource
        self.identifier = identifier
        msg = f"{resource} not found"
        if identifier:
            msg = f"{resource} '{identifier}' not found"
        super().__init__(msg, code="NOT_FOUND", details={
            "resource": resource,
            "identifier": identifier
        })


class ForbiddenError(NeuroShardError):
    """Raised when the operation is forbidden."""
    
    def __init__(self, message: str = "Operation forbidden"):
        super().__init__(message, code="FORBIDDEN")


class InternalError(NeuroShardError):
    """Raised when an internal server error occurs."""
    
    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, code="INTERNAL_ERROR")

