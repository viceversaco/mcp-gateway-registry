"""Authentication provider package for MCP Gateway Registry."""

from .authelia import AutheliaProvider
from .base import AuthProvider
from .cognito import CognitoProvider
from .factory import get_auth_provider
from .keycloak import KeycloakProvider

__all__ = [
    "AuthProvider",
    "AutheliaProvider",
    "CognitoProvider",
    "KeycloakProvider",
    "get_auth_provider"
]