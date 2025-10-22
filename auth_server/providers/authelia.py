"""Authelia authentication provider implementation."""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import jwt
import requests

from .base import AuthProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class AutheliaProvider(AuthProvider):
    """Authelia authentication provider implementation.

    Authelia is an open-source authentication and authorization server
    providing 2-factor authentication and single sign-on (SSO) via OIDC.

    Key architectural notes:
    - Groups are fetched from UserInfo endpoint (OIDC best practice)
    - JWT validation follows standard OIDC flow
    - Uses OpenID Connect Discovery for endpoint configuration
    """

    def __init__(
        self,
        authelia_url: str,
        client_id: str,
        client_secret: str
    ):
        """Initialize Authelia provider.

        Args:
            authelia_url: Base URL of Authelia instance (e.g., https://auth.example.com)
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
        """
        self.authelia_url = authelia_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret

        # Cache for JWKS and configuration
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: int = 3600  # 1 hour

        # Discover OIDC endpoints from well-known configuration
        self._discover_endpoints()

        logger.debug(f"Initialized Authelia provider at {authelia_url}")


    def _discover_endpoints(self):
        """Fetch OIDC discovery document to get endpoint URLs."""
        discovery_url = f"{self.authelia_url}/.well-known/openid-configuration"
        try:
            logger.debug(f"Discovering OIDC endpoints from {discovery_url}")
            response = requests.get(discovery_url, timeout=10)
            response.raise_for_status()
            config = response.json()

            self.issuer = config['issuer']
            self.jwks_url = config['jwks_uri']
            self.token_url = config['token_endpoint']
            self.userinfo_url = config['userinfo_endpoint']
            self.auth_url = config['authorization_endpoint']
            self.end_session_url = config.get('end_session_endpoint')

            logger.info(f"Discovered Authelia endpoints - issuer: {self.issuer}")
        except Exception as e:
            logger.error(f"Failed to discover Authelia endpoints: {e}")
            raise ValueError(f"Cannot discover Authelia OIDC endpoints: {e}")


    def validate_token(
        self,
        token: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Validate Authelia JWT token and fetch user info including groups.

        Note: Authelia stores groups in the UserInfo endpoint, not in JWT claims.
        This follows OIDC best practices for separating identity from authorization data.
        """
        try:
            logger.debug("Validating Authelia JWT token")

            # Get JWKS for validation
            jwks = self.get_jwks()

            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get('kid')

            if not kid:
                raise ValueError("Token missing 'kid' in header")

            # Find matching key
            signing_key = None
            for key in jwks.get('keys', []):
                if key.get('kid') == kid:
                    from jwt import PyJWK
                    signing_key = PyJWK(key).key
                    break

            if not signing_key:
                raise ValueError(f"No matching key found for kid: {kid}")

            # Validate and decode token
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=['RS256'],
                issuer=self.issuer,
                audience=self.client_id,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True
                }
            )

            logger.debug(f"Token validation successful for user: {claims.get('sub', 'unknown')}")

            # Fetch UserInfo to get groups and additional user data
            # This is the OIDC standard location for groups in Authelia
            try:
                user_info = self.get_user_info(token)
                groups = user_info.get('groups', [])
                email = user_info.get('email', claims.get('email'))
                username = user_info.get('preferred_username', claims.get('sub'))
            except Exception as e:
                logger.warning(f"Failed to fetch user info, using claims only: {e}")
                groups = []
                email = claims.get('email')
                username = claims.get('sub')

            # Extract user info from claims and userinfo
            return {
                'valid': True,
                'username': username,
                'email': email,
                'groups': groups,
                'scopes': claims.get('scope', '').split() if claims.get('scope') else [],
                'client_id': claims.get('client_id', self.client_id),
                'method': 'authelia',
                'data': claims
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Authelia token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")


    def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set from Authelia with caching."""
        current_time = time.time()

        # Check if cache is still valid
        if (self._jwks_cache and
            (current_time - self._jwks_cache_time) < self._jwks_cache_ttl):
            logger.debug("Using cached JWKS")
            return self._jwks_cache

        try:
            logger.debug(f"Fetching JWKS from {self.jwks_url}")
            response = requests.get(self.jwks_url, timeout=10)
            response.raise_for_status()

            self._jwks_cache = response.json()
            self._jwks_cache_time = current_time

            logger.debug("JWKS fetched and cached successfully")
            return self._jwks_cache

        except Exception as e:
            logger.error(f"Failed to retrieve JWKS from Authelia: {e}")
            raise ValueError(f"Cannot retrieve JWKS: {e}")


    def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            logger.debug("Exchanging authorization code for token")

            data = {
                'grant_type': 'authorization_code',
                'code': code,
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'redirect_uri': redirect_uri
            }

            response = requests.post(self.token_url, data=data, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token exchange successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to exchange code for token: {e}")
            raise ValueError(f"Token exchange failed: {e}")


    def get_user_info(
        self,
        access_token: str
    ) -> Dict[str, Any]:
        """Get user information from Authelia.

        This is where Authelia stores groups information, following OIDC best practices.
        """
        try:
            logger.debug("Fetching user info from Authelia")

            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get(self.userinfo_url, headers=headers, timeout=10)
            response.raise_for_status()

            user_info = response.json()
            logger.debug(f"User info retrieved for: {user_info.get('preferred_username', 'unknown')}")

            return user_info

        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise ValueError(f"User info retrieval failed: {e}")


    def get_auth_url(
        self,
        redirect_uri: str,
        state: str,
        scope: Optional[str] = None
    ) -> str:
        """Get Authelia authorization URL."""
        logger.debug(f"Generating auth URL with redirect_uri: {redirect_uri}")

        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'scope': scope or 'openid email profile groups',
            'redirect_uri': redirect_uri,
            'state': state
        }

        auth_url = f"{self.auth_url}?{urlencode(params)}"
        logger.debug(f"Generated auth URL: {auth_url}")

        return auth_url


    def get_logout_url(
        self,
        redirect_uri: str
    ) -> str:
        """Get Authelia logout URL."""
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        # Authelia supports OIDC RP-Initiated Logout
        if self.end_session_url:
            params = {
                'client_id': self.client_id,
                'post_logout_redirect_uri': redirect_uri
            }
            logout_url = f"{self.end_session_url}?{urlencode(params)}"
        else:
            # Fallback to basic logout endpoint
            logout_url = f"{self.authelia_url}/logout"

        logger.debug(f"Generated logout URL: {logout_url}")

        return logout_url


    def refresh_token(
        self,
        refresh_token: str
    ) -> Dict[str, Any]:
        """Refresh an access token using a refresh token."""
        try:
            logger.debug("Refreshing access token")

            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }

            response = requests.post(self.token_url, data=data, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token refresh successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to refresh token: {e}")
            raise ValueError(f"Token refresh failed: {e}")


    def validate_m2m_token(
        self,
        token: str
    ) -> Dict[str, Any]:
        """Validate a machine-to-machine token."""
        # M2M tokens use the same validation as regular tokens
        return self.validate_token(token)


    def get_m2m_token(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get machine-to-machine token using client credentials."""
        try:
            logger.debug("Requesting M2M token using client credentials")

            data = {
                'grant_type': 'client_credentials',
                'client_id': client_id or self.client_id,
                'client_secret': client_secret or self.client_secret,
                'scope': scope or 'openid'
            }

            response = requests.post(self.token_url, data=data, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("M2M token generation successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to get M2M token: {e}")
            raise ValueError(f"M2M token generation failed: {e}")


    @lru_cache(maxsize=1)
    def _get_openid_configuration(self) -> Dict[str, Any]:
        """Get OpenID Connect configuration from Authelia."""
        try:
            discovery_url = f"{self.authelia_url}/.well-known/openid-configuration"
            logger.debug(f"Fetching OpenID configuration from {discovery_url}")
            response = requests.get(discovery_url, timeout=10)
            response.raise_for_status()

            config = response.json()
            logger.debug("OpenID configuration retrieved successfully")

            return config

        except requests.RequestException as e:
            logger.error(f"Failed to get OpenID configuration: {e}")
            raise ValueError(f"OpenID configuration retrieval failed: {e}")


    def _check_authelia_health(self) -> bool:
        """Check if Authelia is healthy and accessible."""
        try:
            # Try to fetch OIDC discovery document as a health check
            health_url = f"{self.authelia_url}/.well-known/openid-configuration"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False


    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider-specific information."""
        return {
            'provider_type': 'authelia',
            'authelia_url': self.authelia_url,
            'client_id': self.client_id,
            'endpoints': {
                'auth': self.auth_url,
                'token': self.token_url,
                'userinfo': self.userinfo_url,
                'jwks': self.jwks_url,
                'logout': self.end_session_url or f"{self.authelia_url}/logout",
                'discovery': f"{self.authelia_url}/.well-known/openid-configuration"
            },
            'issuer': self.issuer,
            'healthy': self._check_authelia_health()
        }
