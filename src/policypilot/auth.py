"""JWT auth with a groups claim. Dev: HS256 tokens from /v1/dev/token. Production: verify
RS256 tokens from your IdP (OIDC) with the same claims; see README."""

from __future__ import annotations

import time
from typing import Annotated, Any, cast

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.errors import DomainError
from policypilot.config import PolicySettings
from policypilot.services.answer import User

bearer = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


def issue_token(
    settings: PolicySettings, sub: str, groups: list[str], roles: list[str] | None = None
) -> str:
    now = int(time.time())
    claims = {
        "sub": sub,
        "groups": sorted(set(groups)),
        "roles": roles or [],
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + settings.jwt_ttl_s,
    }
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM)


def verify_token(settings: PolicySettings, token: str) -> User:
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[ALGORITHM],
            audience=settings.jwt_audience,
            options={"require": ["exp", "sub", "aud"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise DomainError(401, "token-expired", "Token expired") from e
    except jwt.InvalidTokenError as e:
        raise DomainError(401, "invalid-token", "Invalid token", type(e).__name__) from e
    groups = claims.get("groups")
    if not isinstance(groups, list) or not groups or not all(isinstance(g, str) for g in groups):
        raise DomainError(401, "invalid-token", "Invalid token", "groups claim missing")
    roles = claims.get("roles") or []
    return User(sub=str(claims["sub"]), groups=list(groups), roles=[str(r) for r in roles])


def policy_settings(request: Request) -> PolicySettings:
    return cast(PolicySettings, request.app.state.settings)


async def current_user(
    request: Request, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> User:
    token = creds.credentials if creds else request.cookies.get("pp_token")
    if not token:
        raise DomainError(401, "unauthenticated", "Missing bearer token")
    return verify_token(policy_settings(request), token)


async def admin_user(user: Annotated[User, Depends(current_user)]) -> User:
    if not user.is_admin:
        raise DomainError(403, "forbidden", "Admin role required")
    return user


UserDep = Annotated[User, Depends(current_user)]
AdminDep = Annotated[User, Depends(admin_user)]
