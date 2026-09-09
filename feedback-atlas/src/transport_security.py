"""Transport and browser boundaries shared by HTTP and WebSocket routes."""

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse


class TransportSecurity:
    def __init__(self, app, *, require_https: bool):
        self.app = app
        self.require_https = require_https

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)

        headers = Headers(scope=scope)
        secure = scope.get("scheme") in {"https", "wss"}
        origin = headers.get("origin")
        expected = f"{'https' if secure else 'http'}://{headers.get('host', '')}"
        reason = None
        if origin is not None and origin != expected:
            reason = "Request origin is not allowed."
        # Health checks contain no credentials and remain usable before TLS starts.
        if (self.require_https and not secure
                and not (scope["type"] == "http" and scope["path"] == "/healthz")):
            reason = "HTTPS is required. Please use the secure classroom address."
        if reason:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await PlainTextResponse(reason, status_code=403)(scope, receive, send)
            return

        async def protected_send(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (b"content-security-policy",
                     b"default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
                     b"style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                     b"font-src 'self' data:; worker-src 'self' blob:; "
                     b"connect-src 'self'; object-src 'none'; base-uri 'none'; "
                     b"frame-ancestors 'none'"),
                ]
            await send(message)

        await self.app(scope, receive, protected_send)
