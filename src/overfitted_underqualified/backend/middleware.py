"""Bound request-body size before Starlette parses multipart uploads."""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _UploadTooLarge(Exception):
    pass


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp, max_upload_bytes: int) -> None:
        self.app = app
        # Allow up to 1 MiB for multipart headers and boundaries around the video part.
        self.max_request_bytes = max_upload_bytes + 1024 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/api/v1/analyze":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_request_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass

        bytes_received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > self.max_request_bytes:
                    raise _UploadTooLarge
            return message

        async def track_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, track_send)
        except _UploadTooLarge:
            if not response_started:
                await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"detail":"Upload exceeds the configured size limit"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
