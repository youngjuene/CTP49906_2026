"""HTTP routes for class-week schedule metadata."""

from __future__ import annotations

from datetime import datetime, timezone

from src.class_schedule import ScheduleConflict, schedule_info, update_schedule


async def broadcast_schedule(atlas, now=None) -> dict:
    when = now or datetime.now(timezone.utc)
    info = schedule_info(atlas, when)
    message = {"t": "schedule", "schedule": info}
    participant, admin = _channels()
    await atlas.hub.broadcast(participant, message)
    await atlas.hub.broadcast(admin, message)
    atlas.schedule_marker = (info["revision"], info["current_week"])
    return info


def register_schedule_routes(app, atlas, request_json, valid_admin_code, clock) -> None:
    from fastapi import HTTPException, Request
    @app.get("/api/schedule")
    async def get_schedule():
        return schedule_info(atlas, clock())

    async def put_schedule(request):
        from fastapi.responses import JSONResponse

        from src.protocol import error

        try:
            body = await request_json(request)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            return JSONResponse(error("MALFORMED"), status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(error("MALFORMED"), status_code=400)

        supplied = body.get("code")
        if not valid_admin_code(supplied, atlas.cfg.admin_code):
            key = request.client.host if request.client else "unknown"
            if not atlas.admin_limiter.allow(key):
                return JSONResponse(error("RATE_LIMITED"), status_code=429)
            return JSONResponse(error("BAD_ACCESS_CODE"), status_code=403)

        if "starts" not in body or "expected_revision" not in body:
            return JSONResponse(error("MALFORMED"), status_code=400)
        try:
            atlas.schedule = update_schedule(
                atlas.store, body["starts"], body["expected_revision"])
        except ScheduleConflict:
            return JSONResponse(error("STALE_SCHEDULE"), status_code=409)
        except (TypeError, ValueError):
            return JSONResponse(error("BAD_SCHEDULE"), status_code=400)

        return await broadcast_schedule(atlas, clock())

    put_schedule.__annotations__["request"] = Request
    app.add_api_route("/api/schedule", put_schedule, methods=["PUT"])


def _channels():
    from src.hub import Channel

    return Channel.PARTICIPANT, Channel.ADMIN
