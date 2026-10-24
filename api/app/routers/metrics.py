from fastapi import APIRouter, Request

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics(request: Request):
    return {
        "workers": request.app.state.registry.snapshot(),
        "events_dropped": request.app.state.writer.dropped,
    }
