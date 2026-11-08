"""StaleGuard FastAPI app. Step 1: /health only. Proxy (/v1/chat/completions) and /verify come in steps 2-3."""

from fastapi import Depends, FastAPI

from staleguard import __version__
from staleguard.bifrost_client import check_bifrost
from staleguard.config import Settings, get_settings
from staleguard.schemas import HealthResponse

app = FastAPI(title="StaleGuard", version=__version__)


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    bifrost = await check_bifrost(settings)
    return HealthResponse(status="ok" if bifrost.ok else "degraded", version=__version__, bifrost=bifrost)
