import httpx


class InferenceClient:
    def __init__(self, base_url: str):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def detect(self, jpeg: bytes) -> list[dict]:
        response = await self._client.post(
            "/detect/binary",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
