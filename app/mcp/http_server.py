"""HTTP-backed MCP server proxy for production use."""

from __future__ import annotations

from typing import Any

import httpx


class HttpMCPServer:
    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")

    async def list_tools(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self._base_url}/tools")
            resp.raise_for_status()
            return resp.json()["tools"]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/tools/{tool_name}",
                json=arguments,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
