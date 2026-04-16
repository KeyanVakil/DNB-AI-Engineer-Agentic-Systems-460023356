"""Typed errors for MCP tool failures."""

from __future__ import annotations


class ToolError(Exception):
    """Raised (and captured into graph state) when an MCP tool call fails."""

    def __init__(self, *, server: str, tool: str, cause: Exception) -> None:
        self.server = server
        self.tool = tool
        self.cause = cause
        super().__init__(f"MCP tool {server}.{tool} failed: {cause}")
