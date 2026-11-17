"""
================================================================================
MCP 客户端调度器 (连接重构后的 api_new_mcp 脑病预测 MCP 服务)
================================================================================
基于官方 `mcp.client.streamable_http` 协议，供总控 DeepSeek 通过标准 MCP 协议
远程调用 `predict_brain_ad_from_nifti` 与 `predict_brain_ad_from_npy` 工具。
"""
import json
from typing import Any, Callable, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.config import settings


class BrainMCPClient:
    """脑病预测 FastMCP 服务异步客户端"""

    def __init__(self, mcp_url: str = settings.BRAIN_AD_MCP_URL):
        self.mcp_url = mcp_url

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """动态拉取远程 MCP Server 暴露的所有工具 Schema"""
        async with streamablehttp_client(self.mcp_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema,
                    }
                    for t in tools_resp.tools
                ]

    async def call_brain_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        on_progress: Optional[Callable[[float, float, Optional[str]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        通过标准 MCP 协议调用脑病预测工具（仅传递 asset_id 凭据与切片参数）。
        """
        async with streamablehttp_client(self.mcp_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                )
                # 解析 MCP TextContent 中的 JSON 负载
                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        try:
                            return json.loads(item.text)
                        except json.JSONDecodeError:
                            return {"raw_text": item.text}
                return {"success": False, "error": "MCP 工具返回为空"}


brain_mcp_client = BrainMCPClient()
