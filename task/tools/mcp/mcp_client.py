from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent, ReadResourceResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        # 1. Create instance of MCPClient with `cls`
        instance = cls(mcp_server_url)
        # 2. Connect to MCP server
        await instance.connect()
        # 3. return created instance
        return instance

    async def connect(self):
        """Connect to MCP server"""
        # 1. Check if session is present, if yes just return to finsh execution
        if self.session is not None:
            return
        # 2. Call `streamablehttp_client` method with `server_url` and set as `self._streams_context`
        self._streams_context = streamablehttp_client(self.server_url)
        # 3. Enter `self._streams_context`, result set as `read_stream, write_stream, _`
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        # 4. Create ClientSession with streams from above and set as `self._session_context`
        self._session_context = ClientSession(
            read_stream=read_stream,
            write_stream=write_stream
        )
        # 5. Enter `self._session_context` and set as self.session
        self.session = await self._session_context.__aenter__()
        # 6. Initialize session and print its result to console
        init_result = await self.session.initialize()
        try:
            await self.session.send_ping()
        except Exception as e:
            # Clean up on connection failure
            await self.close()
            raise ValueError(f"MCP server connection failed: {e}")


    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        if not self.session:
            raise ValueError("MCPClient is not connected. Call 'connect' method first.")
        tools = await self.session.list_tools()
        return [
            MCPToolModel(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema
            ) for tool in tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        if not self.session:
            raise ValueError("MCPClient is not connected. Call 'connect' method first.")
        tool_result: CallToolResult = await self.session.call_tool(tool_name, tool_args)

        if not tool_result.content:
            return None

        content = tool_result.content[0]
        if isinstance(content, TextContent):
            return content.text
        return content

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        if not self.session:
            raise RuntimeError("MCP client not connected.")

        resource_result: ReadResourceResult = await self.session.read_resource(uri)

        if not resource_result.contents:
            raise ValueError(f"No content in resource: {uri}")

        content = resource_result.contents[0]

        if isinstance(content, TextResourceContents):
            return content.text
        elif isinstance(content, BlobResourceContents):
            return content.blob
        else:
            raise ValueError(f"Unexpected content type: {type(content)}")

    async def close(self):
        """Close connection to MCP server"""
        # 1. Close `self._session_context`
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error closing session context: {e}")
        # 2. Close `self._streams_context`
        try:
            if self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error closing streams context: {e}")
        # 3. Set session, _session_context and _streams_context as None
        finally:
            # Clean up references
            self.session = None
            self._session_context = None
            self._streams_context = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False

