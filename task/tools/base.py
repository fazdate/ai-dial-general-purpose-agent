from abc import ABC, abstractmethod
from typing import Any

from aidial_client.types.chat import ToolParam, FunctionParam
from aidial_client.types.chat.legacy.chat_completion import Role
from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.models import ToolCallParams


class BaseTool(ABC):

    async def execute(self, tool_call_params: ToolCallParams) -> Message:
        # 1. Create Message obj with:
        #       - role=Role.TOOL
        #       - name=StrictStr(tool_call_params.tool_call.function.name)
        #       - tool_call_id=StrictStr(tool_call_params.tool_call.id)
        message = Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id)
        )
        # 2. We will use Template method pattern here, so:
        #       - Open `try-except` block
        #       - In `try` block call`_execute` method, then check if result isinstance of Message, if yes then
        #         assign result to created message in 1st step, otherwise set Message `content` as StrictStr(result)
        #       - In `except` block intercept Exception and add it properly to Message `content`
        try:
            result = await self._execute(tool_call_params)
            if isinstance(result, Message):
                message = result
            else:
                message.content = StrictStr(result)
        except Exception as e:
            message.content = StrictStr(f"⚠️ Tool execution error: {e}")
        # 3. Return created message
        return message

    @abstractmethod
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        pass

    @property
    def show_in_stage(self) -> bool:
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def schema(self) -> ToolParam:
        """Provides tool schema according to DIAL specification."""
        #TODO:
        # see https://dialx.ai/dial_api#operation/sendChatCompletionRequest -> `tools`
        # or https://platform.openai.com/docs/guides/function-calling#defining-functions
        return ToolParam(
            type="function",
            function=FunctionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters
            )
        )
