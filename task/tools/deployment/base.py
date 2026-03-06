import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        args = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `prompt` from arguments (by default we provide `prompt` for each deployment tool, use this param name as standard)
        prompt = args.get("prompt", "")
        # 3. Delete `prompt` from `arguments` (there can be provided additional parameters and `prompt` will be added
        #    as user message content and other parameters as `custom_fields`)
        del args["prompt"]
        # 4. Create AsyncDial client (api_version is 2025-01-01-preview)
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview'
        )
        # 5. Call chat completions with:
        #   - messages (here will be just user message. Optionally, in this class you can add system prompt `property`
        #     and if any deployment tool provides system prompt then we need to set it as first message (system prompt))
        #   - stream it
        #   - deployment_name
        #   - extra_body with `custom_fields` https://dialx.ai/dial_api#operation/sendChatCompletionRequest (last request param in documentation)
        #   - **self.tool_parameters (will load all tool parameters that were set up in deployment tools as params, like
        #     `top_p`, `temperature`, etc...)
        chunks = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            deployment_name=self.deployment_name,
            extra_body={
                "custom_fields": {
                    "configuration": {**args}
                }
            },
            **self.tool_parameters
        )
        # 6. Collect content and it to stage, also, collect custom_content -> attachments and if they are present add
        #    them to stage as attachment as well
        content = ""
        custom_content = CustomContent(attachments=[])
        async for chunk in chunks:
            delta = chunk.choices[0].delta
            if delta:
                if delta.content:
                    tool_call_params.stage.append_content(delta.content)
                    content += delta.content
                if delta.custom_content and delta.custom_content.attachments:
                    attachments = delta.custom_content.attachments
                    custom_content.attachments.extend(attachments)

                    for attachment in attachments:
                        tool_call_params.stage.add_attachment(
                            type=attachment.type,
                            title=attachment.title,
                            data=attachment.data,
                            url=attachment.url,
                            reference_url=attachment.reference_url,
                            reference_type=attachment.reference_type,
                        )
        # 7. Return Message with tool role, content, custom_content and tool_call_id
        return Message(
            role=Role.TOOL,
            content=StrictStr(content),
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id)
        )
