from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # In this override impl we just need to add extra actions, we need to propagate attachment to the Choice since
        # in DeploymentTool they were propagated to the stage only as files. The main goal here is show pictures in chat
        # (DIAL Chat support special markdown to load pictures from DIAL bucket directly to the chat)
        # ---
        # 1. Call parent function `_execute` and get result
        msg = await super()._execute(tool_call_params)
        # 2. If attachments are present then filter only "image/png" and "image/jpeg"
        if msg.custom_content and msg.custom_content.attachments:
            for attachment in msg.custom_content.attachments:
                if attachment.type in ["image/png", "image/jpeg"]:
        # 3. Append then as content to choice in such format `f"\n\r![image]({attachment.url})\n\r")`
                    tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")
        # 4. After iteration through attachment if message content is absent add such instruction:
        #    'The image has been successfully generated according to request and shown to user!'
        #    Sometimes models are trying to add generated pictures as well to content (choice), with this instruction
        #    we are notifing LLLM that it was done (but anyway sometimes it will try to add file 😅)
        if not msg.content or msg.content.strip() == "":
            msg.content = StrictStr('The image has been successfully generated according to request and shown to user!')
        return msg

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        return (
            "# Image generator\n"
            "Generates image based on the provided description.\n"
            "## Instructions:\n"
            "- Use that tool when user asks to generate an image based on the description or to visualize some text or information.\n"
            "- Choose the best size from available options based on user request or image type. For specific size requests, use the closest supported option.\n"
            "- When the tool returns a markdown image URL, always include it in your response and follow it with a brief description.\n"
            "## Restrictions:\n"
            "- Never use this tool for data or numerical information visualization."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated."
                },
                "size": {
                    "type": "string",
                    "description": "The size of the generated image.",
                    "enum": [
                        "1024x1024",
                        "1024x1792",
                        "1792x1024"
                    ],
                    "default": "1024x1024"
                },
                "style": {
                    "type": "string",
                    "description": "The style of the generated image. Must be one of `vivid` or `natural`. \n- `vivid` causes the model to lean towards generating hyperrealistic and dramatic images. \n- `natural` causes the model to produce more natural, less realistic looking images.",
                    "enum": [
                        "natural",
                        "vivid"
                    ],
                    "default": "natural"
                },
                "quality": {
                    "type": "string",
                    "description": "The quality of the image that will be generated. ‘hd’ creates images with finer details and greater consistency across the image.",
                    "enum": [
                        "standard",
                        "hd"
                    ],
                    "default": "standard"
                }
            },
            "required": [
                "prompt"
            ]
        }

