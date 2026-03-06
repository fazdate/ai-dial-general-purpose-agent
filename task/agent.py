import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        # 1. Set variables: endpoint, system_prompt, tools
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        # 2. Prepare tools_dict where key will be tool name and vale tool itself. It will help us to find tool faster
        #    on the tool call step
        self._tools_dict = {tool.name: tool for tool in tools}
        # 3. Create dict with `state` name. Inside this dict we need to add `TOOL_CALL_HISTORY_KEY` with empty array.
        #    Here, in state, we will 'hide' tool call history. We need it since we need to preserve full conversation history.
        self.state = {
            TOOL_CALL_HISTORY_KEY: []
        }
        raise NotImplementedError()

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request, response: Response) -> Message:
        # 1. Create AsyncDial, don't forget to provide endpoint as base_url and api_key. Api_key you can take from `request` as well as api_version
        #    JFI: while request you will get Per-request API key (not `dial_api_key` configured in Core config). Read
        #    more about it -> https://docs.dialx.ai/platform/core/per-request-keys
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )
        # 2. Create `chunks` with created AsyncDial client (chat -> completions -> create). Provide it with:
        #    - messages: get messages from `request` and unpack them with `_prepare_messages` method
        #    - tools: provide list with tool schemas
        #    - deployment_name
        #    - make it stream
        chunks = await client.chat.completions.create(
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self.tools],
            deployment_name=deployment_name,
            stream=True,
        )
        # 3. Create:
        #   - `tool_call_index_map` (it is empty dict), here we will collect tool calls by their indexes.
        #      Take a look how tool call streaming output is looks like, it is important! -> https://platform.openai.com/docs/guides/function-calling#streaming
        #   - `content`, here we will collect the content from streaming
        tool_call_index_map: dict[int, ToolCall] = {}
        content = ""
        custom_content = CustomContent(attachments=[])
        # 4. Make async loop through `chunks` and then we need to collect content, tool calls and attachments:
        #   - If chunk has `choices` then:
        #       - Get 1st choice `delta`
        #       - if delta is present:
        #           - if delta content is present then append this content to `choice` (it will be shown in DIAL Chat
        #             choice), concat delta content to `content` variable
        #           - if delta has tool_calls then:
        #               - iterate through tool_calls:
        #                   - if tool call has `id` (first chunk of tool call) then add to `tool_call_index_map` new
        #                     tool_call_delta, key will be index and value tool call delta itself
        #                   - otherwise: get by tool call delta `index` from the `tool_call_index_map` the tool call and
        #                     then check if provided tool_call_delta contains `function`, if yes then you need to get from
        #                     `function` `arguments` (if not present set them as empty string to not attach haphazardly None)
        #                     as `argument_chunk` and add it to the extracted from map tool_call function arguments
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

                if delta and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.id:
                            tool_call_index_map[tool_call_delta.index] = tool_call_delta
                        else:
                            existing_tool_call = tool_call_index_map[tool_call_delta.index]
                            if tool_call_delta.function:
                                argument_chunk = tool_call_delta.function.arguments or ""
                                existing_tool_call.function.arguments = (existing_tool_call.function.arguments or "") + argument_chunk
        # 5. Create `assistant_message`, with role, content and tool_calls. `tool_calls` should be a list with ToolCall
        #    objects generated from `tool_call_index_map` dict values. to create ToolCall use `validate` method (it
        #    will show you the notification that it is deprecated but we need to use it because DIAL SDK is built on top of pydentic.v1)
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content,
            custom_content=custom_content,
            tool_calls=[ToolCall.validate(tc) for tc in tool_call_index_map.values()]
        )
        # 6. Now we at the point where we need to understand if its 'final result' from orchestration model or not:
        #    check if `assistant_message` contains `tool_calls`, if yes then we need:
        #       - create `tasks` list. Iterate through `tool_calls` and call `_process_tool_call` method (do not use
        #         `await` since we will run tool calls execution asynchronously), also you need to provide `conversation_id`
        #         you can get it in `request` headers, its name is `x-conversation-id`
        #       - now `gather` tasks with `asyncio` (here you need to await)
        #       - to the `state` to `TOOL_CALL_HISTORY_KEY` append `assistant_message` as dict and exclude none from this dict
        #       - extend the `state` `TOOL_CALL_HISTORY_KEY` with tool_messages that we executed above
        #       - finally make recursive call
        if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0:
            tasks = [
                self._process_tool_call(
                    tool_call=tool_call,
                    choice=choice,
                    api_key=request.api_key,
                    conversation_id=request.headers.get("x-conversation-id", ""),
                )
                for tool_call in assistant_message.tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)
            self.state[TOOL_CALL_HISTORY_KEY].append(assistant_message.dict(exclude_none=True))
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)
            return await self.handle_request(deployment_name, choice, request, response)
        # 7. We don't have any tool calls and reasy to finish user request. Set choice with `state` and return `assistant_message`
        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        # 1. Unpack messages with `unpack_messages` method (it is implemented, just check the logic in this method)
        unpacked_messages = unpack_messages(messages, self.state.get(TOOL_CALL_HISTORY_KEY, []))
        # 2. Insert as first message the `system_prompt` (probably you have a question why do we need to insert each
        #    call system prompt, the reason is simple - security, if people will know our system prompt then it will be
        #    easier to manipulate LLM, so, best practices are to hide system prompt)
        unpacked_messages.insert(0, {
            "role": Role.SYSTEM.value,
            "content": self.system_prompt,
        })
        # 3. Print history: iterate through unpacked messages and print as json (json.dumps)
        for msg in unpacked_messages:
            print(json.dumps(msg, indent=2))
        print(f"{'-' * 100}\n")
        # 4. Return unpacked messages
        return unpacked_messages

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[str, Any]:
        # 1. Get tool name from tool_call function name
        tool_name = tool_call.function.name
        # 2. Open Stage with StageProcessor (it will be shown in DIAL Chat and Stage serves in our case for
        #    tool call results representation)
        stage = StageProcessor.open_stage(choice, tool_name)
        # 3. Get tool from `_tools_dict` by tool name
        tool = self._tools_dict.get(tool_name)
        # 4. If tool show_in_stage is true then:
        #   - append content to stage "## Request arguments: \n"
        #   - append content to stage f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
        #     it will print arguments as markdown json
        #   - append content to stage "## Response: \n"
        if tool and tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r")
            stage.append_content("## Response: \n")
        # 5. Execute tool
        tool_message = await tool.execute(
            ToolCallParams(
                tool_call=tool_call,
                api_key=api_key,
                choice=choice,
                conversation_id=conversation_id,
                stage=stage,
            )
        )
        # 6. Close stage with StageProcessor
        StageProcessor.close_stage_safely(stage)
        # 7. Return tool message as dict and don't forget to exclude none
        return tool_message.dict(exclude_none=True)
