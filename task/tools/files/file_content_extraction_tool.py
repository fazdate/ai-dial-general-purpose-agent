import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionTool(BaseTool):
    """
    Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
    PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
    USAGE: Start with page=1 (by default)
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "file_content_extraction"

    @property
    def description(self) -> str:
        return ("Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM. "
                "PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated. "
                "USAGE: Start with page=1. If paginated, call again with page=2, page=3, etc. to get remaining content. "
                "Always check response end for pagination info before answering user queries about file content.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "The URL of the file to extract content from."
                },
                "page": {
                    "type": "integer",
                    "description": "For large documents pagination is enabled. Each page consists of 10000 characters.",
                    "default": 1
                }
            },
            "required": ["file_url"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        args = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `file_url` from arguments
        file_url = args.get("file_url")
        # 3. Get `page` from arguments (if none, set as 1 by default)
        page = args.get("page", 1)
        # 4. Get stage from `tool_call_params`
        stage = tool_call_params.stage
        # 5. Append content to stage: "## Request arguments: \n"
        stage.append_content("## Request arguments: \n")
        # 6. Append content to stage: `f"**File URL**: {file_url}\n\r"`
        stage.append_content(f"**File URL**: {file_url}\n\r")
        # 7. If `page` more than 1 then append content to stage: `f"**Page**: {page}\n\r"`
        if page > 1:
            stage.append_content(f"**Page**: {page}\n\r")
        # 8. Append content to stage: "## Response: \n"
        stage.append_content("## Response: \n")
        # 9. Implement `task.utils.dial_file_conent_extractor`, create DialFileContentExtractor and call `extract_text`
        #    method as `content`
        content = DialFileContentExtractor(
            endpoint=self.endpoint,
            api_key=tool_call_params.api_key
        ).extract_text(file_url=file_url)
        # 10. If no `content` present then set it as "Error: File content not found."
        if not content:
            content = "Error: File content not found."
        # 11. If `content` len is more than 10_000 then we need to enable pagination:
        #       - create variable `page_size` as 10_000
        #       - calculate total pages, formula: (`content len` + `page_size` - 1) // `page_size`
        #       - if `page` is less then 1 (potential hallucination from LLM) then set it as 1
        #       - otherwise check if page > total pages (potential hallucination), it yes then set `content` as
        #         `f"Error: Page {page} does not exist. Total pages: {total_pages}"`
        #       - prepare `start_index`: `(page - 1) * page_size`
        #       - prepare `end_index`: `start_index + page_size`
        #       - get page content from `content` that will start with `start_index` and end with `end_index`
        #       - set `content` as `f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"` (It will show to
        #         LLM that it is not full content and it is pageable)
        if len(content) > 10_000:
            page_size = 10_000
            total_pages = (len(content) + page_size - 1) // page_size
            if page < 1:
                page = 1
            elif page > total_pages:
                content = f"Error: Page {page} does not exist. Total pages: {total_pages}"
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            page_content = content[start_index:end_index]
            content = f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"
        # 12. Append content to stage: `f"```text\n\r{content}\n\r```\n\r"` (Will be shown in stage as markdown text)
        stage.append_content(f"```text\n\r{content}\n\r```\n\r")
        # 13. Return `content`
        return content
