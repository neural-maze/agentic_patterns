import json
import re
from typing import List
from typing import Union

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.tool_pattern.utils import validate_arguments
from agentic_patterns.utils import (
    build_prompt_structure,
)

load_dotenv()


TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>
"""


class ToolAgent:
    def __init__(
        self,
        tools: Union[Tool, List[Tool]],
        model: str = "llama3-groq-70b-8192-tool-use-preview",
    ) -> None:
        self.client = Groq()
        self.model = model
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self):
        tools_info = ""
        for tool in self.tools:
            tools_info += tool.fn_signature
        return tools_info

    def parse_tool_call_str(self, tool_call_str: str):
        pattern = r"</?tool_call>"
        clean_tags = re.sub(pattern, "", tool_call_str)

        try:
            tool_call_json = json.loads(clean_tags)
            return tool_call_json
        except json.JSONDecodeError:
            return clean_tags
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "There was some error parsing the Tool's output"

    def run(
        self,
        user_msg: str,
    ) -> str:

        user_prompt = build_prompt_structure(prompt=user_msg, role="user")

        tool_chat_history: List = [
            build_prompt_structure(
                prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(), role="system"
            ),
            user_prompt,
        ]
        agent_chat_history: List = [user_prompt]

        tool_call_str = (
            self.client.chat.completions.create(
                messages=tool_chat_history, model=self.model
            )
            .choices[0]
            .message.content
        )

        tool_call = self.parse_tool_call_str(str(tool_call_str))

        if isinstance(tool_call, dict):
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            tool_call = validate_arguments(tool_call, json.loads(tool.fn_signature))
            print(Fore.GREEN + f"\nTool call dict: \n {tool_call}")

            result = tool.run(**tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n\n {result}")

            agent_chat_history.append(
                build_prompt_structure(prompt=f'f"Observation: {result}"', role="user")
            )

        output = (
            self.client.chat.completions.create(
                messages=agent_chat_history, model=self.model
            )
            .choices[0]
            .message.content
        )

        return str(output)
