"""
SearchR1 Tool Manager for RL-Factory.

This manager aligns RL-Factory's Search R1 behavior with Agent Lightning's approach:
- Uses <search>query</search> and <answer>answer</answer> tags (no <tool_call>)
- No JSON schema tool definitions in the system prompt
- Search results returned as <information>...</information> blocks
- Prompt uses the standard Search R1 instruction format from the paper
"""

import re
import asyncio
import requests
from typing import List, Optional, Dict, Any, Tuple, Union

from omegaconf import OmegaConf
from envs.tool_manager.base_manager import ToolManager

SEARCH_R1_INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as many "
    "times as your want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)


def retrieve_doc(query: str, topk: int = 3) -> str:
    """Retrieve documents from the local RAG service."""
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    proxies = {"http": None, "https": None}
    response = requests.post(
        "http://127.0.0.1:8000/retrieve",
        json=payload,
        proxies=proxies,
        timeout=10,
    )
    response.raise_for_status()
    json_resp = response.json()
    retrieval_result = json_resp["result"][0]
    return passages2string(retrieval_result)


def passages2string(retrieval_result: list) -> str:
    """Format retrieval results into a readable string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


class SearchR1Manager(ToolManager):
    """
    Tool manager that follows the Search R1 paper's approach, aligned with Agent Lightning.

    Instead of using Qwen3's tool-call format (<tool_call> with JSON schema),
    this manager uses simple XML tags:
    - <search>query</search> for search actions
    - <answer>answer</answer> for final answers
    - <information>...</information> for search results
    """

    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.create(verl_config)
        super().__init__(verl_config)

    def _build_tools(self):
        """No MCP tools needed - search is handled directly via HTTP."""
        self.functions = []
        self.tool_map = {}

    @property
    def all_tools(self):
        return self.tool_map

    def parse_response(self, response_content: str) -> Tuple[str, Any]:
        """Parse the model response to extract search or answer actions.

        Returns:
            Tuple of (action_type, content) where:
            - action_type is 'answer' if <answer> tag found
            - action_type is 'actions' if <search> tag found
            - content is the extracted answer text or list of search tool dicts
        """
        # Check for <answer> tag first
        if_answer, answer = self.parse_end_flag(response_content)
        if if_answer:
            return 'answer', answer

        # Check for <search> tag
        search_match = re.search(r'<search>(.*?)</search>', response_content, re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            return 'actions', [{"name": "search", "args": query}]

        # No valid action found - treat as continuing response
        return 'answer', response_content

    def parse_end_flag(self, response_content: str) -> Tuple[bool, Optional[str]]:
        """Check if the response contains an <answer> tag."""
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            return True, answer_section[-1]
        return False, None

    def parse_tools(self, response: str):
        """Parse <search> tags from the response."""
        search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            return [{"name": "search", "args": query}]
        return response

    def execute_actions(self, responses: List[str]):
        """Execute search actions extracted from model responses.

        Returns plain text results (not wrapped in chat-template role markers)
        to align with Agent Lightning's approach of concatenating raw text.
        """
        actions, tool_results_list = [], []
        for response in responses:
            action, content = self.parse_response(response_content=response)
            actions.append(action)

            if action == 'actions':
                # Execute search for each query
                results = []
                for tool in content:
                    query = tool["args"]
                    try:
                        search_result = retrieve_doc(query)
                        result_text = f"\n\n<information>{search_result}</information>\n\n"
                    except Exception as e:
                        result_text = f"\n\n<information>Search failed: {str(e)}</information>\n\n"
                    # Use 'user' role to avoid Qwen3 chat template wrapping with <tool_response>
                    results.append({"role": "user", "content": result_text})
                tool_results_list.append(results)
            else:
                tool_results_list.append({"role": "assistant", "content": content})

        return actions, tool_results_list

    async def execute_all_tools(self, actions, tool_list):
        """Async execution of search actions, compatible with chat_scheduler interface."""
        results = []
        for action, tools in zip(actions, tool_list):
            result = await self._execute_search_batch(action, tools)
            results.append(result)
        return results

    async def _execute_search_batch(self, action, tools):
        """Execute a batch of search operations."""
        if action == 'answer':
            return {"role": "assistant", "content": tools}
        elif action == 'actions':
            search_results = []
            for tool in tools:
                query = tool["args"]
                try:
                    search_result = await asyncio.to_thread(retrieve_doc, query)
                    result_text = f"\n\n<information>{search_result}</information>\n\n"
                except Exception as e:
                    result_text = f"\n\n<information>Search failed: {str(e)}</information>\n\n"
                # Use 'user' role to avoid Qwen3 chat template wrapping with <tool_response>
                search_results.append({"role": "user", "content": result_text})
            return search_results
        else:
            return {"role": "assistant", "content": str(tools)}

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        """Build prompt for SearchR1.

        For 'initial' mode: applies chat template to the initial conversation.
        For 'tool_call' mode: returns raw text content WITHOUT chat template markers,
            aligning with Agent Lightning's approach of plain text concatenation.
            This avoids inserting multi-turn role markers (assistant/user) between turns.
        """
        assert mode in ['initial', 'tool_call', 'assistant_response'], f'Invalid mode: {mode}'

        if mode == 'initial':
            chat = input_data
            enable_thinking = getattr(self.verl_config, 'enable_thinking', True)
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
            return prompt_with_chat_template
        elif mode in ['tool_call', 'assistant_response']:
            # For SearchR1, return raw text content WITHOUT applying chat template.
            # This aligns with AGL's approach: all turns are concatenated as plain text
            # within a single user message, no multi-turn role markers in between.
            if isinstance(input_data, str):
                return input_data
            elif isinstance(input_data, list):
                # Extract text content from message dicts
                return ''.join(msg.get('content', '') for msg in input_data)
            else:
                raise ValueError(f'Unexpected type of input_data {type(input_data)} ({input_data})')
        else:
            raise ValueError(f'Invalid mode: {mode}')
