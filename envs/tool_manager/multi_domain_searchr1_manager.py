"""
Multi-Domain SearchR1 Tool Manager for RL-Factory.

Extends the SearchR1 approach with domain-aware retrieval:
- Uses <search> query </search> tags with mandatory domain routing via <domain> tag
- Calls the multi-domain retrieval server (POST /retrieve with domain parameter)
- Supports biomedical, financial, science domains (auto-discovered by the server)
- Search results returned as <information>...</information> blocks
"""

import re
import asyncio
import requests
from typing import List, Optional, Dict, Any, Tuple, Union

from omegaconf import OmegaConf
from envs.tool_manager.base_manager import ToolManager

MULTI_DOMAIN_SEARCH_INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> with the target "
    "domain specified by <domain> domain_name </domain>. Available domains: {domains}. "
    "The search engine will return the top results between <information> and </information>. "
    "You can search as many times as you want, and you may search across different domains. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

# Default retrieval server URL (overridable via config)
DEFAULT_RETRIEVAL_URL = "http://127.0.0.1:8000"


def retrieve_doc_multi_domain(
    query: str,
    domain: str,
    topk: int = 3,
    retrieval_url: str = DEFAULT_RETRIEVAL_URL,
) -> str:
    """Retrieve documents from the multi-domain retrieval service."""
    payload = {
        "queries": [query],
        "domain": domain,
        "topk": topk,
        "return_scores": False,
    }
    proxies = {"http": None, "https": None}
    response = requests.post(
        f"{retrieval_url}/retrieve",
        json=payload,
        proxies=proxies,
        timeout=10,
    )
    response.raise_for_status()
    json_resp = response.json()
    retrieval_result = json_resp["result"][0]
    return passages2string(retrieval_result, domain)


def get_available_domains(retrieval_url: str = DEFAULT_RETRIEVAL_URL) -> List[str]:
    """Query the server for available domains."""
    proxies = {"http": None, "https": None}
    try:
        response = requests.get(
            f"{retrieval_url}/domains",
            proxies=proxies,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()["domains"]
    except Exception:
        return ["biomedical", "financial", "science"]


def passages2string(retrieval_result: list, domain: str) -> str:
    """Format retrieval results into a readable string with domain label."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        text = doc_item.get("text", "")
        format_reference += f"Doc {idx+1}(Domain: {domain}) {text}\n"
    return format_reference


class MultiDomainSearchR1Manager(ToolManager):
    """
    Tool manager for multi-domain search, following the Search R1 paper's approach.

    Uses simple XML tags with an additional <domain> tag for routing:
    - <search>query</search> for search actions
    - <domain>domain_name</domain> for specifying the target domain
    - <answer>answer</answer> for final answers
    - <information>...</information> for search results
    """

    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.create(verl_config)
        super().__init__(verl_config)
        self.retrieval_url = getattr(verl_config, 'retrieval_url', DEFAULT_RETRIEVAL_URL)
        self.topk = getattr(verl_config, 'retrieval_topk', 3)
        self.domains = get_available_domains(self.retrieval_url)
        self.domains_str = ", ".join(self.domains)

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
            # Extract domain tag
            domain_match = re.search(r'<domain>(.*?)</domain>', response_content, re.DOTALL)
            domain = domain_match.group(1).strip() if domain_match else self.domains[0]
            # Validate domain
            if domain not in self.domains:
                domain = self.domains[0]
            return 'actions', [{"name": "search", "args": query, "domain": domain}]

        # No valid action found - treat as continuing response
        return 'answer', response_content

    def parse_end_flag(self, response_content: str) -> Tuple[bool, Optional[str]]:
        """Check if the response contains an <answer> tag."""
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            return True, answer_section[-1]
        return False, None

    def parse_tools(self, response: str):
        """Parse <search> and <domain> tags from the response."""
        search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            domain_match = re.search(r'<domain>(.*?)</domain>', response, re.DOTALL)
            domain = domain_match.group(1).strip() if domain_match else self.domains[0]
            if domain not in self.domains:
                domain = self.domains[0]
            return [{"name": "search", "args": query, "domain": domain}]
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
                results = []
                for tool in content:
                    query = tool["args"]
                    domain = tool["domain"]
                    try:
                        search_result = retrieve_doc_multi_domain(
                            query, domain, self.topk, self.retrieval_url
                        )
                        result_text = f"\n\n<information>{search_result}</information>\n\n"
                    except Exception as e:
                        result_text = f"\n\n<information>Search failed: {str(e)}</information>\n\n"
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
                domain = tool["domain"]
                try:
                    search_result = await asyncio.to_thread(
                        retrieve_doc_multi_domain, query, domain, self.topk, self.retrieval_url
                    )
                    result_text = f"\n\n<information>{search_result}</information>\n\n"
                except Exception as e:
                    result_text = f"\n\n<information>Search failed: {str(e)}</information>\n\n"
                search_results.append({"role": "user", "content": result_text})
            return search_results
        else:
            return {"role": "assistant", "content": str(tools)}

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        """Build prompt for multi-domain SearchR1.

        For 'initial' mode: applies chat template to the initial conversation.
        For 'tool_call' mode: returns raw text content WITHOUT chat template markers,
            aligning with Agent Lightning's approach of plain text concatenation.
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
            if isinstance(input_data, str):
                return input_data
            elif isinstance(input_data, list):
                return ''.join(msg.get('content', '') for msg in input_data)
            else:
                raise ValueError(f'Unexpected type of input_data {type(input_data)} ({input_data})')
        else:
            raise ValueError(f'Invalid mode: {mode}')
