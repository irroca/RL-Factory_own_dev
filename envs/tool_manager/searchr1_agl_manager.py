"""
AGL-aligned SearchR1 Tool Managers for RL-Factory.

These managers replicate Agent Lightning's (AGL) exact multi-turn rollout
behavior within RLF's training pipeline:

  AGL rollout behavior:
    1. rollout_content = ""
    2. messages = [{"role": "user", "content": INSTRUCTION + Q + rollout_content}]
    3. vLLM chat handler → tokenizer.apply_chat_template(messages) → model input
    4. Model generates response
    5. rollout_content += response + "\\n\\n<information>...\\n\\n"
    6. Repeat: resend full content as single user message

  Key difference from SearchR1Manager / SearchR1MultistepManager:
    - SearchR1Manager: flat concat, info appended in assistant turn (no role markers)
    - SearchR1MultistepManager: info wrapped as user message with turn markers
    - SearchR1AGLManager (this): each turn's prompt is a COMPLETE re-encoding
      via apply_chat_template, with ALL accumulated content inside a single
      user message — exactly as AGL does

  Training token sequence (3-turn example):
    [Prompt1][Resp1][Prompt2][Resp2][Prompt3][Resp3]

    Where:
      Prompt1 = apply_chat_template([{"role":"user","content":"INSTR+Q"}])
      Prompt2 = apply_chat_template([{"role":"user","content":"INSTR+Q+resp1+info1"}])
      Prompt3 = apply_chat_template([{"role":"user","content":"INSTR+Q+resp1+info1+resp2+info2"}])

    Loss mask: 0 for all Prompt tokens, 1 for all Resp tokens.

  Implementation:
    - Sets `reprompt_mode = True` to signal tool_utils.py
    - `build_reprompt()` constructs the full re-encoded prompt for each turn
    - tool_utils.py handles the reprompt assembly (see tool_utils.py for details)
"""

import re
from typing import Optional

from envs.tool_manager.searchr1_manager import SearchR1Manager
from envs.tool_manager.multi_domain_searchr1_manager import MultiDomainSearchR1Manager


class SearchR1AGLManager(SearchR1Manager):
    """
    AGL-aligned SearchR1 manager with full reprompt each turn.

    Each turn's inference prompt is rebuilt from scratch:
      apply_chat_template([{"role": "user", "content": INSTRUCTION + Q + accumulated_rollout}])

    This exactly matches AGL's call_llm() behavior where the full history
    is sent as a single user message every turn.
    """

    reprompt_mode = True  # Signal to tool_utils.py

    def __init__(self, verl_config):
        super().__init__(verl_config)
        # Per-sample state, keyed by batch_idx
        self._user_content: dict[int, str] = {}
        self._rollout_content: dict[int, str] = {}

    def reset_reprompt_state(self, batch_size: int):
        """Reset per-sample state for a new batch."""
        self._user_content = {}
        self._rollout_content = {}

    def init_sample(self, batch_idx: int, user_content: str):
        """Store the initial user message content for a sample."""
        self._user_content[batch_idx] = user_content
        self._rollout_content[batch_idx] = ""

    def accumulate_and_build_reprompt(
        self,
        batch_idx: int,
        response_text: str,
        info_text: str,
        tokenizer,
    ) -> str:
        """Accumulate response + info, return full re-encoded prompt for next turn.

        This replicates AGL's per-turn behavior:
            rollout_content += response + info
            messages = [{"role": "user", "content": prompt + rollout_content}]
            tokenizer.apply_chat_template(messages, ...)

        Args:
            batch_idx: Sample index in the batch
            response_text: Decoded model response text (this turn)
            info_text: Environment feedback text (search result or retry message)
            tokenizer: HF tokenizer with apply_chat_template

        Returns:
            Full prompt string for the next turn, ready to be tokenized
        """
        self._rollout_content[batch_idx] += response_text + info_text

        user_content = self._user_content[batch_idx] + self._rollout_content[batch_idx]
        messages = [{"role": "user", "content": user_content}]

        enable_thinking = getattr(self.verl_config, 'enable_thinking', True)
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return prompt

    @staticmethod
    def extract_user_content(prompt_text: str) -> Optional[str]:
        """Extract user message content from a rendered chat template string.

        Parses the Qwen ChatML format to get the raw user content:
            <|im_start|>user\\n{content}<|im_end|>
        """
        match = re.search(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
            prompt_text,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        return None


class MultiDomainSearchR1AGLManager(MultiDomainSearchR1Manager):
    """
    AGL-aligned multi-domain SearchR1 manager with full reprompt each turn.
    Same reprompt logic as SearchR1AGLManager but for multi-domain search.
    """

    reprompt_mode = True

    def __init__(self, verl_config):
        super().__init__(verl_config)
        self._user_content: dict[int, str] = {}
        self._rollout_content: dict[int, str] = {}

    def reset_reprompt_state(self, batch_size: int):
        self._user_content = {}
        self._rollout_content = {}

    def init_sample(self, batch_idx: int, user_content: str):
        self._user_content[batch_idx] = user_content
        self._rollout_content[batch_idx] = ""

    def accumulate_and_build_reprompt(
        self,
        batch_idx: int,
        response_text: str,
        info_text: str,
        tokenizer,
    ) -> str:
        self._rollout_content[batch_idx] += response_text + info_text

        user_content = self._user_content[batch_idx] + self._rollout_content[batch_idx]
        messages = [{"role": "user", "content": user_content}]

        enable_thinking = getattr(self.verl_config, 'enable_thinking', True)
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return prompt

    @staticmethod
    def extract_user_content(prompt_text: str) -> Optional[str]:
        match = re.search(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
            prompt_text,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        return None
