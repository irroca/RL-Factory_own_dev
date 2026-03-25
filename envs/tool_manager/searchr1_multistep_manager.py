"""
SearchR1 Tool Managers with explicit user→assistant turn alternation.

These managers wrap environment feedback (search results) as user-role messages
with proper chat template markers, creating explicit turn boundaries between
each model generation.

Contrast with the base SearchR1Manager which concatenates raw text without
any role markers — leaving everything in one continuous assistant turn.

Token sequence comparison (response tokens do NOT include <|im_end|> since
vLLM uses include_stop_str_in_output=False by default):

  SearchR1Manager (raw concatenation, one long assistant turn):
    <|im_start|>user\\n{Q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n
    {resp1}                              ← model generates (no <|im_end|> in tokens)
    \\n\\n<information>...\\n\\n             ← raw info text appended directly
    {resp2}                              ← model continues generating

  SearchR1MultistepManager (user→assistant turn alternation):
    <|im_start|>user\\n{Q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n
    {resp1}                              ← model generates (no <|im_end|> in tokens)
    <|im_end|>\\n<|im_start|>user\\n      ← close assistant, open user turn
    \\n\\n<information>...\\n\\n             ← info as user content
    <|im_end|>\\n<|im_start|>assistant\\n  ← close user, open fresh assistant turn
    <think>\\n
    {resp2}                              ← model generates from fresh <think>
"""

from envs.tool_manager.searchr1_manager import SearchR1Manager
from envs.tool_manager.multi_domain_searchr1_manager import MultiDomainSearchR1Manager


def _multistep_get_prompt(input_data, tokenizer, verl_config, add_generation_prompt):
    """Build prompt that creates explicit user→assistant turn alternation.

    Since vLLM response tokens do NOT include <|im_end|> (include_stop_str_in_output
    defaults to False), this function prepends <|im_end|>\\n to:
    1. Close the previous assistant turn
    2. Then wraps the content as a user-role message
    3. Then opens a fresh assistant turn with <think>\\n

    The resulting string, when tokenized and appended to response tokens, produces:
        {resp}<|im_end|>\\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n
    """
    if isinstance(input_data, str):
        content = input_data
    elif isinstance(input_data, list):
        content = ''.join(msg.get('content', '') for msg in input_data)
    else:
        raise ValueError(f'Unexpected type of input_data {type(input_data)} ({input_data})')

    enable_thinking = getattr(verl_config, 'enable_thinking', True)

    # Directly construct the turn transition string.
    # <|im_end|>\n  — close previous assistant turn (resp tokens don't include it)
    # <|im_start|>user\n{content}<|im_end|>\n  — user message with env feedback
    # <|im_start|>assistant\n<think>\n  — fresh assistant generation prompt
    result = "<|im_end|>\n<|im_start|>user\n" + content + "<|im_end|>\n"

    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
        if enable_thinking:
            result += "<think>\n"
        else:
            result += "<think>\n\n</think>\n\n"

    return result


class SearchR1MultistepManager(SearchR1Manager):
    """
    SearchR1 manager with explicit user→assistant turn alternation.

    Compared to the base SearchR1Manager (which concatenates raw text for
    multi-turn sequences, keeping everything in one long assistant turn),
    this manager:
    1. Closes the previous assistant turn with <|im_end|>
    2. Wraps environment feedback as a user-role message
    3. Starts a fresh assistant turn with <|im_start|>assistant\\n<think>\\n

    Each model generation thus starts from a fresh <think> prompt, giving
    the model an explicit "new turn" signal.
    """

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], f'Invalid mode: {mode}'

        if mode == 'initial':
            return super().get_prompt(input_data, tokenizer, mode, add_generation_prompt)
        elif mode in ['tool_call', 'assistant_response']:
            return _multistep_get_prompt(input_data, tokenizer, self.verl_config, add_generation_prompt)
        else:
            raise ValueError(f'Invalid mode: {mode}')


class MultiDomainSearchR1MultistepManager(MultiDomainSearchR1Manager):
    """
    Multi-domain SearchR1 manager with explicit user→assistant turn alternation.

    Same behavior as SearchR1MultistepManager but for multi-domain search.
    """

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], f'Invalid mode: {mode}'

        if mode == 'initial':
            return super().get_prompt(input_data, tokenizer, mode, add_generation_prompt)
        elif mode in ['tool_call', 'assistant_response']:
            return _multistep_get_prompt(input_data, tokenizer, self.verl_config, add_generation_prompt)
        else:
            raise ValueError(f'Invalid mode: {mode}')
