"""
AGL-aligned SearchR1 Tool Managers for RL-Factory.

These managers match Agent Lightning's (AGL) multi-turn message construction approach:
environment feedback (search results) is wrapped as user-role messages with proper
chat template markers, creating explicit user→assistant turn alternation.

Contrast with the base SearchR1Manager which concatenates raw text without role markers.

Token sequence comparison:
  SearchR1Manager (raw):
    <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
    {response_1}\n\n<information>...\n\n{response_2}...

  SearchR1AGLManager (AGL-aligned):
    <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
    {response_1}<|im_end|>\n
    <|im_start|>user\n\n\n<information>...\n\n<|im_end|>\n<|im_start|>assistant\n
    {response_2}...
"""

from envs.tool_manager.searchr1_manager import SearchR1Manager
from envs.tool_manager.multi_domain_searchr1_manager import MultiDomainSearchR1Manager


def _agl_style_get_prompt(input_data, tokenizer, verl_config, add_generation_prompt):
    """Build AGL-style prompt: wrap environment feedback as a user-role message.

    Uses the chat template base-prompt subtraction trick (same as Qwen3Manager)
    to produce proper role markers around the content:
        <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n[<think>\n]

    This is prepended to the model's next generation, closing the previous assistant
    turn and starting a new user→assistant exchange.
    """
    if isinstance(input_data, str):
        content = input_data
    elif isinstance(input_data, list):
        content = ''.join(msg.get('content', '') for msg in input_data)
    else:
        raise ValueError(f'Unexpected type of input_data {type(input_data)} ({input_data})')

    # Build base prompt for subtraction
    base_chat = [
        {'role': 'system', 'content': 'base'},
        {'role': 'user', 'content': 'base'},
    ]
    base_prompt = tokenizer.apply_chat_template(
        conversation=base_chat,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Build full prompt with the info as a user message
    chat = [{'role': 'user', 'content': content}]
    enable_thinking = getattr(verl_config, 'enable_thinking', True)
    temp_prompt = tokenizer.apply_chat_template(
        conversation=base_chat + chat,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    # Subtract base prefix to get only the user-message portion + generation prompt
    prompt = temp_prompt.replace(base_prompt, '', 1)
    return prompt


class SearchR1AGLManager(SearchR1Manager):
    """
    AGL-aligned SearchR1 manager.

    Identical to SearchR1Manager except that environment feedback (search results)
    is wrapped as a proper user-role message using the chat template, matching
    Agent Lightning's approach of putting all context into user messages.

    This creates explicit user→assistant turn alternation in the token sequence,
    rather than raw text concatenation.
    """

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], f'Invalid mode: {mode}'

        if mode == 'initial':
            return super().get_prompt(input_data, tokenizer, mode, add_generation_prompt)
        elif mode in ['tool_call', 'assistant_response']:
            return _agl_style_get_prompt(input_data, tokenizer, self.verl_config, add_generation_prompt)
        else:
            raise ValueError(f'Invalid mode: {mode}')


class MultiDomainSearchR1AGLManager(MultiDomainSearchR1Manager):
    """
    AGL-aligned multi-domain SearchR1 manager.

    Same as MultiDomainSearchR1Manager but wraps environment feedback as
    user-role messages with chat template markers (AGL-style).
    """

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], f'Invalid mode: {mode}'

        if mode == 'initial':
            return super().get_prompt(input_data, tokenizer, mode, add_generation_prompt)
        elif mode in ['tool_call', 'assistant_response']:
            return _agl_style_get_prompt(input_data, tokenizer, self.verl_config, add_generation_prompt)
        else:
            raise ValueError(f'Invalid mode: {mode}')
