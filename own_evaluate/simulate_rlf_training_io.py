#!/usr/bin/env python3
"""Simulate RL-Factory SearchR1 multi-turn training I/O at the token level.

This script reproduces the EXACT logic from RL-Factory's training pipeline:
  - rl_dataset.py:  initial prompt construction (apply_chat_template)
  - tool_utils.py:  postprocess_output / compose_final_output (token concat)
  - searchr1_manager.py:  get_prompt / execute_actions / parse_response
  - base.py (Env):  step()

It uses mock model responses and mock retrieval results to walk through
the complete multi-turn loop, printing the decoded token sequence at each
step so you can see exactly what vLLM receives as input_ids.

Usage:
    # With a local Qwen3 tokenizer (recommended for token-level accuracy):
    python simulate_rlf_training_io.py --model-path /path/to/Qwen3-0.6B

    # Without a tokenizer (uses string-level simulation):
    python simulate_rlf_training_io.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (same as searchr1_manager.py)
# ---------------------------------------------------------------------------

INSTRUCTION_FORMAT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as many "
    "times as your want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

# ---------------------------------------------------------------------------
# Mock data: question, model responses per turn, mock retrieval results
# ---------------------------------------------------------------------------

MOCK_QUESTION = "What year was the Eiffel Tower completed?"

# Each entry: (model_raw_output, mock_search_result_or_None)
# The model outputs include special tokens as vLLM would produce them.
MOCK_TURNS = [
    # Turn 1: model searches
    {
        "model_output": (
            "<think>\n"
            "I need to find the completion year of the Eiffel Tower. Let me search for this.\n"
            "</think>\n\n"
            "<search>Eiffel Tower completion year</search>"
        ),
        "search_result": (
            "Doc 1(Title: Eiffel Tower) The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece "
            "of the 1889 World's Fair.\n"
            "Doc 2(Title: Construction of Eiffel Tower) Construction began on 28 January 1887 and "
            "was completed on 15 March 1889. The tower was officially opened on 31 March 1889.\n"
            "Doc 3(Title: Paris landmarks) The Eiffel Tower, completed in 1889, stands 330 metres "
            "tall and is the most-visited paid monument in the world.\n"
        ),
    },
    # Turn 2: model searches again for verification
    {
        "model_output": (
            "<think>\n"
            "Based on the search results, the Eiffel Tower was completed in 1889. The construction "
            "ran from 1887 to 1889 and it was officially opened on March 31, 1889. Let me verify "
            "this with one more search.\n"
            "</think>\n\n"
            "<search>Eiffel Tower 1889 opening date</search>"
        ),
        "search_result": (
            "Doc 1(Title: Eiffel Tower opening) The Eiffel Tower was inaugurated on 31 March 1889, "
            "and opened to the public on 6 May 1889. It was built for the 1889 Exposition Universelle.\n"
            "Doc 2(Title: World's Fair 1889) The 1889 Exposition Universelle was held in Paris, "
            "celebrating the centennial of the French Revolution. The Eiffel Tower served as the entrance.\n"
            "Doc 3(Title: Gustave Eiffel) Gustave Eiffel's company designed and built the tower, "
            "completing construction in just over two years.\n"
        ),
    },
    # Turn 3: model gives final answer
    {
        "model_output": (
            "<think>\n"
            "I have confirmed from multiple sources that the Eiffel Tower was completed in 1889.\n"
            "</think>\n\n"
            "<answer>1889</answer>"
        ),
        "search_result": None,  # no search needed
    },
]


# ---------------------------------------------------------------------------
# Simulation with actual tokenizer
# ---------------------------------------------------------------------------

def simulate_with_tokenizer(model_path: str) -> None:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    question = MOCK_QUESTION

    # ===== Step 0: Initial prompt (same as rl_dataset.py) =====
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": INSTRUCTION_FORMAT + question},
    ]

    # This is what rl_dataset.py does via searchr1_manager.get_prompt(mode='initial')
    initial_prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Tokenize (same as rl_dataset.py)
    prompt_token_ids = tokenizer.encode(initial_prompt_str, add_special_tokens=False)

    print("=" * 80)
    print("STEP 0: Initial Prompt (apply_chat_template, enable_thinking=True)")
    print("=" * 80)
    print(f"  Token count: {len(prompt_token_ids)}")
    print(f"  Decoded text:")
    print("-" * 40)
    print(initial_prompt_str)
    print("-" * 40)

    # ===== Simulate multi-turn loop (same as tool_utils.py) =====
    # loop_responses_token[0] = [prompt_tokens, response_1_tokens, info_1_tokens, ...]
    loop_responses_token: List[List[int]] = [prompt_token_ids]

    for turn_idx, turn_data in enumerate(MOCK_TURNS):
        model_output = turn_data["model_output"]
        search_result = turn_data["search_result"]

        # In training, vLLM generates response ending with <|im_end|>
        model_output_with_eos = model_output + "<|im_end|>"

        # Tokenize model response (this is what vLLM produces as response token ids)
        response_token_ids = tokenizer.encode(model_output_with_eos, add_special_tokens=False)
        loop_responses_token.append(response_token_ids)

        print()
        print("=" * 80)
        print(f"STEP {turn_idx * 2 + 1}: Model Response (Turn {turn_idx + 1}) — vLLM output")
        print("=" * 80)
        print(f"  Token count: {len(response_token_ids)}")
        print(f"  Decoded text:")
        print("-" * 40)
        print(tokenizer.decode(response_token_ids, skip_special_tokens=False))
        print("-" * 40)

        if search_result is None:
            # answer action → done
            print(f"  Action: ANSWER → loop ends")
            break

        # ===== env.step() processes the response =====
        # searchr1_manager.execute_actions() produces:
        #   [{"role": "user", "content": "\n\n<information>...\n</information>\n\n"}]
        # searchr1_manager.get_prompt(mode='tool_call') returns raw text:
        #   "\n\n<information>...\n</information>\n\n"
        info_text = f"\n\n<information>{search_result}</information>\n\n"

        # tool_utils.py tokenizes the info text directly (no chat template!)
        info_token_ids = tokenizer.encode(info_text, add_special_tokens=False)
        loop_responses_token.append(info_token_ids)

        print()
        print("=" * 80)
        print(f"STEP {turn_idx * 2 + 2}: Environment Feedback (info tokens, raw tokenize)")
        print("=" * 80)
        print(f"  Token count: {len(info_token_ids)}")
        print(f"  Decoded text:")
        print("-" * 40)
        print(tokenizer.decode(info_token_ids, skip_special_tokens=False))
        print("-" * 40)

        # ===== Show accumulated sequence sent to vLLM for next turn =====
        accumulated = list(itertools.chain.from_iterable(loop_responses_token))
        accumulated_str = tokenizer.decode(accumulated, skip_special_tokens=False)

        print()
        print("=" * 80)
        print(f"ACCUMULATED INPUT for Turn {turn_idx + 2} (sent to vLLM as prompt_token_ids)")
        print("=" * 80)
        print(f"  Total tokens: {len(accumulated)}")
        print(f"  Decoded text:")
        print("-" * 40)
        print(accumulated_str)
        print("-" * 40)

    # ===== Final: compose_final_output (for training) =====
    print()
    print("=" * 80)
    print("FINAL: compose_final_output — Training sequence & loss_mask")
    print("=" * 80)

    # prompt part (not included in response/loss_mask)
    prompt_tokens = loop_responses_token[0]
    print(f"\n  PROMPT tokens: {len(prompt_tokens)} (loss_mask = all 0, not trained)")
    print(f"  Decoded:")
    print(f"    {tokenizer.decode(prompt_tokens, skip_special_tokens=False)[:200]}...")

    # response parts (alternating model/env)
    response_parts = loop_responses_token[1:]
    print(f"\n  RESPONSE segments: {len(response_parts)}")

    total_response_tokens = 0
    for seg_idx, seg_tokens in enumerate(response_parts):
        is_model = (seg_idx % 2 == 0)  # (seg_idx + 1) % 2 in code, but 0-indexed here
        mask_val = 1 if is_model else 0
        role = "MODEL (loss_mask=1, gradient ON)" if is_model else "ENV   (loss_mask=0, gradient OFF)"
        decoded = tokenizer.decode(seg_tokens, skip_special_tokens=False)
        total_response_tokens += len(seg_tokens)

        print(f"\n  --- Segment {seg_idx + 1}: {role} ---")
        print(f"  Token count: {len(seg_tokens)}")
        print(f"  Decoded:")
        # Print first 300 chars for readability
        if len(decoded) > 300:
            print(f"    {decoded[:300]}...")
        else:
            print(f"    {decoded}")

    print(f"\n  Total response tokens (for training): {total_response_tokens}")

    # Show the complete training sequence
    full_sequence = list(itertools.chain.from_iterable(loop_responses_token))
    full_decoded = tokenizer.decode(full_sequence, skip_special_tokens=False)
    print(f"\n  FULL TRAINING SEQUENCE ({len(full_sequence)} tokens):")
    print("-" * 40)
    print(full_decoded)
    print("-" * 40)


# ---------------------------------------------------------------------------
# Simulation without tokenizer (string-level, approximate)
# ---------------------------------------------------------------------------

def simulate_without_tokenizer() -> None:
    print("NOTE: No tokenizer available. Showing string-level simulation.")
    print("      For exact token-level behavior, run with --model-path.\n")

    question = MOCK_QUESTION

    # Step 0: Initial prompt text (what apply_chat_template produces)
    initial_prompt = (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{INSTRUCTION_FORMAT}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )

    print("=" * 80)
    print("STEP 0: Initial Prompt (apply_chat_template, enable_thinking=True)")
    print("=" * 80)
    print(initial_prompt)
    print("-" * 40)

    accumulated = initial_prompt
    segments: List[Tuple[str, str]] = []  # (text, "model"/"env")

    for turn_idx, turn_data in enumerate(MOCK_TURNS):
        model_output = turn_data["model_output"]
        search_result = turn_data["search_result"]

        # vLLM output includes <|im_end|>
        model_output_with_eos = model_output + "<|im_end|>"
        segments.append((model_output_with_eos, "model"))

        print()
        print("=" * 80)
        print(f"STEP {turn_idx * 2 + 1}: Model Response (Turn {turn_idx + 1}) — vLLM output")
        print("=" * 80)
        print(model_output_with_eos)
        print("-" * 40)

        accumulated += model_output_with_eos

        if search_result is None:
            print(f"  Action: ANSWER → loop ends")
            break

        # Raw info text (tokenized directly, no chat template)
        info_text = f"\n\n<information>{search_result}</information>\n\n"
        segments.append((info_text, "env"))

        print()
        print("=" * 80)
        print(f"STEP {turn_idx * 2 + 2}: Environment Feedback (raw text, NO chat template)")
        print("=" * 80)
        print(info_text)
        print("-" * 40)

        accumulated += info_text

        print()
        print("=" * 80)
        print(f"ACCUMULATED INPUT for Turn {turn_idx + 2}")
        print("  (This is what vLLM receives as prompt_token_ids, decoded back)")
        print("=" * 80)
        print(accumulated)
        print("-" * 40)

    # Final training sequence
    print()
    print("=" * 80)
    print("FINAL: Training sequence & loss_mask")
    print("=" * 80)
    print(f"\n  PROMPT (loss_mask = all 0):")
    print(f"    {initial_prompt[:200]}...")

    for seg_idx, (text, role) in enumerate(segments):
        is_model = (role == "model")
        mask = "loss_mask=1, gradient ON" if is_model else "loss_mask=0, gradient OFF"
        label = "MODEL" if is_model else "ENV  "

        print(f"\n  --- Segment {seg_idx + 1}: {label} ({mask}) ---")
        if len(text) > 300:
            print(f"    {text[:300]}...")
        else:
            print(f"    {text}")

    print()
    print("=" * 80)
    print("COMPLETE TRAINING SEQUENCE (prompt + all response segments)")
    print("=" * 80)
    full = initial_prompt + "".join(text for text, _ in segments)
    print(full)
    print("-" * 40)

    # Also show what the eval script's searchr1 mode now produces
    print()
    print("=" * 80)
    print("COMPARISON: eval script searchr1 mode (after fix)")
    print("  Should produce IDENTICAL sequence to training above.")
    print("=" * 80)
    print()
    print("  Turn 1 prompt (build_initial_prompt_rlf):")
    print(f"    = initial_prompt (same as above)")
    print()
    print("  After Turn 1 search:")
    print("    accumulated_prompt += raw_output + info_text")
    print("    = initial_prompt + model_output + '\\n\\n<information>...\\n</information>\\n\\n'")
    print("    → MATCHES training's token concat: prompt_tokens + response_tokens + info_tokens")
    print()
    print("  After Turn 2 search:")
    print("    accumulated_prompt += raw_output + info_text")
    print("    → MATCHES training: appends response_2_tokens + info_2_tokens")
    print()
    print("  Turn 3 answer:")
    print("    accumulated_prompt += raw_output  (no <|im_end|> appended, vLLM stops at it)")
    print()
    print("  NOTE: Training uses stop token <|im_end|> in vLLM sampling_params,")
    print("        so the response tokens include <|im_end|>.")
    print("        The eval completions API also uses stop=['<|im_end|>'],")
    print("        but the stop token is NOT included in resp.choices[0].text.")
    print("        That's why eval appends + '<|im_end|>' explicitly after raw output.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate RL-Factory SearchR1 multi-turn training I/O",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to Qwen3 model (for tokenizer). If not provided, uses string simulation.",
    )
    args = parser.parse_args()

    if args.model_path:
        simulate_with_tokenizer(args.model_path)
    else:
        simulate_without_tokenizer()


if __name__ == "__main__":
    main()
