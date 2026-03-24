"""Unified SearchR1 evaluation script for both Agent Lightning and RL-Factory checkpoints.

Evaluates a trained checkpoint on the SearchR1 QA task using vLLM serving,
producing detailed metrics and optional rollout traces. Works with any
checkpoint from either framework — just point --model at the vLLM-served model.

Quick Start:
    # 1. Merge FSDP checkpoint (if needed):
    python -m verl.model_merger merge --backend fsdp \
        --local_dir <checkpoint>/actor --target_dir <checkpoint>_hf

    # 2. Serve model:
    vllm serve <checkpoint>_hf --port 8001

    # 3. Start retrieval server:
    cd agent-lightning/contrib/recipes/search_r1 && bash retrieval_launch.sh

    # 4. Evaluate:
    python eval_searchr1.py \
        --endpoint http://localhost:8001/v1 \
        --model <checkpoint>_hf \
        --data-file agent-lightning/contrib/recipes/search_r1/data/test.parquet \
        --n-samples 500 \
        --label agl \
        --output-dir eval_results

    # 5. Evaluate RL-Factory checkpoint (uses multi-turn chat format):
    python eval_searchr1.py \
        --endpoint http://localhost:8001/v1 \
        --model <other_checkpoint>_hf \
        --data-file agent-lightning/contrib/recipes/search_r1/data/test.parquet \
        --n-samples 500 \
        --label rlf \
        --output-dir eval_results
    # Note: --label rlf auto-selects --prompt-mode searchr1_multiturn (proper
    # multi-turn system/user/assistant roles matching RLF chat_scheduler training).
    # AGL uses --prompt-mode searchr1 (single user message, default for other labels).
    # For legacy RLF checkpoints trained with <tool_call> format, use:
    #   --prompt-mode qwen3_tool

    # 6. Compare:
    python compare_results.py \
        --agl-metrics eval_results/agl_metrics_*.json \
        --rlf-metrics eval_results/rlf_metrics_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# QA Exact Match scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction: str, golden_answers: List[str]) -> int:
    normalized = normalize_answer(prediction)
    for ga in golden_answers:
        if normalize_answer(ga) == normalized:
            return 1
    return 0


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract last <answer>...</answer> from the response."""
    cleaned = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL)
    matches = list(re.finditer(r"<answer>(.*?)</answer>", cleaned, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_em_score(solution_str: str, golden_answers: List[str]) -> float:
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return float(em_check(answer, golden_answers))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_doc(query: str, retrieval_url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3) -> str:
    import requests
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(retrieval_url, json=payload)
    resp.raise_for_status()
    result = resp.json()["result"][0]
    parts = []
    for idx, item in enumerate(result):
        content = item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        parts.append(f"Doc {idx+1}(Title: {title}) {text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt modes
# ---------------------------------------------------------------------------

# SearchR1 original format (used by Agent Lightning)
INSTRUCTION_FORMAT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as "
    "many times as your want. If you find no further external knowledge needed, you can "
    "directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

# Qwen3 tool-call format: system message with tool definitions (used by RL-Factory)
# NOTE: Must match the exact format from training (MCP flat JSON, tool name = "search-query_rag")
QWEN3_TOOL_SYSTEM_MSG = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"name": "search-query_rag", "description": "MCP RAG Query Tool (Synchronous Version)\\n    \\n    '
    'Args:\\n        query: query text\\n        topk: The default number of documents returned is 3\\n        '
    '\\n    Returns:\\n        str: The formatted query result\\n    ", '
    '"parameters": {"type": "object", "properties": {"query": {"title": "Query", "type": "string"}, '
    '"topk": {"default": 3, "title": "Topk", "type": "integer"}}, "required": ["query"]}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


# ---------------------------------------------------------------------------
# Action parsing — supports <search>/<answer> and <tool_call> formats
# ---------------------------------------------------------------------------

def extract_action(response: str, prompt_mode: str = "searchr1") -> Tuple[Optional[str], str]:
    """Parse action from response.

    In 'searchr1'/'searchr1_multiturn' mode: looks for <search>/<answer> first, then <tool_call>.
    In 'qwen3_tool' mode: looks for <tool_call>/<answer>.
    """
    # Both modes support <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    if prompt_mode in ("searchr1", "searchr1_multiturn"):
        # SearchR1 format: <search>/<answer>
        match = re.search(r"<(search|answer)>(.*?)</\1>", response, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        # Fallback: also try <tool_call> in case model uses it
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if tc_match:
            try:
                call = json.loads(tc_match.group(1).strip())
                if call.get("name") in ("search", "query_rag", "search-query_rag", "rag_query", "RAGQuery"):
                    query = call.get("arguments", {}).get("query", "")
                    return "search", query
            except json.JSONDecodeError:
                pass
    else:
        # Qwen3 tool-call format: <tool_call>/<answer>
        # Check <answer> first (it signals end)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        # Then check <tool_call>
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if tc_match:
            try:
                call = json.loads(tc_match.group(1).strip())
                if call.get("name") in ("query_rag", "search-query_rag", "search", "rag_query", "RAGQuery"):
                    query = call.get("arguments", {}).get("query", "")
                    return "search", query
            except json.JSONDecodeError:
                pass

    return None, ""


def postprocess_response(response: str) -> str:
    if "</search>" in response:
        return response.split("</search>")[0] + "</search>"
    if "</answer>" in response:
        return response.split("</answer>")[0] + "</answer>"
    if "</tool_call>" in response:
        return response.split("</tool_call>")[0] + "</tool_call>"
    return response


# ---------------------------------------------------------------------------
# Single sample evaluation
# ---------------------------------------------------------------------------

def _build_messages_searchr1(
    question: str, rollout_content: str
) -> List[Dict[str, str]]:
    """Build messages in SearchR1 format: single user message with concatenated history."""
    prompt = INSTRUCTION_FORMAT + question
    return [{"role": "user", "content": prompt + rollout_content}]


def _build_messages_searchr1_multiturn(
    question: str,
    assistant_responses: List[str],
    user_feedbacks: List[str],
) -> List[Dict[str, str]]:
    """Build messages in SearchR1 multi-turn format, matching RLF chat_scheduler training.

    Uses proper multi-turn chat messages (system/user/assistant roles) instead of
    concatenating everything into a single user message.
    - System: default Qwen3 system message (no tool definitions)
    - User: SearchR1 instruction + question
    - Assistant: model response
    - User: search results wrapped in <information> tags
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": INSTRUCTION_FORMAT + question},
    ]
    for i, asst_resp in enumerate(assistant_responses):
        messages.append({"role": "assistant", "content": asst_resp})
        if i < len(user_feedbacks):
            messages.append({"role": "user", "content": user_feedbacks[i]})
    return messages


# Qwen3 ChatML special tokens
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_THINK_PREFIX = "<think>\n\n</think>\n\n"


def _build_raw_prompt_qwen3_tool(
    question: str,
    assistant_responses: List[str],
    tool_responses: List[str],
) -> str:
    """Build raw prompt string in Qwen3 tool-call format, exactly matching
    the RL-Factory GRPO training format (token-level faithful).

    The prompt is constructed manually instead of using the chat template API
    because during training:
    1. tools are passed to apply_chat_template (activating the tools branch)
    2. enable_thinking=False injects <think>\\n\\n</think>\\n\\n before every
       assistant generation (including subsequent turns)
    3. Multi-turn prompts are built by concatenating token IDs directly,
       preserving the empty think block for all assistant responses

    The OpenAI chat API cannot reproduce this exactly because the Qwen3
    template strips empty think blocks from intermediate assistant messages.
    """
    # System message with tool definitions
    prompt = (
        f"{_IM_START}system\n"
        f"{QWEN3_TOOL_SYSTEM_MSG}{_IM_END}\n"
    )
    # User question
    prompt += (
        f"{_IM_START}user\n"
        f"{INSTRUCTION_FORMAT}{question}{_IM_END}\n"
    )

    # Interleave assistant responses and tool responses
    for i, asst_resp in enumerate(assistant_responses):
        # Each assistant response is prefixed with empty think block
        prompt += (
            f"{_IM_START}assistant\n"
            f"{_THINK_PREFIX}"
            f"{asst_resp}{_IM_END}\n"
        )
        # Tool response (if exists for this turn)
        if i < len(tool_responses):
            prompt += (
                f"{_IM_START}user\n"
                f"<tool_response>\n{tool_responses[i]}\n</tool_response>"
                f"{_IM_END}\n"
            )

    # Generation prompt for next assistant turn (with empty think block)
    prompt += (
        f"{_IM_START}assistant\n"
        f"{_THINK_PREFIX}"
    )
    return prompt


def evaluate_sample(
    client: OpenAI,
    model: str,
    question: str,
    golden_answers: List[str],
    sample_id: str,
    data_source: str,
    temperature: float = 0.0,
    max_turns: int = 4,
    retrieval_url: str = "http://127.0.0.1:8000/retrieve",
    prompt_mode: str = "searchr1",
) -> Dict[str, Any]:
    """Run multi-turn inference on one sample and compute metrics.

    prompt_mode:
        'searchr1'           — Original SearchR1 format (<search>/<information>), single user message.
                               Uses chat completions API. Matches AGL training format.
        'searchr1_multiturn' — SearchR1 format with proper multi-turn chat roles
                               (system/user/assistant). Uses chat completions API with
                               enable_thinking=True. Matches RLF chat_scheduler training format.
        'qwen3_tool'         — Qwen3 tool-call format (<tool_call>/<tool_response>), raw prompt.
                               Uses completions API. For legacy RLF checkpoints.
    """
    rollout_content = ""  # tracks full text for scoring (both modes)
    turn_id = 0
    finished = False
    search_queries: List[str] = []
    turn_details: List[Dict[str, Any]] = []
    total_tokens_in = 0
    total_tokens_out = 0

    # Multi-turn modes: track assistant responses and tool/user responses separately
    qwen3_asst_responses: List[str] = []
    qwen3_tool_responses: List[str] = []
    # searchr1_multiturn mode: track assistant responses and user feedbacks
    mt_asst_responses: List[str] = []
    mt_user_feedbacks: List[str] = []

    t_start = time.time()

    try:
        while turn_id < max_turns and not finished:
            turn_id += 1
            t_turn = time.time()

            if prompt_mode == "searchr1":
                messages = _build_messages_searchr1(question, rollout_content)
                input_prompt_for_log = json.dumps(messages, ensure_ascii=False)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500,
                )
                raw = resp.choices[0].message.content or ""
                usage = resp.usage
            elif prompt_mode == "searchr1_multiturn":
                messages = _build_messages_searchr1_multiturn(
                    question, mt_asst_responses, mt_user_feedbacks,
                )
                input_prompt_for_log = json.dumps(messages, ensure_ascii=False, indent=2)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                raw = resp.choices[0].message.content or ""
                usage = resp.usage
            else:
                raw_prompt = _build_raw_prompt_qwen3_tool(
                    question, qwen3_asst_responses, qwen3_tool_responses,
                )
                input_prompt_for_log = raw_prompt
                resp = client.completions.create(
                    model=model,
                    prompt=raw_prompt,
                    temperature=temperature,
                    max_tokens=500,
                    stop=[_IM_END, "<|endoftext|>"],
                )
                raw = resp.choices[0].text or ""
                usage = resp.usage

            if usage:
                total_tokens_in += usage.prompt_tokens
                total_tokens_out += usage.completion_tokens

            valid_resp = postprocess_response(raw)
            rollout_content += valid_resp
            action, content = extract_action(valid_resp, prompt_mode)

            turn_info: Dict[str, Any] = {
                "turn": turn_id,
                "action": action,
                "action_content": content,
                "input_prompt": input_prompt_for_log,
                "raw_response_unprocessed": raw,
                "raw_response": valid_resp,
                "latency_s": round(time.time() - t_turn, 3),
            }
            if usage:
                turn_info["tokens_in"] = usage.prompt_tokens
                turn_info["tokens_out"] = usage.completion_tokens

            if action == "answer":
                finished = True
                if prompt_mode == "qwen3_tool":
                    qwen3_asst_responses.append(valid_resp)
                elif prompt_mode == "searchr1_multiturn":
                    mt_asst_responses.append(valid_resp)
            elif action == "search":
                search_queries.append(content)
                t_ret = time.time()
                search_result = retrieve_doc(content, retrieval_url=retrieval_url)
                turn_info["retrieval_latency_s"] = round(time.time() - t_ret, 3)
                turn_info["search_result_snippet"] = search_result[:500]

                if prompt_mode == "searchr1":
                    rollout_content += f"\n\n<information>{search_result}</information>\n\n"
                elif prompt_mode == "searchr1_multiturn":
                    mt_asst_responses.append(valid_resp)
                    feedback = f"\n\n<information>{search_result}</information>\n\n"
                    mt_user_feedbacks.append(feedback)
                    rollout_content += feedback
                else:
                    qwen3_asst_responses.append(valid_resp)
                    qwen3_tool_responses.append(search_result)
                    rollout_content += f"\n\n<tool_response>{search_result}</tool_response>\n\n"
            else:
                if prompt_mode == "searchr1":
                    rollout_content += (
                        "\nMy previous action is invalid. If I want to search, I should put "
                        "the query between <search> and </search>. If I want to give the final "
                        "answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                elif prompt_mode == "searchr1_multiturn":
                    mt_asst_responses.append(valid_resp)
                    retry_msg = (
                        "\nMy previous action is invalid. If I want to search, I should put "
                        "the query between <search> and </search>. If I want to give the final "
                        "answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                    mt_user_feedbacks.append(retry_msg)
                    rollout_content += retry_msg
                else:
                    # Invalid action: record as assistant turn, add retry as tool_response
                    qwen3_asst_responses.append(valid_resp)
                    retry_msg = (
                        "Your previous response did not contain a valid tool call or answer. "
                        "Use <tool_call> to search or <answer> to give the final answer."
                    )
                    qwen3_tool_responses.append(retry_msg)
                    rollout_content += "\n" + retry_msg + "\n"

            turn_details.append(turn_info)

        # Force final answer if not finished
        if not finished:
            t_turn = time.time()
            if prompt_mode == "searchr1":
                messages = _build_messages_searchr1(question, rollout_content)
                input_prompt_for_log = json.dumps(messages, ensure_ascii=False)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500,
                )
                raw = resp.choices[0].message.content or ""
                usage = resp.usage
            elif prompt_mode == "searchr1_multiturn":
                messages = _build_messages_searchr1_multiturn(
                    question, mt_asst_responses, mt_user_feedbacks,
                )
                input_prompt_for_log = json.dumps(messages, ensure_ascii=False, indent=2)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                raw = resp.choices[0].message.content or ""
                usage = resp.usage
            else:
                raw_prompt = _build_raw_prompt_qwen3_tool(
                    question, qwen3_asst_responses, qwen3_tool_responses,
                )
                input_prompt_for_log = raw_prompt
                resp = client.completions.create(
                    model=model,
                    prompt=raw_prompt,
                    temperature=temperature,
                    max_tokens=500,
                    stop=[_IM_END, "<|endoftext|>"],
                )
                raw = resp.choices[0].text or ""
                usage = resp.usage

            if usage:
                total_tokens_in += usage.prompt_tokens
                total_tokens_out += usage.completion_tokens
            rollout_content += raw
            turn_details.append({
                "turn": turn_id + 1,
                "action": "forced_final",
                "input_prompt": input_prompt_for_log,
                "raw_response_unprocessed": raw,
                "raw_response": raw,
                "latency_s": round(time.time() - t_turn, 3),
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
            })

    except Exception as e:
        return {
            "id": sample_id,
            "question": question,
            "data_source": data_source,
            "error": str(e),
            "em_score": 0.0,
        }

    total_time = time.time() - t_start

    em = compute_em_score(rollout_content, golden_answers)
    predicted = extract_solution(rollout_content)

    has_answer_tag = bool(re.search(r"<answer>.*?</answer>", rollout_content, re.DOTALL))
    has_think_tag = bool(re.search(r"<think>.*?</think>", rollout_content, re.DOTALL))
    has_tool_call_tag = bool(re.search(r"<tool_call>.*?</tool_call>", rollout_content, re.DOTALL))

    return {
        "id": sample_id,
        "question": question,
        "data_source": data_source,
        "golden_answers": golden_answers,
        "predicted_answer": predicted,
        "em_score": em,
        "num_turns": turn_id + (0 if finished else 1),
        "num_searches": len(search_queries),
        "search_queries": search_queries,
        "has_answer_tag": has_answer_tag,
        "has_think_tag": has_think_tag,
        "has_tool_call_tag": has_tool_call_tag,
        "total_time_s": round(total_time, 3),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "turn_details": turn_details,
        "full_response": rollout_content,
        "prompt_mode": prompt_mode,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    em_scores = [r["em_score"] for r in valid]

    metrics: Dict[str, Any] = {
        "total_samples": len(results),
        "successful_samples": len(valid),
        "error_samples": len(errors),
        "overall_em_accuracy": sum(em_scores) / len(em_scores) if em_scores else 0.0,
    }

    if valid:
        turns = [r["num_turns"] for r in valid]
        metrics["avg_turns"] = sum(turns) / len(turns)
        metrics["max_turns"] = max(turns)
        metrics["min_turns"] = min(turns)

        searches = [r["num_searches"] for r in valid]
        metrics["avg_searches"] = sum(searches) / len(searches)
        metrics["samples_with_search"] = sum(1 for s in searches if s > 0)
        metrics["search_rate"] = metrics["samples_with_search"] / len(valid)

        metrics["answer_tag_rate"] = sum(1 for r in valid if r["has_answer_tag"]) / len(valid)
        metrics["think_tag_rate"] = sum(1 for r in valid if r["has_think_tag"]) / len(valid)
        metrics["tool_call_tag_rate"] = sum(1 for r in valid if r.get("has_tool_call_tag", False)) / len(valid)

        times = [r["total_time_s"] for r in valid]
        metrics["avg_time_s"] = round(sum(times) / len(times), 3)
        metrics["p50_time_s"] = round(sorted(times)[len(times) // 2], 3)
        metrics["p90_time_s"] = round(sorted(times)[int(len(times) * 0.9)], 3)
        metrics["total_time_s"] = round(sum(times), 3)

        tokens_in = [r["total_tokens_in"] for r in valid]
        tokens_out = [r["total_tokens_out"] for r in valid]
        metrics["avg_tokens_in"] = round(sum(tokens_in) / len(tokens_in), 1)
        metrics["avg_tokens_out"] = round(sum(tokens_out) / len(tokens_out), 1)
        metrics["total_tokens"] = sum(tokens_in) + sum(tokens_out)

        by_source: Dict[str, List[float]] = defaultdict(list)
        for r in valid:
            by_source[r["data_source"]].append(r["em_score"])
        metrics["per_source_em"] = {
            src: {"accuracy": sum(scores) / len(scores), "count": len(scores)}
            for src, scores in sorted(by_source.items())
        }

        correct = [r for r in valid if r["em_score"] > 0]
        incorrect = [r for r in valid if r["em_score"] == 0]
        if correct:
            metrics["correct_avg_turns"] = sum(r["num_turns"] for r in correct) / len(correct)
            metrics["correct_avg_searches"] = sum(r["num_searches"] for r in correct) / len(correct)
        if incorrect:
            metrics["incorrect_avg_turns"] = sum(r["num_turns"] for r in incorrect) / len(incorrect)
            metrics["incorrect_avg_searches"] = sum(r["num_searches"] for r in incorrect) / len(incorrect)

    return metrics


def print_metrics_report(metrics: Dict[str, Any], label: str, model_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"  SearchR1 Evaluation Report [{label}]")
    print(f"  Model: {model_name}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n--- Overall Performance ---")
    print(f"  Total Samples:       {metrics.get('total_samples', 0)}")
    print(f"  Successful:          {metrics.get('successful_samples', 0)}")
    print(f"  Errors:              {metrics.get('error_samples', 0)}")
    print(f"  EM Accuracy:         {metrics.get('overall_em_accuracy', 0):.4f}")

    print(f"\n--- Search Behavior ---")
    print(f"  Avg Turns:           {metrics.get('avg_turns', 0):.2f}")
    print(f"  Avg Searches:        {metrics.get('avg_searches', 0):.2f}")
    print(f"  Search Rate:         {metrics.get('search_rate', 0):.2%}")

    print(f"\n--- Format Compliance ---")
    print(f"  Answer Tag Rate:     {metrics.get('answer_tag_rate', 0):.2%}")
    print(f"  Think Tag Rate:      {metrics.get('think_tag_rate', 0):.2%}")
    print(f"  Tool Call Tag Rate:  {metrics.get('tool_call_tag_rate', 0):.2%}")

    print(f"\n--- Latency ---")
    print(f"  Avg Time/Sample:     {metrics.get('avg_time_s', 0):.3f}s")
    print(f"  P50 Time:            {metrics.get('p50_time_s', 0):.3f}s")
    print(f"  P90 Time:            {metrics.get('p90_time_s', 0):.3f}s")
    print(f"  Total Time:          {metrics.get('total_time_s', 0):.1f}s")

    print(f"\n--- Token Usage ---")
    print(f"  Avg Input Tokens:    {metrics.get('avg_tokens_in', 0):.0f}")
    print(f"  Avg Output Tokens:   {metrics.get('avg_tokens_out', 0):.0f}")
    print(f"  Total Tokens:        {metrics.get('total_tokens', 0)}")

    per_source = metrics.get("per_source_em", {})
    if per_source:
        print(f"\n--- Per Data Source ---")
        for src, info in per_source.items():
            print(f"  {src:20s}  EM={info['accuracy']:.4f}  (n={info['count']})")

    if "correct_avg_turns" in metrics:
        print(f"\n--- Correct vs Incorrect ---")
        print(f"  Correct Avg Turns:   {metrics.get('correct_avg_turns', 0):.2f}")
        print(f"  Correct Avg Search:  {metrics.get('correct_avg_searches', 0):.2f}")
        print(f"  Incorrect Avg Turns: {metrics.get('incorrect_avg_turns', 0):.2f}")
        print(f"  Incorrect Avg Search:{metrics.get('incorrect_avg_searches', 0):.2%}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Rollout report
# ---------------------------------------------------------------------------

def _write_rollout_report(
    results: List[Dict[str, Any]], path: str, label: str, model: str
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# SearchR1 Detailed Rollout Report [{label}]\n\n")
        f.write(f"**Model:** {model}\n\n")
        f.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Samples:** {len(results)}\n\n")

        correct = sum(1 for r in results if r.get('em_score', 0) > 0)
        f.write(f"**Overall EM:** {correct}/{len(results)} = {correct/len(results):.2%}\n\n")
        f.write("---\n\n")

        for i, r in enumerate(results):
            em_icon = "\u2705" if r.get("em_score", 0) > 0 else "\u274c"
            f.write(f"## Sample {i+1}: {r.get('id', '?')} {em_icon}\n\n")
            f.write(f"- **Data Source:** {r.get('data_source', '?')}\n")
            f.write(f"- **Question:** {r.get('question', '?')}\n")
            f.write(f"- **Golden Answers:** {r.get('golden_answers', [])}\n")
            f.write(f"- **Predicted Answer:** {r.get('predicted_answer', 'None')}\n")
            f.write(f"- **EM Score:** {r.get('em_score', 0)}\n")
            f.write(f"- **Turns:** {r.get('num_turns', 0)} | "
                    f"Searches: {r.get('num_searches', 0)} | "
                    f"Time: {r.get('total_time_s', 0)}s\n")
            f.write(f"- **Tokens:** in={r.get('total_tokens_in', 0)} "
                    f"out={r.get('total_tokens_out', 0)}\n\n")

            if "error" in r:
                f.write(f"**ERROR:** {r['error']}\n\n")
            else:
                f.write("### Turn-by-Turn Trace\n\n")
                for t in r.get("turn_details", []):
                    f.write(f"#### Turn {t['turn']} \u2014 Action: `{t.get('action', '?')}`\n\n")
                    if t.get("input_prompt"):
                        input_text = t["input_prompt"]
                        f.write(f"**Input Prompt (sent to model):**\n```\n{input_text}\n```\n\n")
                    if t.get("raw_response_unprocessed"):
                        raw_unproc = t["raw_response_unprocessed"]
                        f.write(f"**Raw Model Output (unprocessed):**\n```\n{raw_unproc}\n```\n\n")
                    if t.get("raw_response"):
                        resp_text = t["raw_response"]
                        f.write(f"**Postprocessed Response:**\n```\n{resp_text}\n```\n\n")
                    if t.get("action_content"):
                        f.write(f"**Action Content:** {t['action_content']}\n\n")
                    if t.get("search_result_snippet"):
                        f.write(f"**Search Result (snippet):**\n```\n{t['search_result_snippet']}\n```\n\n")
                    f.write(f"**Latency:** {t.get('latency_s', '?')}s")
                    if t.get("retrieval_latency_s"):
                        f.write(f" (retrieval: {t['retrieval_latency_s']}s)")
                    f.write(f" | Tokens: in={t.get('tokens_in', '?')} out={t.get('tokens_out', '?')}\n\n")

            f.write("---\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SearchR1 evaluation for Agent Lightning / RL-Factory checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--endpoint", type=str, required=True,
                        help="vLLM API endpoint (e.g., http://localhost:8001/v1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name as registered in vLLM (must match vllm serve path)")
    parser.add_argument("--label", type=str, default="eval",
                        help="Label for output files, e.g. 'agl' or 'rlf' (default: eval)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to test parquet file")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--retrieval-url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--rollout-file", type=str, default=None,
                        help="Small parquet/json for detailed rollout trace")
    parser.add_argument("--prompt-mode", type=str, default=None,
                        choices=["searchr1", "searchr1_multiturn", "qwen3_tool"],
                        help="Prompt format: 'searchr1' for <search> tags in single user message (AGL), "
                             "'searchr1_multiturn' for <search> tags with proper multi-turn chat roles (RLF), "
                             "'qwen3_tool' for legacy RLF checkpoints trained with <tool_call> + system tools. "
                             "Default: searchr1 for agl, searchr1_multiturn for rlf.")
    args = parser.parse_args()

    # Auto-detect prompt mode from label
    # AGL uses searchr1 (single user message), RLF uses searchr1_multiturn (proper multi-turn chat).
    # Legacy RLF checkpoints trained with <tool_call> format need qwen3_tool (specify explicitly).
    if args.prompt_mode is None:
        if args.label == "rlf":
            args.prompt_mode = "searchr1_multiturn"
        else:
            args.prompt_mode = "searchr1"
    print(f"Prompt mode: {args.prompt_mode}")

    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.data_file)
    if args.n_samples is not None:
        df = df.head(args.n_samples)
    dataset = df.to_dict(orient="records")
    print(f"Loaded {len(dataset)} samples from {args.data_file}")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "token-abc123")
    client = OpenAI(base_url=args.endpoint, api_key=api_key)

    # ---- Main evaluation ----
    results: List[Dict[str, Any]] = []
    for i, sample in enumerate(dataset):
        golden = list(sample["golden_answers"])
        sid = sample.get("id", f"sample_{i}")
        dsrc = sample.get("data_source", "unknown")

        record = evaluate_sample(
            client=client, model=args.model,
            question=sample["question"], golden_answers=golden,
            sample_id=sid, data_source=dsrc,
            temperature=args.temperature, max_turns=args.max_turns,
            retrieval_url=args.retrieval_url,
            prompt_mode=args.prompt_mode,
        )
        results.append(record)

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            running_em = sum(r["em_score"] for r in results) / len(results)
            print(f"  [{i+1}/{len(dataset)}] Running EM: {running_em:.4f}")

    metrics = compute_aggregate_metrics(results)
    metrics["label"] = args.label
    metrics["framework"] = args.label
    metrics["model"] = args.model
    metrics["temperature"] = args.temperature
    metrics["max_turns"] = args.max_turns
    metrics["data_file"] = args.data_file
    metrics["prompt_mode"] = args.prompt_mode

    print_metrics_report(metrics, args.label, args.model)

    # ---- Save outputs ----
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = os.path.basename(args.model.rstrip("/"))
    prefix = f"{args.label}_"

    metrics_path = os.path.join(args.output_dir, f"{prefix}metrics_{model_tag}_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    details_path = os.path.join(args.output_dir, f"{prefix}details_{model_tag}_{timestamp}.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"Per-sample details saved to {details_path}")

    # ---- Detailed rollout on small sample set ----
    if args.rollout_file:
        print(f"\n--- Running detailed rollout on {args.rollout_file} ---")
        if args.rollout_file.endswith(".parquet"):
            rollout_df = pd.read_parquet(args.rollout_file)
        else:
            rollout_df = pd.read_json(args.rollout_file)
        rollout_dataset = rollout_df.to_dict(orient="records")
        print(f"Rollout samples: {len(rollout_dataset)}")

        rollout_results: List[Dict[str, Any]] = []
        for j, sample in enumerate(rollout_dataset):
            golden = list(sample["golden_answers"])
            sid = sample.get("id", f"rollout_{j}")
            dsrc = sample.get("data_source", "unknown")
            rec = evaluate_sample(
                client=client, model=args.model,
                question=sample["question"], golden_answers=golden,
                sample_id=sid, data_source=dsrc,
                temperature=args.temperature, max_turns=args.max_turns,
                retrieval_url=args.retrieval_url,
                prompt_mode=args.prompt_mode,
            )
            rollout_results.append(rec)
            em_icon = "\u2705" if rec.get("em_score", 0) > 0 else "\u274c"
            print(f"  [{j+1}/{len(rollout_dataset)}] {sid} {em_icon} EM={rec['em_score']}")

        rollout_report_path = os.path.join(
            args.output_dir, f"{prefix}rollout_report_{model_tag}_{timestamp}.md"
        )
        _write_rollout_report(rollout_results, rollout_report_path, args.label, args.model)

        rollout_json_path = os.path.join(
            args.output_dir, f"{prefix}rollout_details_{model_tag}_{timestamp}.json"
        )
        with open(rollout_json_path, "w", encoding="utf-8") as f:
            json.dump(rollout_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Rollout report: {rollout_report_path}")
        print(f"Rollout details: {rollout_json_path}")


if __name__ == "__main__":
    main()
