"""Standalone rollout tracing script for SearchR1 models.

Records every turn's raw input prompt and raw model output for a small set of
samples (e.g. 100). Supports both 'searchr1' (<search>/<answer>) and
'qwen3_tool' (<tool_call>/<tool_response>) prompt formats.

Usage:
    # 1. Serve model & start retrieval server (same as eval_searchr1.py)

    # 2. Run rollout trace (searchr1 format, default):
    python rollout_searchr1.py \
        --endpoint http://localhost:8001/v1 \
        --model <checkpoint>_hf \
        --data-file agent-lightning/contrib/recipes/search_r1/data/test.parquet \
        --n-samples 100 \
        --output-dir rollout_results

    # 3. Run rollout trace (qwen3_tool format for legacy RLF):
    python rollout_searchr1.py \
        --endpoint http://localhost:8001/v1 \
        --model <checkpoint>_hf \
        --data-file agent-lightning/contrib/recipes/search_r1/data/test.parquet \
        --n-samples 100 \
        --prompt-mode qwen3_tool \
        --output-dir rollout_results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# QA scoring helpers
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
# Prompt templates (identical to eval_searchr1.py)
# ---------------------------------------------------------------------------

INSTRUCTION_FORMAT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as "
    "many times as your want. If you find no further external knowledge needed, you can "
    "directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

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
# Action parsing
# ---------------------------------------------------------------------------

def extract_action(response: str, prompt_mode: str = "searchr1") -> Tuple[Optional[str], str]:
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    if prompt_mode in ("searchr1", "searchr1_multiturn"):
        match = re.search(r"<(search|answer)>(.*?)</\1>", response, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
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
        if answer_match:
            return "answer", answer_match.group(1).strip()
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
# Prompt building
# ---------------------------------------------------------------------------

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_THINK_PREFIX = "<think>\n\n</think>\n\n"


def build_messages_searchr1(question: str, rollout_content: str) -> List[Dict[str, str]]:
    prompt = INSTRUCTION_FORMAT + question
    return [{"role": "user", "content": prompt + rollout_content}]


def build_messages_searchr1_multiturn(
    question: str,
    assistant_responses: List[str],
    user_feedbacks: List[str],
) -> List[Dict[str, str]]:
    """Build messages in SearchR1 multi-turn format, matching RLF chat_scheduler training.

    Uses proper multi-turn chat messages (system/user/assistant roles).
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


def build_initial_prompt_searchr1_multiturn(question: str) -> str:
    """Build initial raw prompt matching RLF training's token-level format.

    Reproduces tokenizer.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": INSTRUCTION_FORMAT + question}],
        add_generation_prompt=True, enable_thinking=True, tokenize=False)

    Subsequent turns are built by raw string concatenation:
        accumulated += response + "<|im_end|>" + raw_info_text
    matching the token-level concatenation in RLF training's ToolUtils
    (no additional role markers between turns).
    """
    return (
        f"{_IM_START}system\n"
        f"{_IM_END}\n"
        f"{_IM_START}user\n"
        f"{INSTRUCTION_FORMAT}{question}{_IM_END}\n"
        f"{_IM_START}assistant\n"
        f"<think>\n"
    )


def build_raw_prompt_qwen3_tool(
    question: str,
    assistant_responses: List[str],
    tool_responses: List[str],
) -> str:
    prompt = (
        f"{_IM_START}system\n"
        f"{QWEN3_TOOL_SYSTEM_MSG}{_IM_END}\n"
    )
    prompt += (
        f"{_IM_START}user\n"
        f"{INSTRUCTION_FORMAT}{question}{_IM_END}\n"
    )
    for i, asst_resp in enumerate(assistant_responses):
        prompt += (
            f"{_IM_START}assistant\n"
            f"{_THINK_PREFIX}"
            f"{asst_resp}{_IM_END}\n"
        )
        if i < len(tool_responses):
            prompt += (
                f"{_IM_START}user\n"
                f"<tool_response>\n{tool_responses[i]}\n</tool_response>"
                f"{_IM_END}\n"
            )
    prompt += (
        f"{_IM_START}assistant\n"
        f"{_THINK_PREFIX}"
    )
    return prompt


# ---------------------------------------------------------------------------
# Single sample rollout — full trace recording
# ---------------------------------------------------------------------------

def run_sample_rollout(
    client: OpenAI,
    model: str,
    question: str,
    golden_answers: List[str],
    sample_id: str,
    data_source: str,
    temperature: float = 0.0,
    max_turns: int = 4,
    max_tokens: int = 2048,
    retrieval_url: str = "http://127.0.0.1:8000/retrieve",
    prompt_mode: str = "searchr1",
) -> Dict[str, Any]:
    """Run multi-turn inference, recording every raw input/output per turn."""

    rollout_content = ""
    turn_id = 0
    finished = False
    search_queries: List[str] = []
    turns: List[Dict[str, Any]] = []
    total_tokens_in = 0
    total_tokens_out = 0

    qwen3_asst_responses: List[str] = []
    qwen3_tool_responses: List[str] = []
    # searchr1_multiturn mode: accumulated raw prompt (matches RLF token concatenation)
    accumulated_prompt = ""

    t_start = time.time()

    try:
        while turn_id < max_turns and not finished:
            turn_id += 1
            t_turn = time.time()

            # --- Build input ---
            if prompt_mode == "searchr1":
                messages = build_messages_searchr1(question, rollout_content)
                raw_input = json.dumps(messages, ensure_ascii=False, indent=2)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_output = resp.choices[0].message.content or ""
                usage = resp.usage
            elif prompt_mode == "searchr1_multiturn":
                if not accumulated_prompt:
                    accumulated_prompt = build_initial_prompt_searchr1_multiturn(question)
                raw_input = accumulated_prompt
                resp = client.completions.create(
                    model=model,
                    prompt=accumulated_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[_IM_END, "<|endoftext|>"],
                )
                raw_output = resp.choices[0].text or ""
                usage = resp.usage
            else:
                raw_input = build_raw_prompt_qwen3_tool(
                    question, qwen3_asst_responses, qwen3_tool_responses,
                )
                resp = client.completions.create(
                    model=model,
                    prompt=raw_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[_IM_END, "<|endoftext|>"],
                )
                raw_output = resp.choices[0].text or ""
                usage = resp.usage

            if usage:
                total_tokens_in += usage.prompt_tokens
                total_tokens_out += usage.completion_tokens

            valid_resp = postprocess_response(raw_output)
            rollout_content += valid_resp
            action, action_content = extract_action(valid_resp, prompt_mode)

            turn_record: Dict[str, Any] = {
                "turn": turn_id,
                "raw_input": raw_input,
                "raw_output": raw_output,
                "postprocessed_output": valid_resp,
                "action": action,
                "action_content": action_content,
                "latency_s": round(time.time() - t_turn, 3),
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
            }

            # --- Handle action ---
            if action == "answer":
                finished = True
                if prompt_mode == "qwen3_tool":
                    qwen3_asst_responses.append(valid_resp)
                elif prompt_mode == "searchr1_multiturn":
                    accumulated_prompt += raw_output
            elif action == "search":
                search_queries.append(action_content)
                t_ret = time.time()
                search_result = retrieve_doc(action_content, retrieval_url=retrieval_url)
                turn_record["retrieval_latency_s"] = round(time.time() - t_ret, 3)
                turn_record["search_result"] = search_result

                if prompt_mode == "searchr1":
                    rollout_content += f"\n\n<information>{search_result}</information>\n\n"
                elif prompt_mode == "searchr1_multiturn":
                    info_text = f"\n\n<information>{search_result}</information>\n\n"
                    accumulated_prompt += raw_output + _IM_END + info_text
                    rollout_content += info_text
                else:
                    qwen3_asst_responses.append(valid_resp)
                    qwen3_tool_responses.append(search_result)
                    rollout_content += f"\n\n<tool_response>{search_result}</tool_response>\n\n"
            else:
                # Invalid action
                if prompt_mode == "searchr1":
                    retry_msg = (
                        "\nMy previous action is invalid. If I want to search, I should put "
                        "the query between <search> and </search>. If I want to give the final "
                        "answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                elif prompt_mode == "searchr1_multiturn":
                    retry_msg = (
                        "\nMy previous action is invalid. If I want to search, I should put "
                        "the query between <search> and </search>. If I want to give the final "
                        "answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                    accumulated_prompt += raw_output + _IM_END + retry_msg
                else:
                    retry_msg = (
                        "Your previous response did not contain a valid tool call or answer. "
                        "Use <tool_call> to search or <answer> to give the final answer."
                    )
                    qwen3_asst_responses.append(valid_resp)
                    qwen3_tool_responses.append(retry_msg)
                turn_record["retry_message"] = retry_msg
                rollout_content += "\n" + retry_msg + "\n"

            turns.append(turn_record)

        # --- Force final answer if not finished ---
        if not finished:
            turn_id += 1
            t_turn = time.time()
            if prompt_mode == "searchr1":
                messages = build_messages_searchr1(question, rollout_content)
                raw_input = json.dumps(messages, ensure_ascii=False, indent=2)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_output = resp.choices[0].message.content or ""
                usage = resp.usage
            elif prompt_mode == "searchr1_multiturn":
                raw_input = accumulated_prompt
                resp = client.completions.create(
                    model=model,
                    prompt=accumulated_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[_IM_END, "<|endoftext|>"],
                )
                raw_output = resp.choices[0].text or ""
                usage = resp.usage
            else:
                raw_input = build_raw_prompt_qwen3_tool(
                    question, qwen3_asst_responses, qwen3_tool_responses,
                )
                resp = client.completions.create(
                    model=model,
                    prompt=raw_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[_IM_END, "<|endoftext|>"],
                )
                raw_output = resp.choices[0].text or ""
                usage = resp.usage

            if usage:
                total_tokens_in += usage.prompt_tokens
                total_tokens_out += usage.completion_tokens
            rollout_content += raw_output

            turns.append({
                "turn": turn_id,
                "raw_input": raw_input,
                "raw_output": raw_output,
                "postprocessed_output": raw_output,
                "action": "forced_final",
                "action_content": "",
                "latency_s": round(time.time() - t_turn, 3),
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
            })

    except Exception as e:
        return {
            "id": sample_id,
            "question": question,
            "data_source": data_source,
            "golden_answers": golden_answers,
            "error": str(e),
            "em_score": 0.0,
            "turns": turns,
        }

    em = compute_em_score(rollout_content, golden_answers)
    predicted = extract_solution(rollout_content)

    return {
        "id": sample_id,
        "question": question,
        "data_source": data_source,
        "golden_answers": golden_answers,
        "predicted_answer": predicted,
        "em_score": em,
        "num_turns": turn_id,
        "num_searches": len(search_queries),
        "search_queries": search_queries,
        "total_time_s": round(time.time() - t_start, 3),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "prompt_mode": prompt_mode,
        "full_rollout": rollout_content,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_markdown_report(
    results: List[Dict[str, Any]], path: str, model: str, prompt_mode: str,
) -> None:
    correct = sum(1 for r in results if r.get("em_score", 0) > 0)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SearchR1 Rollout Trace Report\n\n")
        f.write(f"- **Model:** {model}\n")
        f.write(f"- **Prompt Mode:** {prompt_mode}\n")
        f.write(f"- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Samples:** {len(results)}\n")
        f.write(f"- **Overall EM:** {correct}/{len(results)} = {correct/len(results):.2%}\n\n")
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

            for t in r.get("turns", []):
                f.write(f"### Turn {t['turn']} — Action: `{t.get('action', '?')}`\n\n")

                f.write("**Raw Input (sent to model):**\n")
                f.write(f"```\n{t['raw_input']}\n```\n\n")

                f.write("**Raw Output (model response, unprocessed):**\n")
                f.write(f"```\n{t['raw_output']}\n```\n\n")

                if t["raw_output"] != t["postprocessed_output"]:
                    f.write("**Postprocessed Output:**\n")
                    f.write(f"```\n{t['postprocessed_output']}\n```\n\n")

                if t.get("action_content"):
                    f.write(f"**Action Content:** {t['action_content']}\n\n")

                if t.get("search_result"):
                    f.write("**Search Result:**\n")
                    f.write(f"```\n{t['search_result']}\n```\n\n")

                if t.get("retry_message"):
                    f.write(f"**Retry Message:** {t['retry_message']}\n\n")

                latency_parts = [f"{t.get('latency_s', '?')}s"]
                if t.get("retrieval_latency_s"):
                    latency_parts.append(f"retrieval: {t['retrieval_latency_s']}s")
                f.write(f"**Latency:** {' | '.join(latency_parts)} | "
                        f"Tokens: in={t.get('tokens_in', '?')} out={t.get('tokens_out', '?')}\n\n")

            f.write("---\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone rollout tracing for SearchR1 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--endpoint", type=str, required=True,
                        help="vLLM API endpoint (e.g., http://localhost:8001/v1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name as registered in vLLM")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to test parquet file")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to trace (default: 100)")
    parser.add_argument("--prompt-mode", type=str, default=None,
                        choices=["searchr1", "searchr1_multiturn", "qwen3_tool"],
                        help="Prompt format: 'searchr1' (AGL, single user msg), "
                             "'searchr1_multiturn' (RLF, multi-turn chat), "
                             "'qwen3_tool' (legacy RLF). "
                             "Default: auto-detect from --label (rlf -> searchr1_multiturn, else searchr1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per turn (default: 2048). Training has no per-turn limit.")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--retrieval-url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output-dir", type=str, default="rollout_results")
    parser.add_argument("--label", type=str, default="rollout",
                        help="Label prefix for output files (default: rollout)")
    args = parser.parse_args()

    # Auto-detect prompt mode from label
    # AGL uses searchr1 (single user message), RLF uses searchr1_multiturn (proper multi-turn chat).
    if args.prompt_mode is None:
        if args.label == "rlf":
            args.prompt_mode = "searchr1_multiturn"
        else:
            args.prompt_mode = "searchr1"

    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.data_file)
    df = df.head(args.n_samples)
    dataset = df.to_dict(orient="records")
    print(f"Loaded {len(dataset)} samples from {args.data_file}")
    print(f"Prompt mode: {args.prompt_mode}")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "token-abc123")
    client = OpenAI(base_url=args.endpoint, api_key=api_key)

    results: List[Dict[str, Any]] = []
    for i, sample in enumerate(dataset):
        golden = list(sample["golden_answers"])
        sid = sample.get("id", f"sample_{i}")
        dsrc = sample.get("data_source", "unknown")

        record = run_sample_rollout(
            client=client, model=args.model,
            question=sample["question"], golden_answers=golden,
            sample_id=sid, data_source=dsrc,
            temperature=args.temperature, max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            retrieval_url=args.retrieval_url,
            prompt_mode=args.prompt_mode,
        )
        results.append(record)

        em_icon = "\u2705" if record.get("em_score", 0) > 0 else "\u274c"
        running_em = sum(r.get("em_score", 0) for r in results) / len(results)
        print(f"  [{i+1}/{len(dataset)}] {sid} {em_icon} EM={record.get('em_score', 0):.0f}  "
              f"turns={record.get('num_turns', '?')}  searches={record.get('num_searches', '?')}  "
              f"running_EM={running_em:.4f}")

    # --- Summary ---
    valid = [r for r in results if "error" not in r]
    em_scores = [r["em_score"] for r in valid]
    overall_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    print(f"\n{'='*60}")
    print(f"  Final EM: {overall_em:.4f}  ({sum(1 for e in em_scores if e > 0)}/{len(em_scores)})")
    print(f"  Errors: {len(results) - len(valid)}")
    print(f"{'='*60}")

    # --- Save outputs ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = os.path.basename(args.model.rstrip("/"))
    prefix = f"{args.label}_"

    # JSON with full traces
    json_path = os.path.join(args.output_dir, f"{prefix}traces_{model_tag}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Full traces (JSON): {json_path}")

    # Readable markdown report
    md_path = os.path.join(args.output_dir, f"{prefix}report_{model_tag}_{timestamp}.md")
    write_markdown_report(results, md_path, args.model, args.prompt_mode)
    print(f"Markdown report:    {md_path}")


if __name__ == "__main__":
    main()
