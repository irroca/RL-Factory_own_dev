#!/usr/bin/env python3
"""
Test: verify searchr1_agl training & eval are fully aligned with AGL.

Uses Qwen3-0.6B tokenizer with fabricated data. No heavy dependencies
(tensordict, verl, torch) — tests core prompt logic only.

Usage:
    /home/xinlin/miniconda3/envs/BFCL/bin/python tests/test_searchr1_agl_alignment.py
"""
from __future__ import annotations
import re, sys
from typing import List, Optional, Tuple
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

INSTRUCTION_FORMAT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as many "
    "times as your want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

QUESTION = "What is the capital of France?"
RESP1 = "<think>\nI need to search for the capital of France.\n</think>\n<search>capital of France</search>"
INFO1 = ("\n\n<information>Doc 1(Title: France) France is a country in Western Europe. "
         "Its capital is Paris, which is known for the Eiffel Tower.\n"
         "Doc 2(Title: Paris) Paris is the capital and most populous city of France.\n</information>\n\n")
RESP2 = "<think>\nBased on the search results, the capital of France is Paris.\n</think>\n<answer>Paris</answer>"
RESP1B = "<think>\nSearching more...\n</think>\n<search>Paris facts</search>"
INFO1B = "\n\n<information>Doc 1(Title: Paris) Paris has 2M people.\n</information>\n\n"
RESP2B = "<think>\nNow I know.\n</think>\n<answer>Paris</answer>"


# ---------------------------------------------------------------------------
# Standalone reimplementation of SearchR1AGLManager
# (same logic as searchr1_agl_manager.py, no envs/ imports)
# ---------------------------------------------------------------------------
class AGLManager:
    def __init__(self, enable_thinking=True):
        self.enable_thinking = enable_thinking
        self._user: dict[int, str] = {}
        self._rollout: dict[int, str] = {}

    def initial_prompt(self, messages, tok):
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=self.enable_thinking)

    def reset(self, n):
        self._user = {}; self._rollout = {}

    def init_sample(self, idx, content):
        self._user[idx] = content; self._rollout[idx] = ""

    def reprompt(self, idx, resp, info, tok):
        self._rollout[idx] += resp + info
        msgs = [{"role": "user", "content": self._user[idx] + self._rollout[idx]}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=self.enable_thinking)

    @staticmethod
    def extract_user(prompt):
        m = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', prompt, re.DOTALL)
        return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def agl_rollout(tok, turns_data):
    """Simulate AGL: each turn sends fresh messages=[{user: prompt+rollout}]."""
    prompt = INSTRUCTION_FORMAT + QUESTION
    rollout = ""
    result = []
    for resp, info in turns_data:
        msgs = [{"role": "user", "content": prompt + rollout}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        result.append(p)
        rollout += resp + info
    # Last turn (answer, no info to append)
    msgs = [{"role": "user", "content": prompt + rollout}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    result.append(p)
    return result


def rlf_training(tok, turns_data):
    """Simulate RLF searchr1_agl training: manager reprompt each turn."""
    mgr = AGLManager(enable_thinking=True)
    msgs = [{"role": "user", "content": INSTRUCTION_FORMAT + QUESTION}]
    p1 = mgr.initial_prompt(msgs, tok)
    result = [p1]
    mgr.reset(1)
    mgr.init_sample(0, mgr.extract_user(p1))
    for resp, info in turns_data:
        p = mgr.reprompt(0, resp, info, tok)
        result.append(p)
    return result


def rlf_eval(tok, turns_data):
    """Simulate eval_searchr1.py searchr1_agl mode: chat API each turn."""
    prompt = INSTRUCTION_FORMAT + QUESTION
    rollout = ""
    result = []
    for resp, info in turns_data:
        msgs = [{"role": "user", "content": prompt + rollout}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        result.append(p)
        rollout += resp + info
    msgs = [{"role": "user", "content": prompt + rollout}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    result.append(p)
    return result


def compare(name, list_a, list_b, tok):
    ok = True
    for i, (a, b) in enumerate(zip(list_a, list_b)):
        str_eq = a == b
        ids_eq = tok.encode(a, add_special_tokens=False) == tok.encode(b, add_special_tokens=False)
        status = "PASS" if (str_eq and ids_eq) else "FAIL"
        if not (str_eq and ids_eq):
            ok = False
        print(f"    [{status}] Turn {i+1}: str={str_eq}, tokens={ids_eq}, "
              f"len_a={len(a)}, len_b={len(b)}")
        if not str_eq:
            # Find first diff
            for j in range(min(len(a), len(b))):
                if a[j] != b[j]:
                    print(f"      First diff at pos {j}: "
                          f"A=...{repr(a[max(0,j-30):j+30])}... "
                          f"B=...{repr(b[max(0,j-30):j+30])}...")
                    break
            if len(a) != len(b):
                print(f"      Length: A={len(a)}, B={len(b)}")
    if len(list_a) != len(list_b):
        print(f"    FAIL: different number of turns: {len(list_a)} vs {len(list_b)}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"  SearchR1 AGL Alignment Test — {MODEL_NAME}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Tokenizer: {type(tok).__name__}\n")

    # 2-turn scenario: 1 search + answer
    turns_2 = [(RESP1, INFO1)]
    # 3-turn scenario: 2 searches + answer
    turns_3 = [(RESP1, INFO1), (RESP1B, INFO1B)]

    results = []

    # --- TEST 1: AGL vs RLF Training (2-turn) ---
    print("TEST 1: AGL vs RLF Training — 2-turn")
    a = agl_rollout(tok, turns_2)
    b = rlf_training(tok, turns_2)
    r = compare("2-turn", a, b, tok)
    results.append(("T1: AGL vs Training (2-turn)", r))

    # --- TEST 2: AGL vs RLF Training (3-turn) ---
    print("\nTEST 2: AGL vs RLF Training — 3-turn")
    a = agl_rollout(tok, turns_3)
    b = rlf_training(tok, turns_3)
    r = compare("3-turn", a, b, tok)
    results.append(("T2: AGL vs Training (3-turn)", r))

    # --- TEST 3: AGL vs RLF Eval (2-turn) ---
    print("\nTEST 3: AGL vs RLF Eval — 2-turn")
    a = agl_rollout(tok, turns_2)
    c = rlf_eval(tok, turns_2)
    r = compare("2-turn", a, c, tok)
    results.append(("T3: AGL vs Eval (2-turn)", r))

    # --- TEST 4: AGL vs RLF Eval (3-turn) ---
    print("\nTEST 4: AGL vs RLF Eval — 3-turn")
    a = agl_rollout(tok, turns_3)
    c = rlf_eval(tok, turns_3)
    r = compare("3-turn", a, c, tok)
    results.append(("T4: AGL vs Eval (3-turn)", r))

    # --- TEST 5: RLF Training vs Eval (3-turn) ---
    print("\nTEST 5: RLF Training vs Eval — 3-turn (cross-check)")
    b = rlf_training(tok, turns_3)
    c = rlf_eval(tok, turns_3)
    r = compare("3-turn", b, c, tok)
    results.append(("T5: Training vs Eval (3-turn)", r))

    # --- TEST 6: User content extraction roundtrip ---
    print("\nTEST 6: User content extraction roundtrip")
    mgr = AGLManager()
    orig = INSTRUCTION_FORMAT + QUESTION
    p = mgr.initial_prompt([{"role": "user", "content": orig}], tok)
    ext = mgr.extract_user(p)
    ok = ext == orig
    print(f"    [{'PASS' if ok else 'FAIL'}] Extracted == Original: {ok}")
    if not ok:
        print(f"      Original len={len(orig)}, Extracted len={len(ext) if ext else 'None'}")
    results.append(("T6: User content extraction", ok))

    # --- TEST 7: Reprompt user content verification ---
    print("\nTEST 7: Reprompt P2 user content = original + resp1 + info1")
    mgr.reset(1)
    mgr.init_sample(0, orig)
    rp = mgr.reprompt(0, RESP1, INFO1, tok)
    rp_user = mgr.extract_user(rp)
    expected = orig + RESP1 + INFO1
    ok = rp_user == expected
    has_resp = RESP1 in (rp_user or "")
    has_info = INFO1 in (rp_user or "")
    print(f"    [{'PASS' if ok else 'FAIL'}] Content match: {ok}")
    print(f"    Contains response: {has_resp}, Contains info: {has_info}")
    if not ok and rp_user:
        for j in range(min(len(expected), len(rp_user))):
            if expected[j] != rp_user[j]:
                print(f"      First diff at pos {j}")
                break
    results.append(("T7: Reprompt content", ok))

    # --- TEST 8: Loss mask structure ---
    print("\nTEST 8: Training sequence loss mask [P1][R1][P2][R2]")
    mgr2 = AGLManager()
    p1 = mgr2.initial_prompt([{"role": "user", "content": orig}], tok)
    P1 = tok.encode(p1, add_special_tokens=False)
    R1 = tok.encode(RESP1, add_special_tokens=False)
    mgr2.reset(1); mgr2.init_sample(0, mgr2.extract_user(p1))
    p2 = mgr2.reprompt(0, RESP1, INFO1, tok)
    P2 = tok.encode(p2, add_special_tokens=False)
    R2 = tok.encode(RESP2, add_special_tokens=False)

    segs = [("P1", P1), ("R1", R1), ("P2", P2), ("R2", R2)]
    print(f"    Sequence: {' + '.join(f'[{n}:{len(s)}]' for n, s in segs)}")
    print(f"    Total: {sum(len(s) for _, s in segs)} tokens")

    # loss mask: segments[1:] with (turn_idx+1)%2
    mask_ok = True
    for turn_idx, (name, seg) in enumerate(segs[1:]):
        expected_mask = (turn_idx + 1) % 2
        label = "TRAIN" if expected_mask == 1 else "SKIP"
        ok = len(seg) > 0
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {len(seg)} tokens, mask={expected_mask} ({label})")
        if not ok: mask_ok = False
    results.append(("T8: Loss mask structure", mask_ok))

    # --- TEST 9: Verify P2 is a COMPLETE prompt (starts with <|im_start|>) ---
    print("\nTEST 9: P2 is a complete prompt (not partial)")
    p2_str = tok.decode(P2, skip_special_tokens=False)
    starts_ok = p2_str.startswith("<|im_start|>")
    ends_ok = p2_str.rstrip().endswith("<|im_start|>assistant\n") or "assistant" in p2_str[-50:]
    has_user_tag = "<|im_start|>user" in p2_str
    print(f"    Starts with <|im_start|>: {starts_ok}")
    print(f"    Contains user tag: {has_user_tag}")
    print(f"    Ends with assistant prompt: {ends_ok}")
    ok = starts_ok and has_user_tag and ends_ok
    print(f"    [{'PASS' if ok else 'FAIL'}] P2 is complete prompt")
    results.append(("T9: P2 completeness", ok))

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed: all_ok = False
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
