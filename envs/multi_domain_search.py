"""
Multi-Domain Search Environment for RL-Factory.

Extends SearchEnv with domain-aware reward computation for multi-domain QA tasks.
Supports biomedical, financial, and science domains with per-domain EM scoring.
"""

import re
import json
import string
import random
import torch
from .search import SearchEnv, _normalize_answer, _em_check, _extract_solution, _check_alternate_tags


class MultiDomainSearchEnv(SearchEnv):
    """Search environment for multi-domain retrieval tasks.

    Inherits single-domain SearchEnv behavior (EM reward, step reward)
    and adds per-domain score tracking for validation diagnostics.
    reward_mode inherited from SearchEnv: 'agl' (default) or 'multi_dim'.
    """

    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)

    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        if self.reward_mode == 'multi_dim':
            return self._compute_score_multi_dim_domain(data, tokenizer, if_val)
        else:
            return self._compute_score_agl_domain(data, tokenizer, if_val)

    # =========================================================================
    # AGL-aligned reward with per-domain diagnostics
    # =========================================================================
    def _compute_score_agl_domain(self, data, tokenizer, if_val=False):

        def compute_score_em(solution_str, ground_truth, format_score=0.0, score=1.0):
            """Pure EM scoring aligned with AGL qa_em.py compute_score_em."""
            answer = _extract_solution(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1

            if do_print:
                print(f"--------------------------------")
                print(f"Golden answers: {ground_truth['target']}")
                print(f"Extracted answer: {answer}")
                print(f"Solution string: {solution_str[:200]}...")

            if answer is None:
                return 0.0
            else:
                if _em_check(answer, ground_truth['target']):
                    return score
                else:
                    return format_score

        scores = []
        domain_scores = {}  # Track per-domain scores for diagnostics

        for i in range(len(data)):
            data_item = data[i]
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            score = compute_score_em(response_str, ground_truth)
            scores.append([score])

            # Track per-domain scores during validation
            if if_val:
                extra_info = processed_data.get('extra_info', {}) or {}
                domain = extra_info.get('domain', 'unknown')
                if domain not in domain_scores:
                    domain_scores[domain] = []
                domain_scores[domain].append(score)

        # Print per-domain diagnostics during validation
        if if_val and domain_scores:
            self._print_domain_diagnostics(scores, domain_scores)

        return scores

    # =========================================================================
    # Original RLF multi-dimensional reward with per-domain diagnostics
    # =========================================================================
    def _compute_score_multi_dim_domain(self, data, tokenizer, if_val=False):

        def extract_solution_multi_dim(solution_str):
            think_pattern = r'<think>.*?</think>'
            solution_str = re.sub(think_pattern, '', solution_str, flags=re.DOTALL)
            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.finditer(answer_pattern, solution_str, re.DOTALL)
            matches = list(match)
            if len(matches) <= 0:
                return None
            return matches[-1].group(1).strip()

        def compute_score_em(solution_str, ground_truth, format_score=0.0, score=1.0):
            answer = extract_solution_multi_dim(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1

            if do_print:
                print(f"--------------------------------")
                print(f"[multi_dim] Golden answers: {ground_truth['target']}")
                print(f"[multi_dim] Extracted answer: {answer}")
                print(f"[multi_dim] Solution string: {solution_str[:200]}...")

            answer_format_score = format_score if _check_alternate_tags(solution_str, r"</?answer>") else (-1 * format_score)
            num_score = 0
            if _check_alternate_tags(solution_str, r"</?tool_call>"):
                tool_call_format_score = format_score
                pattern = r"<tool_call>(.*?)</tool_call>"
                matches = re.findall(pattern, solution_str, re.DOTALL)
                if len(matches) == 0:
                    tool_call_format_score = -1 * format_score
                else:
                    success_num, fail_num = 0, 0
                    for idx, content in enumerate(matches):
                        content_stripped = content.strip()
                        try:
                            parsed = json.loads(content_stripped)
                            success_num += 1
                        except json.JSONDecodeError:
                            fail_num += 1
                    tool_call_format_score = 2 * format_score * success_num / (success_num + fail_num) - format_score
                    if success_num + fail_num > 2:
                        tool_call_format_score -= 0.5 * format_score
                        num_score = -format_score
            else:
                tool_call_format_score = -0.5 * format_score

            total_format_score = answer_format_score + num_score

            if answer is None:
                return -1 * format_score + 0.5 * total_format_score
            else:
                if _em_check(answer, ground_truth['target']):
                    return score + 0.5 * total_format_score
                else:
                    return total_format_score

        format_score = 0.0 if if_val else 0.1
        scores = []
        domain_scores = {}

        for i in range(len(data)):
            data_item = data[i]
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            score = compute_score_em(response_str, ground_truth, format_score=format_score)
            scores.append([score])

            if if_val:
                extra_info = processed_data.get('extra_info', {}) or {}
                domain = extra_info.get('domain', 'unknown')
                if domain not in domain_scores:
                    domain_scores[domain] = []
                domain_scores[domain].append(score)

        if if_val and domain_scores:
            self._print_domain_diagnostics(scores, domain_scores)

        return scores

    @staticmethod
    def _print_domain_diagnostics(scores, domain_scores):
        print("=" * 50)
        print("Multi-Domain Validation Scores:")
        for domain, d_scores in sorted(domain_scores.items()):
            avg = sum(d_scores) / len(d_scores) if d_scores else 0.0
            print(f"  {domain}: {avg:.4f} ({len(d_scores)} samples)")
        all_scores = [s[0] for s in scores]
        print(f"  Overall: {sum(all_scores)/len(all_scores):.4f} ({len(all_scores)} samples)")
        print("=" * 50)
