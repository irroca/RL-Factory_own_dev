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
from .search import SearchEnv


class MultiDomainSearchEnv(SearchEnv):
    """Search environment for multi-domain retrieval tasks.

    Inherits single-domain SearchEnv behavior (EM reward, step reward)
    and adds per-domain score tracking for validation diagnostics.
    """

    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)

    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        def normalize_answer(s):
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def em_check(prediction, golden_answers):
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]
            normalized_prediction = normalize_answer(prediction)
            score = 0
            for golden_answer in golden_answers:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1
                    break
            return score

        def extract_solution(solution_str):
            """Extract the last <answer>...</answer> span from the solution string."""
            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.finditer(answer_pattern, solution_str, re.DOTALL)
            matches = list(match)
            if len(matches) == 0:
                return None
            return matches[-1].group(1).strip()

        def compute_score_em(solution_str, ground_truth, format_score=0.0, score=1.0):
            """Pure EM scoring aligned with AGL qa_em.py compute_score_em."""
            answer = extract_solution(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1

            if do_print:
                print(f"--------------------------------")
                print(f"Golden answers: {ground_truth['target']}")
                print(f"Extracted answer: {answer}")
                print(f"Solution string: {solution_str[:200]}...")

            if answer is None:
                return 0.0
            else:
                if em_check(answer, ground_truth['target']):
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
            print("=" * 50)
            print("Multi-Domain Validation Scores:")
            for domain, d_scores in sorted(domain_scores.items()):
                avg = sum(d_scores) / len(d_scores) if d_scores else 0.0
                print(f"  {domain}: {avg:.4f} ({len(d_scores)} samples)")
            all_scores = [s[0] for s in scores]
            print(f"  Overall: {sum(all_scores)/len(all_scores):.4f} ({len(all_scores)} samples)")
            print("=" * 50)

        return scores
