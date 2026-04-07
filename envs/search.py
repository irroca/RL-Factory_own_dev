import re
import json
import string
import random
import torch
from .base import Env


def _normalize_answer(s):
    """Lowercase, remove punctuation/articles, and normalize whitespace."""
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


def _em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = _normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = _normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def _extract_solution(solution_str):
    """Extract the last <answer>...</answer> span from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def _check_alternate_tags(text, tag_pattern):
    """Check whether XML tags of the given pattern alternate properly (no nesting)."""
    found = re.findall(tag_pattern, text)
    if not found:
        return False
    match = re.match(r"<\/?(\w+)>", found[0])
    if not match:
        return False
    tagname = match.group(1)
    open_tag = f"<{tagname}>"
    close_tag = f"</{tagname}>"

    tags = re.findall(tag_pattern, text)
    stack = []
    for tag in tags:
        if tag == open_tag:
            if stack:
                return False
            stack.append(tag)
        elif tag == close_tag:
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0


class SearchEnv(Env):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False
        # reward_mode: 'agl' (pure EM, default) or 'multi_dim' (original RLF multi-dimensional)
        self.reward_mode = getattr(config, 'reward_mode', 'agl')

    def get_step_reward(self, responses, format_score=0.1):
        step_reward = []
    
        for response in responses:
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)
            if temp_action == 'answer':
                step_reward.append(torch.nan)
            else:
                if temp_tool_list[0]['name'] == '<empty>':
                    step_reward.append(-0.5 * format_score)
                else:
                    fail_number = 0
                    for i in range(len(temp_tool_list )):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                    step_rew = ((len(temp_tool_list) - 2 *fail_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)
       

        return step_reward

    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        if self.reward_mode == 'multi_dim':
            return self._compute_score_multi_dim(data, tokenizer, if_val)
        elif self.reward_mode == 'format_reward':
            return self._compute_score_format_reward(data, tokenizer, if_val)
        else:
            return self._compute_score_agl(data, tokenizer, if_val)

    # =========================================================================
    # AGL-aligned reward: pure Exact Match, no format reward.
    # =========================================================================
    def _compute_score_agl(self, data, tokenizer, if_val=False):

        def compute_score_em(solution_str, ground_truth, format_score=0.0, score=1.0):
            """Pure EM scoring aligned with AGL qa_em.py compute_score_em."""
            answer = _extract_solution(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1

            if do_print:
                print(f"--------------------------------")
                print(f"Golden answers: {ground_truth['target']}")
                print(f"Extracted answer: {answer}")
                print(f"Solution string: {solution_str}")

            if answer is None:
                return 0.0
            else:
                if _em_check(answer, ground_truth['target']):
                    return score
                else:
                    return format_score

        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']
            score = compute_score_em(response_str, ground_truth)
            scores.append([score])

        return scores

    # =========================================================================
    # Format Reward: 4-way reward (Empirical Study, arXiv:2505.15117)
    #   correct + good format  → 1.0
    #   correct + bad format   → 1.0 - λ_f
    #   wrong   + good format  → max(λ_f, 0.1)   (floor 0.1, DynaSearcher)
    #   wrong   + bad format   → 0.0
    # =========================================================================
    def _compute_score_format_reward(self, data, tokenizer, if_val=False):
        lambda_f = 0.2

        def _check_format(solution_str):
            """Check whether the response has valid tag structure.

            Validates:
            - <answer> tags alternate properly (required)
            - <search> tags alternate properly (if present)
            - <think> tags alternate properly (if present)
            """
            answer_ok = _check_alternate_tags(solution_str, r"</?answer>")
            if not answer_ok:
                return False
            search_tags = re.findall(r'</?search>', solution_str)
            if search_tags:
                if not _check_alternate_tags(solution_str, r"</?search>"):
                    return False
            think_tags = re.findall(r'</?think>', solution_str)
            if think_tags:
                if not _check_alternate_tags(solution_str, r"</?think>"):
                    return False
            return True

        def compute_score(solution_str, ground_truth):
            answer = _extract_solution(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1

            if do_print:
                print(f"--------------------------------")
                print(f"[format_reward] Golden answers: {ground_truth['target']}")
                print(f"[format_reward] Extracted answer: {answer}")
                print(f"[format_reward] Solution string: {solution_str[:200]}...")

            format_ok = _check_format(solution_str)
            em_correct = answer is not None and _em_check(answer, ground_truth['target'])

            if em_correct and format_ok:
                return 1.0
            elif em_correct and not format_ok:
                return 1.0 - lambda_f
            elif not em_correct and format_ok:
                return max(lambda_f, 0.1)
            else:
                return 0.0

        scores = []
        for i in range(len(data)):
            data_item = data[i]
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']
            score = compute_score(response_str, ground_truth)
            scores.append([score])

        return scores

    # =========================================================================
    # Original RL-Factory multi-dimensional reward:
    #   EM + format reward (answer tag validity, tool_call JSON validity, etc.)
    # =========================================================================
    def _compute_score_multi_dim(self, data, tokenizer, if_val=False):

        def extract_solution_multi_dim(solution_str):
            """Extract answer after stripping <think> blocks (original RLF behavior)."""
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
                print(f"[multi_dim] Solution string: {solution_str}")

            answer_format_score = format_score if _check_alternate_tags(solution_str, r"</?answer>") else (-1 * format_score)

            # Check <search> tag format (migrated from <tool_call>)
            search_format_score = 0
            num_score = 0
            search_tags = re.findall(r'<search>.*?</search>', solution_str, re.DOTALL)
            if search_tags:
                if _check_alternate_tags(solution_str, r"</?search>"):
                    search_format_score = format_score
                else:
                    search_format_score = -0.5 * format_score
                if len(search_tags) > 2:
                    search_format_score -= 0.5 * format_score
                    num_score = -format_score
            # else: no search tags — model answered directly, no penalty

            total_format_score = answer_format_score + search_format_score + num_score

            if answer is None:
                return -1 * format_score + 0.5 * total_format_score
            else:
                if _em_check(answer, ground_truth['target']):
                    return score + 0.5 * total_format_score
                else:
                    return total_format_score

        format_score = 0.0 if if_val else 0.1
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

            score = compute_score_em(response_str, ground_truth, format_score=format_score)
            scores.append([score])

        return scores
