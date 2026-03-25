"""
Preprocess multi-domain QA datasets to parquet format for RL-Factory training.

Supports loading QA data from HuggingFace or local JSONL files for each domain
(biomedical, financial, science) and converts them to the unified parquet format
expected by RL-Factory's RLHFDataset.

Usage:
    # From HuggingFace datasets (BioASQ, FinQA, SciQ)
    python scripts/multi_domain_search_data.py --local_dir ./data/multi_domain_search

    # From local JSONL files (each line: {"question": "...", "golden_answers": [...], "domain": "..."})
    python scripts/multi_domain_search_data.py \
        --local_dir ./data/multi_domain_search \
        --from_local \
        --train_files domain1_train.jsonl domain2_train.jsonl \
        --test_files domain1_test.jsonl domain2_test.jsonl

Expected JSONL format per line:
    {"question": "What is type 2 diabetes?", "golden_answers": ["A metabolic disorder ..."], "domain": "biomedical"}
"""

import os
import json
import argparse

import datasets


# Available domain names (align with multi-domain retriever server)
DOMAINS = ["biomedical", "financial", "science"]

MULTI_DOMAIN_SEARCH_INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> with the target "
    "domain specified by <domain> domain_name </domain>. Available domains: {domains}. "
    "The search engine will return the top results between <information> and </information>. "
    "You can search as many times as you want, and you may search across different domains. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, to search for information: "
    "<search> What is insulin resistance? </search><domain> biomedical </domain>. "
    "To provide a final answer: <answer> Beijing </answer>. Question: "
)


def make_prefix(question, domains_str):
    """Build the instruction prefix for a question."""
    question = question.strip()
    if question and question[-1] not in '?？':
        question += '?'
    instruction = MULTI_DOMAIN_SEARCH_INSTRUCTION.format(domains=domains_str)
    return f"{instruction}{question}\n"


def build_record(question, golden_answers, domain, data_source, split, idx, domains_str):
    """Build a single training record in RL-Factory format."""
    prefix = make_prefix(question, domains_str)
    solution = {"target": golden_answers}

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prefix}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "domain": domain,
        },
    }


def load_local_jsonl(file_path):
    """Load records from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_local_files(file_list, split, domains_str):
    """Process local JSONL files into RL-Factory format."""
    all_records = []
    idx = 0
    for fpath in file_list:
        raw = load_local_jsonl(fpath)
        for item in raw:
            question = item['question']
            golden_answers = item['golden_answers']
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]
            domain = item.get('domain', 'unknown')
            data_source = f"multi_domain_{domain}"
            record = build_record(question, golden_answers, domain, data_source, split, idx, domains_str)
            all_records.append(record)
            idx += 1
    return all_records


def process_hf_datasets(domains_str):
    """Load and process QA datasets from HuggingFace for each domain."""
    train_records = []
    test_records = []
    train_idx = 0
    test_idx = 0

    # --- Biomedical: Use BioASQ-style data from FlashRAG if available, else NQ subset ---
    try:
        print("Loading biomedical QA data (PubMedQA)...")
        ds = datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
        for split_name, target_list, counter_name in [("train", train_records, "train_idx")]:
            if split_name in ds:
                for item in ds[split_name]:
                    question = item.get("question", "")
                    answer = item.get("long_answer", item.get("final_decision", ""))
                    if not question or not answer:
                        continue
                    record = build_record(
                        question, [answer], "biomedical", "multi_domain_biomedical",
                        split_name, eval(counter_name), domains_str
                    )
                    target_list.append(record)
                    if counter_name == "train_idx":
                        train_idx += 1
                    else:
                        test_idx += 1
        print(f"  Biomedical: {train_idx} train records loaded")
    except Exception as e:
        print(f"  Warning: Could not load biomedical data: {e}")

    # --- Science: Use SciQ ---
    try:
        print("Loading science QA data (SciQ)...")
        ds = datasets.load_dataset("allenai/sciq", trust_remote_code=True)
        for split_name in ["train", "test"]:
            if split_name not in ds:
                continue
            target = train_records if split_name == "train" else test_records
            for item in ds[split_name]:
                question = item.get("question", "")
                answer = item.get("correct_answer", "")
                if not question or not answer:
                    continue
                cur_idx = train_idx if split_name == "train" else test_idx
                record = build_record(
                    question, [answer], "science", "multi_domain_science",
                    split_name, cur_idx, domains_str
                )
                target.append(record)
                if split_name == "train":
                    train_idx += 1
                else:
                    test_idx += 1
        print(f"  Science: added to train/test")
    except Exception as e:
        print(f"  Warning: Could not load science data: {e}")

    return train_records, test_records


def main():
    parser = argparse.ArgumentParser(description="Preprocess multi-domain QA data for RL-Factory")
    parser.add_argument('--local_dir', default='./data/multi_domain_search',
                        help='Output directory for parquet files')
    parser.add_argument('--from_local', action='store_true',
                        help='Load from local JSONL files instead of HuggingFace')
    parser.add_argument('--train_files', nargs='+', default=[],
                        help='Local JSONL files for training data')
    parser.add_argument('--test_files', nargs='+', default=[],
                        help='Local JSONL files for test data')
    parser.add_argument('--domains', nargs='+', default=DOMAINS,
                        help='Domain names to include')

    args = parser.parse_args()
    domains_str = ", ".join(args.domains)

    if args.from_local:
        print(f"Loading from local files...")
        train_records = process_local_files(args.train_files, 'train', domains_str)
        test_records = process_local_files(args.test_files, 'test', domains_str)
    else:
        print("Loading from HuggingFace datasets...")
        train_records, test_records = process_hf_datasets(domains_str)

    os.makedirs(args.local_dir, exist_ok=True)

    if train_records:
        train_ds = datasets.Dataset.from_list(train_records)
        train_path = os.path.join(args.local_dir, 'train.parquet')
        train_ds.to_parquet(train_path)
        print(f"Saved {len(train_records)} train records to {train_path}")

    if test_records:
        test_ds = datasets.Dataset.from_list(test_records)
        test_path = os.path.join(args.local_dir, 'test.parquet')
        test_ds.to_parquet(test_path)
        print(f"Saved {len(test_records)} test records to {test_path}")

    print("Done!")


if __name__ == '__main__':
    main()
