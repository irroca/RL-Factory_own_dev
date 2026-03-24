# SearchR1 评测脚本使用指南

## 脚本概览

| 脚本 | 用途 | 输出 |
|------|------|------|
| `eval_searchr1.py` | 全量评测，计算 EM 准确率等聚合指标 | metrics JSON + per-sample JSONL |
| `rollout_searchr1.py` | 小样本详细 trace，记录每一轮的完整原始输入输出 | 全量 trace JSON + 可读 Markdown 报告 |

---

## 前置条件

```bash
# 1. 合并 FSDP checkpoint（如需要）
python -m verl.model_merger merge --backend fsdp \
    --local_dir <checkpoint>/actor --target_dir <checkpoint>_hf

# 2. 启动 vLLM 服务
vllm serve <checkpoint>_hf --port 8001

# 3. 启动检索服务
cd agent-lightning/contrib/recipes/search_r1 && bash retrieval_launch.sh
```

---

## eval_searchr1.py 用法

### 评测 AGL 模型

```bash
python eval_searchr1.py \
    --endpoint http://localhost:8001/v1 \
    --model <checkpoint>_hf \
    --data-file data/test.parquet \
    --n-samples 500 \
    --label agl \
    --output-dir eval_results
```

### 评测 RLF 模型

```bash
python eval_searchr1.py \
    --endpoint http://localhost:8001/v1 \
    --model <checkpoint>_hf \
    --data-file data/test.parquet \
    --n-samples 500 \
    --label rlf \
    --output-dir eval_results
```

`--label rlf` 会自动选择 `--prompt-mode searchr1_multiturn`。

### 主要参数

| 参数 | 说明 |
|------|------|
| `--endpoint` | vLLM API 地址，如 `http://localhost:8001/v1` |
| `--model` | vLLM 中注册的模型名（需与 `vllm serve` 的路径一致） |
| `--label` | 输出文件前缀，`agl` 或 `rlf`。`rlf` 会自动选择 `searchr1_multiturn` 模式 |
| `--prompt-mode` | 手动指定 prompt 格式：`searchr1` / `searchr1_multiturn` / `qwen3_tool` |
| `--n-samples` | 评测样本数（默认全部） |
| `--max-turns` | 最大搜索轮数（默认 4） |
| `--rollout-file` | 可选，指定小样本文件用于生成详细 rollout 报告 |

### 输出文件

- `{label}_metrics_{model}_{timestamp}.json` — 聚合指标（EM、搜索率、延迟等）
- `{label}_details_{model}_{timestamp}.jsonl` — 每个样本的详细结果

---

## rollout_searchr1.py 用法

用于小样本（如 100 个）的详细 trace 记录，完整保存每一轮发给模型的原始输入和模型的原始输出。

### 记录 AGL 模型 rollout

```bash
python rollout_searchr1.py \
    --endpoint http://localhost:8001/v1 \
    --model <checkpoint>_hf \
    --data-file data/test.parquet \
    --n-samples 100 \
    --prompt-mode searchr1 \
    --output-dir rollout_results
```

### 记录 RLF 模型 rollout

```bash
python rollout_searchr1.py \
    --endpoint http://localhost:8001/v1 \
    --model <checkpoint>_hf \
    --data-file data/test.parquet \
    --n-samples 100 \
    --prompt-mode searchr1_multiturn \
    --output-dir rollout_results
```

### 输出文件

- `{label}_traces_{model}_{timestamp}.json` — 结构化 JSON，包含每轮的 `raw_input`（发送给模型的完整内容）和 `raw_output`（模型原始返回）
- `{label}_report_{model}_{timestamp}.md` — 可读的 Markdown 报告，逐轮展示输入输出

---

## 三种 Prompt Mode 详解

### 为什么需要不同的 prompt mode

两个框架（AGL 和 RLF）都通过 **OpenAI Chat Completions API** 调用 vLLM 后端。vLLM 收到请求后，会用模型的 **chat template**（如 Qwen3 的 ChatML 格式）将 messages 列表转换为带 `<|im_start|>` / `<|im_end|>` 特殊 token 的文本序列，再送入模型推理。

**vLLM 之后的处理逻辑完全一致**，区别仅在于前端构造 messages 列表的方式不同。评测时必须用和训练时一样的 messages 结构，否则 chat template 生成的 token 序列不是模型训练时见过的分布，会导致准确率大幅下降。

---

### `searchr1`（AGL 格式）

**对应 AGL 训练方式。** 所有内容拼接在一条 user 消息中，不使用多轮角色切换。

#### 发送给 API 的 messages

第一轮：
```json
[{"role": "user", "content": "Answer the given question... Question: What is X?"}]
```

第二轮（搜索后）：
```json
[{"role": "user", "content": "Answer the given question... Question: What is X?<search>query</search>\n\n<information>Doc 1...</information>\n\n"}]
```

#### vLLM chat template 转换后模型看到的 token 序列

```
<|im_start|>user
Answer the given question... Question: What is X?<search>query</search>

<information>Doc 1...</information>

<|im_end|>
<|im_start|>assistant
```

**特点：**
- 无 system 消息
- 所有历史（模型回复、搜索结果）都拼接在同一条 user 消息里
- 模型看到的是一段连续文本，没有角色分隔符

---

### `searchr1_multiturn`（RLF 格式）

**对应 RLF chat_scheduler 训练方式。** 使用正规的多轮对话结构，每轮有明确的角色分隔。

#### 发送给 API 的 messages

第一轮：
```json
[
  {"role": "system", "content": ""},
  {"role": "user", "content": "Answer the given question... Question: What is X?"}
]
```

第二轮（搜索后）：
```json
[
  {"role": "system", "content": ""},
  {"role": "user", "content": "Answer the given question... Question: What is X?"},
  {"role": "assistant", "content": "<think>reasoning...</think>\n<search>query</search>"},
  {"role": "user", "content": "\n\n<information>Doc 1...</information>\n\n"}
]
```

#### vLLM chat template 转换后模型看到的 token 序列

```
<|im_start|>system
<|im_end|>
<|im_start|>user
Answer the given question... Question: What is X?<|im_end|>
<|im_start|>assistant
<search>query</search><|im_end|>
<|im_start|>user

<information>Doc 1...</information>

<|im_end|>
<|im_start|>assistant
```

**特点：**
- 有 system 消息但**内容为空**（与 RLF 训练一致，chat_scheduler 对空 system content 调用 `apply_chat_template` 产生的就是空内容）
- 模型回复和搜索结果分别是独立的 assistant / user 消息
- 中间 assistant 消息中的 `<think>` 标签会被 Qwen3 chat template **自动剥离**（`enable_thinking=True` 时的行为）
- 通过 `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` 传递给 vLLM，与训练时一致

---

### `qwen3_tool`（旧版 RLF 格式）

**对应旧版 RLF 使用 tool_agent_loop 训练方式。** 使用 Qwen3 的 tool-call 格式，手工拼接 raw prompt 绕过 chat template。

#### 直接发送给 vLLM 的原始 prompt（不经过 chat template）

```
<|im_start|>system
# Tools
You may call one or more functions...
<tools>{"name": "search-query_rag", ...}</tools>
...<|im_end|>
<|im_start|>user
Answer the given question... Question: What is X?<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>{"name": "search-query_rag", "arguments": {"query": "..."}}</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1...
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

```

**特点：**
- 使用 Completions API（非 Chat Completions API）直接发送原始 prompt 字符串
- system 消息中包含完整的 JSON tool schema 定义
- 每个 assistant 回复前都有空的 `<think>\n\n</think>\n\n` 块（对应训练时 `enable_thinking=False` 注入的空 think block）
- 使用 `<tool_call>` / `<tool_response>` 代替 `<search>` / `<information>`

---

## 格式对比总结

| 维度 | `searchr1` (AGL) | `searchr1_multiturn` (RLF) | `qwen3_tool` (旧版 RLF) |
|------|---|---|---|
| API | Chat Completions | Chat Completions | Completions (raw prompt) |
| System 消息 | 无 | 有，内容为空 `""` | 有，包含 tool schema |
| 多轮结构 | 单条 user 消息拼接全部历史 | 多轮 system/user/assistant 交替 | 手工拼接多轮 ChatML tokens |
| 搜索指令 | `<search>query</search>` | `<search>query</search>` | `<tool_call>{"name":...}</tool_call>` |
| 搜索结果 | `<information>...</information>` | `<information>...</information>` | `<tool_response>...</tool_response>` |
| enable_thinking | 不传 | True | N/A（手工注入空 think block） |
| think 标签处理 | 保留在拼接文本中 | 被 chat template 自动剥离 | 手工注入空 think block |
