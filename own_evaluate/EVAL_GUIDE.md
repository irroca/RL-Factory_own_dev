# SearchR1 评测脚本使用指南

## 脚本概览

| 脚本 | 用途 | 输出 |
|------|------|------|
| `eval_searchr1.py` | 全量评测，计算 EM 准确率等聚合指标 | metrics JSON + per-sample JSONL |
| `rollout_searchr1.py` | 小样本详细 trace，记录每一轮的完整原始输入输出 | 全量 trace JSON + 可读 Markdown 报告 |
| `simulate_rlf_training_io.py` | 模拟 RLF 训练时的多 turn token 拼接逻辑（不需要 vLLM） | 终端输出每步的 decoded token 序列 |

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

`--label rlf` 会自动选择 `--prompt-mode searchr1`。

### 主要参数

| 参数 | 说明 |
|------|------|
| `--endpoint` | vLLM API 地址，如 `http://localhost:8001/v1` |
| `--model` | vLLM 中注册的模型名（需与 `vllm serve` 的路径一致） |
| `--label` | 输出文件前缀。`rlf` 自动选择 `searchr1` 模式，`rlf_multistep` 自动选择 `searchr1_multistep` 模式，其他默认 `agl` |
| `--prompt-mode` | 手动指定 prompt 格式：`agl` / `searchr1` / `searchr1_multistep` / `qwen3_tool` |
| `--n-samples` | 评测样本数（默认全部） |
| `--max-turns` | 最大搜索轮数（默认 4） |
| `--temperature` | 生成温度（默认 0.0，即贪心解码）。训练时为 0.7 |
| `--top-p` | Top-p 采样参数（默认 1.0）。训练时 RLF 为 0.95，AGL 为 1.0 |
| `--use-train-stops` | 添加 `</search>` 和 `</answer>` 作为 vLLM stop strings 并启用 `include_stop_str_in_output=True`，与训练时 addstopstring 配置对齐。不加此 flag 时，模型可能在一轮内生成完整推理链，与训练行为不一致 |
| `--rollout-file` | 可选，指定小样本文件用于生成详细 rollout 报告 |

### 评测超参数对齐说明

训练和评测的采样参数需保持一致以确保训推对齐。不同评测模式下的默认参数如下：

| 参数 | 训练 (RLF) | 训练 (AGL) | 标准评测（默认） | 训推对齐评测 |
|------|-----------|-----------|----------------|------------|
| temperature | 0.7 | 0.7 | **0.0** | 0.0 |
| top_p | 0.95 | 1.0 | **1.0** | 1.0 |
| stop strings | `["<\|im_end\|>","</search>","</answer>"]` | N/A（chat API） | `["<\|im_end\|>"]` | **`["<\|im_end\|>","</search>","</answer>"]`** |
| include_stop_str_in_output | True | N/A | False | **True** |

> **注意**：`--use-train-stops` 是训推对齐的关键参数。没有此 flag 时，vLLM 不会在 `</search>` 处停止生成，模型会在一轮内自行续写搜索结果（自问自答），environment 无法注入真实检索结果。addstopstring_dosample 训练的模型**必须**使用此 flag 评测才能体现多轮搜索行为。

### 评测 addstopstring_dosample 训练的模型

```bash
# 必须加 --use-train-stops 以对齐训练时的 stop strings 行为
python eval_searchr1.py \
    --endpoint http://localhost:8001/v1 \
    --model <checkpoint>_hf \
    --data-file data/test.parquet \
    --n-samples 5000 \
    --label rlf_pure_em \
    --prompt-mode searchr1 \
    --use-train-stops \
    --output-dir eval_results
```

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
    --prompt-mode agl \
    --output-dir rollout_results
```

### 记录 RLF 模型 rollout

```bash
python rollout_searchr1.py \
    --endpoint http://localhost:8001/v1 \
    --model <checkpoint>_hf \
    --data-file data/test.parquet \
    --n-samples 100 \
    --prompt-mode searchr1 \
    --output-dir rollout_results
```

### 输出文件

- `{label}_traces_{model}_{timestamp}.json` — 结构化 JSON，包含每轮的 `raw_input`（发送给模型的完整内容）和 `raw_output`（模型原始返回）
- `{label}_report_{model}_{timestamp}.md` — 可读的 Markdown 报告，逐轮展示输入输出

---

## 三种 Prompt Mode 详解

### 为什么需要不同的 prompt mode

两个框架（AGL 和 RLF）在训练时构造模型输入的方式不同：

- **AGL**：通过 **OpenAI Chat Completions API** 调用 vLLM，将所有历史拼接在一条 user 消息内，每轮由 vLLM 对 messages 重新 apply chat template
- **RLF**：在 **token 层面直接拼接**——首轮对初始 messages apply 一次 chat template，后续轮次的环境反馈作为裸文本 tokenize 后直接拼接到 token 序列末尾，**不**重新 apply chat template

评测时必须用和训练时一致的输入构造方式，否则模型看到的 token 序列分布不匹配，会导致准确率大幅下降。

- `agl`：使用 Chat Completions API，匹配 AGL 训练
- `searchr1`：使用 Completions API + 手工 prompt 拼接，匹配 RLF `searchr1` 训练
- `searchr1_multistep`：使用 Completions API + user→assistant turn 交替，匹配 RLF `searchr1_multistep` 训练
- `qwen3_tool`：使用 Completions API + 手工 ChatML prompt，匹配旧版 RLF tool-call 训练

---

### `agl`（AGL 格式）

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

### `searchr1`（RLF 格式）

**对应 RLF `generate_sequences_loop` + `ToolUtils` + `tool_manager=searchr1` 训练方式。** 使用 Completions API 发送手工拼接的原始 prompt 字符串，严格复现训练时 token 级拼接逻辑。

#### 工作原理

RLF 训练时的多 turn 交互**不是**标准的多轮对话。它在 token 层面工作：

1. **首轮**：对初始 messages 调用一次 `apply_chat_template(enable_thinking=True)` 得到带完整角色标记的 prompt
2. **后续每轮**：模型 response tokens（自然含 `<|im_end|>`）+ 环境反馈 tokens（直接 tokenize 原始文本，**不** apply chat template）直接拼接到序列末尾
3. **vLLM** 每轮接收的是完整的拼接 token 序列作为 `prompt_token_ids`

为了在评测时复现这个逻辑，eval 脚本：
- 使用 **Completions API**（`client.completions.create(prompt=...)`）而非 Chat Completions API
- 首轮 prompt 手工构造，等价于 `apply_chat_template` 的输出
- 后续轮次通过字符串拼接 `accumulated_prompt += raw_output + "<|im_end|>" + info_text` 模拟 token 拼接
- 使用 `stop=["<|im_end|>", "<|endoftext|>"]` 控制生成停止

#### 首轮 prompt（等价于 apply_chat_template 输出）

```
<|im_start|>system
<|im_end|>
<|im_start|>user
Answer the given question... Question: What is X?<|im_end|>
<|im_start|>assistant
<think>
```

#### 第二轮 prompt（搜索后，token 级拼接）

```
<|im_start|>system
<|im_end|>
<|im_start|>user
Answer the given question... Question: What is X?<|im_end|>
<|im_start|>assistant
<think>
<think>                                          ← vLLM 生成的 response 开头
I need to find... Let me search.
</think>

<search>query</search><|im_end|>                ← response 自然结束

<information>Doc 1(Title: ...) text...           ← 环境反馈，直接拼接，无角色标记
Doc 2(Title: ...) text...
</information>

```

**注意**：环境反馈 `<information>` 块前后**没有** `<|im_start|>user`、`<|im_end|>`、`<|im_start|>assistant` 等角色标记。模型从 `</information>\n\n` 后直接续写，这与训练时的行为完全一致。

#### 完整 3 轮交互的 token 序列

```
[PROMPT: apply_chat_template 输出，有完整角色标记]
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{instruction+question}<|im_end|>\n<|im_start|>assistant\n<think>\n

[RESPONSE 1: vLLM 生成，loss_mask=1]
<think>\nI need to find...\n</think>\n\n<search>query1</search><|im_end|>

[INFO 1: 环境反馈，裸文本 tokenize，loss_mask=0]
\n\n<information>Doc 1... Doc 3\n</information>\n\n

[RESPONSE 2: vLLM 生成，loss_mask=1]
<think>\nBased on search results...\n</think>\n\n<search>query2</search><|im_end|>

[INFO 2: 环境反馈，裸文本 tokenize，loss_mask=0]
\n\n<information>Doc 1... Doc 3\n</information>\n\n

[RESPONSE 3: vLLM 生成，loss_mask=1]
<think>\nI have confirmed...\n</think>\n\n<answer>Paris</answer><|im_end|>
```

**特点：**
- 使用 Completions API 直接发送原始 prompt 字符串
- 只有首轮 prompt 有完整的角色标记（`<|im_start|>system/user/assistant`）
- 后续轮次的环境反馈是**裸文本拼接**，无角色标记
- 与训练时 `ToolUtils.postprocess_output()` 的 token 拼接逻辑完全一致

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

| 维度 | `agl` (AGL) | `searchr1` (RLF) | `searchr1_multistep` (RLF) | `qwen3_tool` (旧版 RLF) |
|------|---|---|---|---|
| API | Chat Completions | **Completions** (raw prompt) | **Completions** (raw prompt) | Completions (raw prompt) |
| System 消息 | 无 | 有，内容为空 | 有，内容为空 | 有，包含 tool schema |
| 多轮结构 | 单条 user 消息拼接全部历史 | Token 级拼接，后续裸文本 | Token 级拼接 + user→assistant 交替 | 手工拼接多轮 ChatML tokens |
| 搜索指令 | `<search>query</search>` | `<search>query</search>` | `<search>query</search>` | `<tool_call>{"name":...}</tool_call>` |
| 搜索结果 | `<information>...</information>` | `<information>...</information>` | `<information>...</information>` | `<tool_response>...</tool_response>` |
| 角色标记 | 每轮重新 apply（仅一条 user 消息） | 仅首轮有，后续无角色标记 | 每轮有完整角色标记 | 每轮手工拼 ChatML 标记 |
| enable_thinking | 不传 | 首轮 prompt 中包含 `<think>\n` | 每轮有 `<think>\n` | N/A（手工注入空 think block） |
