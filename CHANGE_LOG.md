# Change Log

## 2026-03-30: 多领域数据集构建与领域均衡采样

### 多领域训练数据集

训练数据由 5 个跨 3 领域的 QA 数据集合并而成，完整处理管线位于 `Search_agent_data/data_preprocess/`。

**数据来源与规模：**

| 数据集 | 领域 | train 行数 | test 行数 | 来源说明 |
|--------|------|-----------|----------|---------|
| PubMedQA | biomedical | 169,724 | 42,438 | 生物医学问答，answer 为 yes/no/maybe |
| BioASQ | biomedical | 209 | 117 | 生物医学抽取式问答 |
| FinQA | financial | 6,214 | 1,203 | 金融数值推理，answer 为数值 |
| OminEval | financial | 1,079 | 452 | 金融抽取式问答（中文） |
| SciQ | science | 9,751 | 2,447 | 科学多选问答 |

**合并后按领域统计：**

| 领域 | train | test | 占比 |
|------|-------|------|------|
| biomedical | 169,933 | 42,555 | 90.9% |
| science | 9,751 | 2,447 | 5.2% |
| financial | 7,293 | 1,655 | 3.9% |
| **合计** | **186,977** | **46,657** | |

**处理管线（`data_preprocess/`）：**

1. `preprocess_{bioasq,pubmedqa,omnieval,finqa,sciq}.py` — 各数据集清洗、去重、去泄漏，统一为 `query, relevant_passage, answer` 格式
2. `merge_datasets.py` — 合并为 `merged_dataset/{train,test}.{csv,jsonl}`，分配 `qid`/`pid`，添加 `domain`/`source` 字段
3. `build_kb.py` — 按 domain 生成知识库 JSONL（`multi_domain_dataset/knowledge_base/`）
4. `prepare_retriever_corpus.py` — 转换为检索器 corpus 格式（`multi_domain_retriever/data/`）
5. `build_faiss_index.py` — 基于 multilingual-e5-base 构建 FAISS 索引
6. `scripts/multi_domain_search_data.py`（RL-Factory 侧）— 从 JSONL 生成训练用 parquet（`data/multi_domain_search/`）

一键重建：`bash rebuild_all.sh`

**数据完整性验验证结果：**
- 所有 processed → merged → knowledge_base → retriever corpus 行数链路一致 ✅
- FAISS 索引大小精确匹配 `向量数 × 768 × 4 bytes` ✅
- BioASQ/OminEval/FinQA 已做 train/test query 去泄漏 ✅
- PubMedQA（10 条）和 SciQ（27 条）存在轻微 query 重叠（随机 split，未做 query 级去泄漏）

### 领域均衡采样器（DomainWeightedSampler）

**问题：** 数据集严重不均衡，biomedical 占 90.9%，默认 `RandomSampler` 下 financial/science 领域被严重欠采样。

**方案：** 实现 `DomainWeightedSampler`，通过 `sampler.class_path` 扩展点注入，不改变训练代码。

**新增文件：**

| 文件 | 说明 |
|------|------|
| `verl/experimental/dataset/domain_weighted_sampler.py` | `DomainWeightedSampler` 类，继承 `AbstractSampler` |

**修改文件：**

| 文件 | 修改内容 |
|------|----------|
| `verl/trainer/config/data/legacy_data.yaml` | `sampler` 下新增 `strategy`、`domain_key`、`domain_weights`、`temperature` 字段定义（Hydra struct 要求预声明） |
| `main_grpo_multi_domain_search.sh` | 添加 sampler 配置参数、`val_batch_size=128` |

**支持三种策略：**

| 策略 | 配置 | 效果 |
|------|------|------|
| `equal` | `strategy=equal` | 三领域等概率采样（各 33.3%） |
| `custom` | `strategy=custom` + `domain_weights` | 自定义权重，如 bio:fin:sci = 2:1:1 → 50%:25%:25% |
| `temperature` | `strategy=temperature` + `temperature=T` | $w_i \propto n_i^{1/T}$，T 越大越均衡 |

**训练脚本中配置示例（当前默认 custom 2:1:1）：**

```bash
data.sampler.class_path=pkg://verl.experimental.dataset.domain_weighted_sampler
data.sampler.class_name=DomainWeightedSampler
data.sampler.strategy=custom
data.sampler.domain_key=data_source
data.sampler.domain_weights.biomedical=2
data.sampler.domain_weights.financial=1
data.sampler.domain_weights.science=1
data.dataloader_num_workers=0   # 自定义 sampler 要求
```

**200 步 × batch 128 下的采样分析（custom 2:1:1）：**

| 领域 | 数据量 | 采样权重 | 预期采样数 | 重复次数 |
|------|--------|---------|-----------|---------|
| biomedical | 169,933 | 50% | ~12,800 | 0.08× |
| science | 9,751 | 25% | ~6,400 | 0.66× |
| financial | 7,293 | 25% | ~6,400 | 0.88× |

三个领域的数据在 200 步内均无需重复采样。

**命令行切换策略：**

```bash
# 等权
data.sampler.strategy=equal

# 温度平滑 T=3（bio 58.7%, sci 22.7%, fin 20.6%）
data.sampler.strategy=temperature data.sampler.temperature=3.0

# 自定义权重
data.sampler.strategy=custom data.sampler.domain_weights.biomedical=3
```

---

## 2026-03-27: searchr1_agl 模式改为完全对齐 AGL（full reprompt）

### 背景

AGL 框架（Agent Lightning）的多轮 rollout 行为为：每一轮将上一轮的 response 和搜索结果拼接到 user content 后面，**以完整的单条 user message 重发**给 vLLM：

```python
rollout_content = ""
while turn_id < max_turns:
    response = call_llm(client, model, prompt + rollout_content)
    rollout_content += response + "\n\n<information>...</information>\n\n"
    # 下一轮：messages = [{"role": "user", "content": prompt + rollout_content}]
```

此前的 `searchr1_agl` 仅在评测层面使用 chat API 对齐 AGL，**训练时仍然使用 flat token 拼接（info 追加在 assistant turn 内部）**，与 AGL 的训练 token 序列不同。

现在改为**训练和评测都完全对齐 AGL**：每轮通过 `apply_chat_template` 重新编码完整 user message，生成全新的 prompt tokens。

### 训练 token 序列对比

| 模式 | 训练序列结构 | resp1 在哪里 | info 在哪里 |
|---|---|---|---|
| `searchr1` | `[P1][R1][info1][R2][info2][R3]` | assistant turn 内 | assistant turn 内（无角色标记） |
| `searchr1_multistep` | `[P1][R1][<\|im_end\|>user_info<\|im_end\|>asst][R2]...` | assistant turn 内 | user message 中 |
| `searchr1_agl`（新） | `[P1][R1][P2][R2][P3][R3]` | P2 的 user content 中 | P2 的 user content 中 |

其中 `P_N = apply_chat_template([{"role":"user","content":"INSTR+Q+resp1+info1+...+resp(N-1)+info(N-1)"}])`

**`searchr1_agl` 训练序列展开（2 轮 search 示例）：**
```
[P1: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{INSTR+Q}<|im_end|>\n<|im_start|>assistant\n<think>\n]
[R1: {resp1_tokens}]                                              ← loss=1
[P2: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{INSTR+Q+resp1+info1}<|im_end|>\n<|im_start|>assistant\n<think>\n]
[R2: {resp2_tokens}]                                              ← loss=1
```

### 实现方式

| 组件 | 变更 |
|------|------|
| `SearchR1AGLManager` | 设置 `reprompt_mode = True`；新增 `reset_reprompt_state()`、`init_sample()`、`accumulate_and_build_reprompt()`、`extract_user_content()` 方法 |
| `tool_utils.py` | 检测 `reprompt_mode`；初始化时解码 prompt 提取 user content 并调用 `init_sample()`；每轮调用 `accumulate_and_build_reprompt()` 获取完整 reprompt 作为下一轮推理的 prompt（非 flat concat） |

**`tool_utils.py` reprompt 模式关键差异：**

| 行为 | 标准模式 | reprompt 模式 |
|------|---------|---------------|
| info 处理 | `tokenizer(info_str)` → 追加到 `loop_responses_token` | 调用 `accumulate_and_build_reprompt()` → 完整 reprompt tokens |
| 下一轮 prompt | `chain(loop_responses_token[idx])` 平坦拼接所有段 | **仅使用 reprompt tokens**（不含之前的段） |
| `loop_responses_token` 存储 | `[P1, R1, info1, R2, info2, R3]` | `[P1, R1, P2, R2, P3, R3]`（P2 是完整 reprompt） |
| loss mask | 交替：R→1, info→0 | 交替：R→1, P→0（相同的奇偶逻辑） |

### 已知限制

在 RLF 的单序列训练中，训练序列为 `[P1][R1][P2][R2]...`，计算 `log π(R2 | P1+R1+P2)` 时前缀包含 P1+R1。而 AGL 的 Triplet 系统中，`log π(R2 | P2)` 仅以 P2 为前缀。这是 RLF 单序列架构与 AGL 独立 Triplet 架构的内在差异。但由于 P2 已包含完整历史（Q+resp1+info1），额外的 P1+R1 前缀是冗余信息，不会影响模型学到的行为模式。

### 当前所有 Prompt Mode 总览

**训练 tool_manager：**

| tool_manager 名称 | Manager 类 | 训练行为 |
|---|---|---|
| `searchr1` | `SearchR1Manager` | 原始文本拼接，单 assistant turn，info 无角色标记 |
| `searchr1_multistep` | `SearchR1MultistepManager` | 显式 user→assistant turn 交替 |
| `searchr1_agl` | `SearchR1AGLManager` | **完全对齐 AGL：每轮 full reprompt，info 在 user content 中** |
| `multi_domain_searchr1` | `MultiDomainSearchR1Manager` | 多领域版 searchr1 |
| `multi_domain_searchr1_multistep` | `MultiDomainSearchR1MultistepManager` | 多领域版 searchr1_multistep |
| `multi_domain_searchr1_agl` | `MultiDomainSearchR1AGLManager` | 多领域版 searchr1_agl |

**评测 prompt_mode：**

| `--prompt-mode` | API 方式 | 多轮构建 | `--label` 自动检测 | 对应训练 |
|---|---|---|---|---|
| `agl` | chat.completions | 全部内容塞入单个 user message | 默认 | AGL 框架 |
| `searchr1_agl` | chat.completions | 与 `agl` 完全相同 | `rlf_agl` | RLF `tool_manager=searchr1_agl` |
| `searchr1` | completions (raw) | 原始文本拼接 | `rlf` | RLF `tool_manager=searchr1` |
| `searchr1_multistep` | completions (raw) | 显式 user→assistant turn 交替 | `rlf_multistep` | RLF `tool_manager=searchr1_multistep` |
| `qwen3_tool` | completions (raw) | `<tool_call>/<tool_response>` 格式 | 需手动指定 | legacy RLF |

### 使用方式

```bash
# === 训练 ===
bash main_grpo_searchr1.sh                # searchr1（flat concat）
bash main_grpo_searchr1_multistep.sh      # searchr1_multistep（user→assistant 交替）
bash main_grpo_searchr1_agl.sh            # searchr1_agl（完全对齐 AGL：每轮 full reprompt）

# === 评测 ===
python eval_searchr1.py --label rlf_agl ...        # 自动选择 searchr1_agl prompt mode
python eval_searchr1.py --prompt-mode searchr1_agl ... # 显式指定
```

---

## 2026-03-23: 恢复原 RL-Factory 多维度 Reward 作为可选模式

### 背景

此前（2026-03-21）为对齐 AGL 框架，将 `SearchEnv` 的 reward 简化为纯 Exact Match (EM)，移除了原 RL-Factory 的格式奖惩（format reward）、`<tool_call>` JSON 校验、XML 标签配对检查等多维度 reward 逻辑。为后续做对照实验，现将原 RL-Factory 的多维度 reward 恢复并保留为可选模式。

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `envs/search.py` | 新增 `reward_mode` 配置项，提取公共函数到模块级别，新增 `_compute_score_multi_dim()` 方法 |
| `envs/multi_domain_search.py` | 同步支持 `reward_mode`，新增 `_compute_score_multi_dim_domain()` 方法 |

### 设计

通过 `config.reward_mode` 配置项切换 reward 计算方式：

| `reward_mode` | 方法 | 行为 |
|---|---|---|
| `agl`（默认） | `_compute_score_agl()` | 纯 EM：答对 1.0，其余 0.0 |
| `multi_dim` | `_compute_score_multi_dim()` | 原 RLF 多维度 reward（见下表） |

**`multi_dim` 模式 Reward 组成：**

| 组件 | 计算方式 |
|------|----------|
| `answer_format_score` | `<answer>` 标签配对合法 → `+format_score` ，否则 `-format_score` |
| `tool_call_format_score` | `<tool_call>` 标签合法且 JSON 可解析 → 按成功率缩放；>2 次调用额外扣分 |
| `total_format_score` | `answer_format_score + num_score` |
| answer=None | `-format_score + 0.5 * total_format_score` |
| EM match | `1.0 + 0.5 * total_format_score` |
| EM 不匹配 | `total_format_score` |

其中 `format_score` = 0.1（训练）/ 0.0（验证）。

### 使用方式

在训练脚本中添加 `actor_rollout_ref.env.reward_mode=multi_dim` 即可切换到原 RLF reward：

```bash
# 使用 AGL 纯 EM reward（默认，无需额外配置）
bash main_grpo_searchr1.sh

# 使用原 RLF 多维度 reward（对照实验）
bash main_grpo_searchr1.sh actor_rollout_ref.env.reward_mode=multi_dim
```

### 代码重构说明

- `normalize_answer()`、`em_check()`、`extract_solution()`、`check_alternate_tags()` 提升为模块级公共函数（加下划线前缀），`SearchEnv` 和 `MultiDomainSearchEnv` 共享
- `_compute_score_with_rules()` 现为分发方法，根据 `reward_mode` 路由到对应实现
- `MultiDomainSearchEnv` 的 per-domain 诊断打印抽取为 `_print_domain_diagnostics()` 静态方法，两种 reward 模式均复用

---

## 2026-03-22: 新增多领域检索 Search Agent 及训练环境

### 背景

在单域 Search R1 的基础上，引入多领域检索能力（biomedical / financial / science），使模型能够通过 `<domain>` 标签路由到不同领域的 FAISS 索引进行检索，构建跨领域问答的 RL 训练流程。

### 新增文件

| 文件 | 说明 |
|------|------|
| `envs/tool_manager/multi_domain_searchr1_manager.py` | 多领域检索 Tool Manager，扩展 SearchR1Manager，新增 `<domain>` 标签解析和多领域 HTTP 检索调用 |
| `envs/multi_domain_search.py` | 多领域检索环境 `MultiDomainSearchEnv`，继承 SearchEnv，增加逐领域验证分数统计 |
| `scripts/multi_domain_search_data.py` | 数据预处理脚本，将多领域 QA 数据转换为 RL-Factory 所需的 parquet 格式 |
| `main_grpo_multi_domain_search.sh` | GRPO 训练脚本，超参数与 `main_grpo_searchr1.sh` 完全对齐 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `envs/tool_manager/__init__.py` | 注册 `multi_domain_searchr1` → `MultiDomainSearchR1Manager` |
| `envs/__init__.py` | 注册 `multi_domain_search` → `MultiDomainSearchEnv` |

### 关键设计

- **Prompt 格式**: 在 Search R1 指令基础上增加 `<domain> domain_name </domain>` 标签，模型需同时输出 `<search>` 和 `<domain>` 标签
- **检索路由**: 调用多领域检索服务器 `POST /retrieve`，通过 `domain` 参数路由到对应领域的 FAISS 索引
- **Reward 对齐**: 沿用 AGL 纯 Exact Match (EM) 评分，answer 正确 → 1.0，其余 → 0.0
- **验证诊断**: 验证阶段输出各领域单独的 EM 分数，便于分析跨领域表现差异
- **超参数对齐**: 与 `main_grpo_searchr1.sh` 完全一致（batch_size=128, lr=1e-6, max_turns=4, n=5 等）

### 使用方式

```bash
# 1. 启动多领域检索服务器
cd /path/to/Search_agent_checkpoints/multi_domain_retriever
bash multi_domain_launch.sh train 8000

# 2. 准备训练数据（从本地 JSONL 文件）
python scripts/multi_domain_search_data.py \
    --from_local \
    --train_files train.jsonl \
    --test_files test.jsonl \
    --local_dir ./data/multi_domain_search

# 3. 启动训练
bash main_grpo_multi_domain_search.sh
```

---

## 2026-03-21: Reward 计算对齐 AGL (Pure Exact Match)

**文件**: `envs/search.py` — `SearchEnv._compute_score_with_rules()`

### 背景

AGL 框架（`agent-lightning/contrib/recipes/search_r1/qa_em.py`）使用纯二值 Exact Match (EM) 作为 reward：答对得 1.0，其余为 0.0。RLF 原有实现在 EM 基础上叠加了格式奖惩（format reward），导致两个框架的训练信号不一致，无法公平对比。

### 修改内容

| 对比项 | 修改前 (RLF) | 修改后 (= AGL) |
|--------|-------------|----------------|
| `format_score` | 训练时 0.1，验证时 0.0 | 始终 **0.0** |
| `answer=None` 时 reward | `-0.1 + 0.5 * format_bonus`（可为负） | **0.0** |
| EM match 时 reward | `1.0 + 0.5 * format_bonus`（可 > 1.0） | **1.0** |
| EM 不匹配时 reward | `format_bonus`（训练时非零） | **0.0** |
| `<think>` 块处理 | 先用正则去除 `<think>...</think>` 再提取 answer | 不去除，直接提取 `<answer>` |
| XML 标签配对检查 | `check_alternate_tags()` 检查 `<answer>`/`<tool_call>`/`<search>` 标签合法性 | 移除 |
| `<tool_call>` JSON 校验 | 解析 JSON 合法性，不合法则扣分 | 移除 |
| 多余 answer 标签惩罚 | 无 | 无（AGL 也无此逻辑） |

### 移除的函数

- `check_alternate_tags()` — XML 标签配对检查辅助函数

### Reward 信号对照

```
AGL:  answer 正确 → 1.0 | answer 错误或无 answer → 0.0
RLF:  answer 正确 → 1.0 | answer 错误或无 answer → 0.0  (修改后)
```
