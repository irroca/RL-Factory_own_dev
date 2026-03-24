# Change Log

## 2026-03-24: 新增 AGL 风格消息构建模式 (searchr1_agl)

### 背景

AGL 框架（Agent Lightning）与 RLF 在多轮 LLM 调用时的消息拼接方式存在差异：

| 维度 | RLF SearchR1 (原) | AGL |
|------|-------------------|-----|
| 环境反馈格式 | 原始文本直接拼接，无角色标记 | 作为 user 消息发送，每次调用都重新应用 chat template |
| Token 序列 | `...<\|im_start\|>assistant\n{resp}\n\n<information>...\n\n{resp2}...` | `...{resp}<\|im_end\|>\n<\|im_start\|>user\n<information>...<\|im_end\|>\n<\|im_start\|>assistant\n{resp2}...` |
| 角色转换 | 无（所有后续内容都在一个 assistant turn 内） | 有（environment feedback → user turn → 新 assistant turn） |

为支持公平对照实验，现新增 AGL 风格的消息构建模式，将环境反馈（检索结果）包装为 user-role 消息并应用 chat template，使 token 序列具有显式的 user↔assistant 交替标记。

### 新增文件

| 文件 | 说明 |
|------|------|
| `envs/tool_manager/searchr1_agl_manager.py` | AGL 风格消息构建的 Tool Manager，包含 `SearchR1AGLManager` 和 `MultiDomainSearchR1AGLManager` 两个类 |
| `main_grpo_searchr1_agl.sh` | 使用 AGL 风格消息构建的 GRPO 训练脚本，超参数与 `main_grpo_searchr1.sh` 完全一致 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `envs/tool_manager/__init__.py` | 注册 `searchr1_agl` → `SearchR1AGLManager`，`multi_domain_searchr1_agl` → `MultiDomainSearchR1AGLManager` |

### 关键设计

核心差异在 `get_prompt(mode='tool_call')` 方法：

| 模式 | `get_prompt(mode='tool_call')` 行为 |
|------|--------------------------------------|
| `searchr1`（原） | 返回原始文本，不添加任何 chat template 标记 |
| `searchr1_agl`（新） | 使用 `apply_chat_template` 将内容包装为 user-role 消息，添加 `<\|im_start\|>user` / `<\|im_end\|>` / `<\|im_start\|>assistant` 标记 |

AGL 风格使用 Qwen3Manager 相同的 base-prompt 减法技巧：构造 dummy base_chat → 拼接实际消息 → apply_chat_template → 减去 base 前缀，得到纯净的角色标记 + 内容。

### 使用方式

```bash
# 方式一：使用 AGL 风格消息构建的训练脚本（单域检索）
bash main_grpo_searchr1_agl.sh

# 方式二：在现有脚本中切换 tool_manager 即可
bash main_grpo_searchr1.sh actor_rollout_ref.env.tool_manager=searchr1_agl

# 方式三：多领域检索 + AGL 风格
bash main_grpo_multi_domain_search.sh actor_rollout_ref.env.tool_manager=multi_domain_searchr1_agl
```

对照实验建议：

```bash
# 实验组 A：原 RLF 消息构建
bash main_grpo_searchr1.sh

# 实验组 B：AGL 风格消息构建
bash main_grpo_searchr1_agl.sh
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
