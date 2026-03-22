# Change Log

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
