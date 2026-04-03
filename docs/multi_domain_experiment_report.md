# Multi-Domain Search Agent 实验总结

> **目标读者**：团队内部 / Mentor 汇报  
> **实验框架**：RL-Factory (RLF)  
> **模型**：Qwen3-0.6B  
> **日期**：2026 年 3 月 — 4 月

---

## 1. 研究动机

在单域 Search R1 Agent 实验成功的基础上，我们尝试将其扩展到跨领域场景。目标是训练一个能够在 **biomedical**、**financial**、**science** 三个领域自主进行检索和问答的 Search Agent，模型需要学会：

1. 判断问题属于哪个领域
2. 生成 `<domain>` 标签路由到对应领域的知识库
3. 生成 `<search>` 标签发起检索
4. 综合检索结果回答问题

**结论前置**：由于数据集质量问题（biomedical 占比 90%+ 且任务过于简单），该方向在初步验证后暂停迭代。本报告总结数据构建工作、初步实验结果、以及项目搁置的具体原因。

---

## 2. 多领域数据集构建

### 2.1 数据来源

我们从 5 个跨 3 领域的公开 QA 数据集构建训练和测试集：

| 数据集 | 领域 | Train 行数 | Test 行数 | Answer 类型 | 来源说明 |
|--------|------|-----------|----------|------------|---------|
| PubMedQA | biomedical | 169,724 | 42,438 | yes / no / maybe | 生物医学文献问答 |
| BioASQ | biomedical | 209 | 117 | 抽取式短答案 | 生物医学抽取问答 |
| FinQA | financial | 6,214 | 1,203 | 数值（计算推理） | 金融数值推理 |
| OmniEval | financial | 1,079 | 452 | 抽取式短答案 | 金融知识抽取问答 |
| SciQ | science | 9,751 | 2,447 | 多选（单一正确选项） | 科学常识问答 |

### 2.2 合并后领域分布

| 领域 | Train | Test | 合计占比 |
|------|-------|------|---------|
| biomedical | 169,933 | 42,555 | **90.9%** |
| science | 9,751 | 2,447 | 5.2% |
| financial | 7,293 | 1,655 | 3.9% |
| **合计** | **186,977** | **46,657** | |

**核心问题**：biomedical 领域占比超过 90%，数据严重不均衡。

### 2.3 处理管线

完整的数据处理管线（位于 `data_preprocess/`）：

| 步骤 | 脚本 | 输出 |
|------|------|------|
| 1. 各数据集清洗、去重、去泄漏 | `preprocess_{bioasq,pubmedqa,omnieval,finqa,sciq}.py` | `processed_dataset/` |
| 2. 合并为统一格式 | `merge_datasets.py` | `merged_dataset/{train,test}.jsonl` |
| 3. 按 domain 生成知识库 | `build_kb.py` | `multi_domain_dataset/knowledge_base/` |
| 4. 转换为检索器 corpus 格式 | `prepare_retriever_corpus.py` | `multi_domain_retriever/data/` |
| 5. 构建 FAISS 索引 | `build_faiss_index.py` | 基于 multilingual-e5-base |
| 6. 生成训练用 parquet | `scripts/multi_domain_search_data.py`（RLF 侧） | `data/multi_domain_search/` |

一键重建：`bash rebuild_all.sh`

**数据质量验证：**
- 所有 processed → merged → knowledge_base → retriever corpus 行数链路一致 ✓
- FAISS 索引大小符合预期（`向量数 × 768 × 4 bytes`）✓
- BioASQ / OmniEval / FinQA 已做 train/test query 级去泄漏 ✓
- PubMedQA（10 条）和 SciQ（27 条）存在轻微 query 重叠（随机 split，未做 query 级去泄漏）

---

## 3. 领域均衡采样

### 3.1 问题

默认 `RandomSampler` 下，每个 batch 中 biomedical 样本占 ~90%，financial 和 science 领域被严重欠采样，模型几乎无法学到这两个领域的搜索和问答能力。

### 3.2 解决方案：DomainWeightedSampler

实现了 `DomainWeightedSampler`，通过 RLF 的 `sampler.class_path` 扩展点注入，不修改核心训练代码。支持三种策略：

| 策略 | 配置 | 效果 |
|------|------|------|
| `equal` | `strategy=equal` | 三领域等概率采样（各 33.3%） |
| `custom` | `strategy=custom` + `domain_weights` | 自定义权重，如 bio:fin:sci = 2:1:1 |
| `temperature` | `strategy=temperature` + `temperature=T` | $w_i \propto n_i^{1/T}$，T 越大越均衡 |

### 3.3 实际使用配置

训练中采用 **custom 2:1:1** 策略：

| 领域 | 数据量 | 采样权重 | 每 batch 预期占比 |
|------|--------|---------|----------------|
| biomedical | 169,933 | 50% | 64 / 128 |
| science | 9,751 | 25% | 32 / 128 |
| financial | 7,293 | 25% | 32 / 128 |

在 200 步 × batch 128 的训练中，三个领域的数据均无需重复采样（总采样 25,600，小于最小领域 7,293 × 4 = 29,172 的有效容量）。

---

## 4. 框架扩展

### 4.1 多领域 Tool Manager

在单域 `SearchR1Manager` 基础上扩展 `MultiDomainSearchR1Manager`：

- 模型需同时输出 `<search>query</search>` 和 `<domain>domain_name</domain>` 标签
- 调用多领域检索服务器 `POST /retrieve`，通过 `domain` 参数路由到对应领域的 FAISS 索引
- 支持三种 prompt 模式变体：`multi_domain_searchr1` / `_multistep` / `_agl`

### 4.2 多领域 Search 环境

`MultiDomainSearchEnv` 继承 `SearchEnv`，新增：

- 验证阶段输出各领域独立的 EM 分数
- 支持 `agl` 和 `multi_dim` 两种 reward mode
- 领域诊断信息打印

---

## 5. 实验结果与问题诊断

### 5.1 训练配置

与单域实验对齐的超参数（batch_size=128, lr=1e-6, max_turns=4, n=5），额外配置：

- 检索服务：`http://127.0.0.1:8001`（多领域 FAISS 索引）
- 采样策略：custom 2:1:1（bio:50%, fin:25%, sci:25%）
- Reward mode：agl（Pure EM）

### 5.2 训练指标

| 指标 | 值 |
|------|-----|
| 训练 reward（收敛后） | [TODO: 从 Wandb 补充] |
| 单步耗时 | [TODO: 从 Wandb 补充] |
| 收敛步数 | [TODO: 从 Wandb 补充] |

### 5.3 Per-Domain EM 分解（核心发现）

| 领域 | 验证集 EM | 分析 |
|------|----------|------|
| biomedical | **~90%** | PubMedQA 答案为 yes/no/maybe，任务过于简单 |
| financial | [TODO] | 数据量少（7.3K），数值推理类答案 EM 匹配困难 |
| science | [TODO] | 数据量适中（9.7K），多选形式 |

### 5.4 问题诊断

**问题一：Biomedical 领域误导性高分**

PubMedQA 占总数据 90%+，其答案类型为 yes / no / maybe 三分类。模型无需深入理解医学内容，仅靠简单的文本匹配即可获得 ~90% 的 EM。这导致：
- 整体 reward 被 biomedical 带高，训练信号被简单样本主导
- 无法反映模型在 financial / science 上的真实学习进度

**问题二：Financial 数据量不足且答案格式复杂**

FinQA 的答案为数值计算结果（如 "23.5%", "$1,234"），Exact Match 对数值格式敏感（"23.5%" vs "0.235"），导致 EM 偏低且噪声大。OmniEval 仅 1,079 条，数据量不足以支撑有效学习。

**问题三：领域不均衡的根本矛盾**

即使使用 DomainWeightedSampler 将采样权重调整为 2:1:1，biomedical 领域的高 EM 仍然主导 reward 信号。模型倾向于对所有问题都生成 biomedical 风格的简短回答，而非学习金融数值推理或科学推理的能力。

### 5.5 与单域 Baseline 对比

| 指标 | 单域 (Step 200) | 多域 [TODO] | 说明 |
|------|---------------|------------|------|
| Overall EM | 33.62% | [TODO] | 不可直接对比（数据集不同） |
| 搜索行为 | avg 1.13 | [TODO] | — |
| 格式合规 | 99.92% | [TODO] | — |

---

## 6. 结论与未来方向

### 6.1 完成的工作

| 工作项 | 状态 |
|--------|------|
| 5 数据集预处理管线 | ✅ 完成并验证 |
| 多领域知识库 + FAISS 索引构建 | ✅ 完成 |
| DomainWeightedSampler 实现 | ✅ 完成，支持 3 种策略 |
| MultiDomainSearchR1Manager | ✅ 完成，支持 3 种 prompt 模式 |
| MultiDomainSearchEnv | ✅ 完成，支持 per-domain 诊断 |
| 初步训练与评测 | ✅ 完成 |

### 6.2 项目搁置原因

搁置原因是**数据集质量问题**，而非框架能力不足：

1. PubMedQA 的 yes/no/maybe 任务过于简单，无法作为有效的 biomedical QA 训练数据
2. Financial 和 science 领域数据量不足，与 biomedical 的量级差距过大
3. 即使用采样器调权，简单样本的高 reward 仍主导训练信号

### 6.3 未来改进方向

若后续重启此方向，建议：

| 改进 | 具体方案 |
|------|---------|
| **替换 PubMedQA** | 使用需要抽取式/生成式答案的 biomedical QA 数据集（如 BioASQ 的 factoid/list 类型，或 MedQA） |
| **扩充 fin/sci 数据** | 增加金融和科学领域的高质量 QA 数据集至万级以上 |
| **答案归一化** | 对 FinQA 数值答案做格式归一化后再计算 EM |
| **Per-domain reward** | 考虑按领域设置不同的 reward 权重或归一化，避免简单领域主导 |
| **难度过滤** | 过滤掉 EM 过高（>0.8）的简单样本，保留有挑战性的训练数据 |

### 6.4 框架层面的价值

尽管项目暂停，以下框架层面的组件已验证可用，可直接复用于其他多领域场景：

- 多领域检索路由（`<domain>` 标签 → FAISS 索引路由）
- DomainWeightedSampler（不均衡数据集的通用解决方案）
- Per-domain 验证诊断（便于分析跨领域表现差异）
- 完整的数据处理管线（可替换数据源后复用）
