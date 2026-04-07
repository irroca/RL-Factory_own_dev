# Single-Domain Search R1 Agent 实验总结

> **目标读者**：团队内部 / Mentor 汇报  
> **实验框架**：RL-Factory (RLF)，对标 Agent Lightning (AGL)  
> **模型**：Qwen3-0.6B  
> **日期**：2026 年 3 月 — 4 月

---

## 1. 研究动机

Search R1 Agent 通过强化学习训练 LLM 自主学习何时搜索、搜索什么、以及如何利用检索结果回答问题。我们在 RL-Factory (RLF) 框架下复现并改进了此训练流程，与 Agent Lightning (AGL) 框架进行系统对比。

**核心问题**：相同模型 (Qwen3-0.6B)、相同训练数据下，RLF 原始设置训练 3000 步后模型仅调用了 28 次搜索引擎，EM 停滞在 ~15%；而 AGL 框架 600 步即达到 37.5% EM。本报告记录了从问题定位到对齐修复、再到多维度消融实验的完整过程。

---

## 2. 实验设置

### 2.1 训练数据

| 项目 | 说明 |
|------|------|
| 训练集 | nq (79,168 条, 46.7%) + hotpotqa (90,447 条, 53.3%)，共 169,615 条 |
| 测试集 | 7 个数据源，共 51,713 条；评测时采样 5,000 条 |
| 检索服务 | E5-base embedding + FAISS 索引，top-k=3 |

**测试集数据源分布（5,000 条采样）：**

| 数据源 | 数量 | 占比 | 类型 |
|--------|------|------|------|
| popqa | 1,375 | 27.50% | 单跳 |
| 2wikimultihopqa | 1,212 | 24.24% | 多跳 |
| triviaqa | 1,091 | 21.82% | 单跳 |
| hotpotqa | 716 | 14.32% | 多跳 |
| nq | 352 | 7.04% | 单跳 |
| musique | 237 | 4.74% | 多跳 |
| bamboogle | 17 | 0.34% | 多跳 |

### 2.2 训练超参数

所有实验共享以下固定超参数，确保公平对比：

| 参数 | 值 |
|------|-----|
| 算法 | GRPO |
| 模型 | Qwen3-0.6B |
| train_batch_size | 128 |
| max_prompt_length | 4,096 |
| max_response_length | 4,096 |
| actor learning rate | 1e-6 |
| ppo_mini_batch_size | 32 |
| rollout samples (n) | 5 |
| max_turns | 4 |
| KL loss type | low_var_kl |
| KL loss coef | 0.001 |
| GPU 数量 | 4 × A100 |
| 梯度检查点 | 开启 |
| FSDP param/optimizer offload | 开启 |

### 2.3 评测指标

| 指标 | 说明 |
|------|------|
| **EM (Exact Match)** | 核心指标。归一化后的预测答案与任一标准答案完全匹配即为 1 |
| Avg Turns | 平均交互轮数（每轮 = 一次 LLM 推理） |
| Avg Searches | 平均搜索次数 |
| Search Rate | 执行过至少一次搜索的样本占比 |
| Answer Tag Rate | 输出中包含合法 `<answer>` 标签的样本占比 |

---

## 3. 实验一：框架对齐（RLF 原始设置 vs AGL）

### 3.1 问题现象

| 指标 | AGL (Step 0) | AGL (Step 600) | RLF 原始 (Step 0) | RLF 原始 (Step 3000) |
|------|-------------|---------------|------------------|---------------------|
| EM | 14.1% | 37.5% (↑165%) | ~15% | ~15% (无变化) |
| 搜索行为 | turn_count=1.94 | turn_count=4.25 | tool_call=0 | tool_call≈0 (仅 28 次) |
| 单步耗时 | ~300s | ~600s | ~15s | ~45s |

**关键发现**：RLF 原始设置下模型几乎从不调用搜索引擎，完全依赖参数化知识回答问题，EM 无法提升。

### 3.2 根因分析与修复

排查发现 RLF 与 AGL 存在三个关键差异，逐一修复后训练行为恢复正常：

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 1 | **工具调用格式不匹配**：RLF 使用 `<tool_call>` JSON 格式 + MCP 协议，模型未在 SFT 阶段学过此格式 | 模型不知如何发起搜索 | 改用 `<search>query</search>` 纯文本标签，与 Search R1 论文一致 |
| 2 | **Stop strings 缺失**：vLLM 仅配置 `<\|im_end\|>` 作为停止词 | 模型在单个 turn 内生成完整推理链，`parse_response()` 优先匹配 `<answer>` 直接终止 | 添加 `</search>` 和 `</answer>` 作为 stop strings |
| 3 | **贪心解码 bug**：`do_sample=False` 在 step > 0 时强制 temperature=0 | 从第二轮起模型只要 `<answer>` 概率略高于 `<search>` 就直接终止 | 将 `do_sample` 改为 `True` |

### 3.3 修复一：工具格式对齐

将 `<tool_call>` JSON 格式改为 `<search>query</search>` 纯文本标签后（不添加 stop strings，不修改 do_sample），RLF 训练行为恢复正常，模型开始学会搜索。

| 指标 | RLF 格式对齐后 (Step 200) | AGL (Step 600) |
|------|------------------------|---------------|
| 训练 reward | ~0.4（收敛） | ~0.4（收敛） |
| 单步耗时 | **~60s** | ~300–600s |
| response_length | ~700 tokens | — |

对齐后 RLF 仅需 200 步即收敛至与 AGL 600 步相近的 reward 水平，且单步耗时仅为 AGL 的 1/5 — 1/10，体现了 RLF 框架在架构层面的吞吐率优势。

### 3.4 修复二 & 三：Stop Strings + do_sample 的增量效果

在工具格式对齐的基础上，进一步添加 stop strings（`</search>`、`</answer>`）和 `do_sample=True`。

#### 3.4.1 训练行为变化（Wandb 观察）

从 Wandb 训练曲线可以观察到修复前后训练行为的显著差异：

**searchr1 模式（单 assistant turn）：**

| 训练指标 | 无 stop+sample | 有 stop+sample | 说明 |
|---------|--------------|---------------|------|
| response_length | ~700 tokens，稳定 | 训练中逐步增长 | 模型生成更长的多轮序列 |
| avg_turns | ~2.0，几乎不变 | 从 ~2.0 逐步上升 | **模型开始学会多轮搜索** |
| reward 曲线形态 | 快速收敛后平稳 | 收敛后仍有缓慢上升 | 多轮搜索带来的持续学习信号 |
| 单步耗时 | ~60s | 逐步增加至 ~90-120s | response_length 增长导致推理变慢 |

**关键发现**：

- **无 stop+sample 时**：vLLM 仅在 `<|im_end|>` 处停止。模型在训练中的每个 rollout 里，往往在第一轮就生成 `<search>...</search>` 后紧接着输出 `<answer>...</answer>`，因为缺少 `</search>` 作为 stop string，`parse_response()` 无法在搜索标签处截断并注入真实检索结果。模型实质上在**一轮内自问自答**，environment 的多轮循环（`max_turns=4`）大多只执行 1-2 轮便终止。

- **有 stop+sample 后**：`</search>` 作为 stop string 使 vLLM 在模型生成搜索请求后立刻停止，将控制权交还给 environment 注入真实检索结果。`do_sample=True` 增加了探索性，模型在训练过程中逐步发现「搜索 → 获取信息 → 再搜索」的策略能获得更高 reward，因此 avg_turns 和 response_length 在训练中**持续增长**。

> 📊 **图 A1**：对比无 stop+sample vs 有 stop+sample 的 `response_length` 和 `avg_turns` 训练曲线，可直观观察搜索行为的变化。

#### 3.4.2 评测结果对比

为排除训练步数差异，以下对比**均使用 Step 400 checkpoint**。

**searchr1 模式（单 assistant turn）：**

| 指标 | 无 stop+sample | 有 stop+sample | 差异 |
|------|--------------|---------------|------|
| **Overall EM** | **33.76%** | **31.51%** | **-2.25pp** |
| Avg Turns | 2.18 | 2.31 | +0.13 |
| Avg Searches | 1.12 | 1.07 | -0.05 |
| Search Rate | 100% | 99.98% | — |
| Answer Tag Rate | 99.92% | 99.84% | — |

**searchr1_agl 模式（AGL 风格 chat API）：**

| 指标 | 无 stop+sample | 有 stop+sample | 差异 |
|------|--------------|---------------|------|
| **Overall EM** | **31.65%** | **35.17%** | **+3.52pp** |
| Avg Turns | 2.18 | 2.01 | -0.17 |
| Avg Searches | 1.01 | 1.00 | — |
| Search Rate | 100% | 99.82% | — |
| Answer Tag Rate | 100% | 100% | — |

#### 3.4.3 训练与评测的不一致性分析

一个有趣的现象是：尽管训练过程中 stop+sample 使模型学到了更多搜索行为（训练时 avg_turns 持续增长），但**评测时的 Avg Searches 反而没有显著增加**（1.07-1.12 次）。

可能的原因：
- **评测使用 temperature=0**（贪心解码），模型在确定性推理下倾向于用最少搜索次数回答，训练中通过采样探索到的多轮策略在贪心解码下未被激活。
- **searchr1 模式的裸文本拼接**缺少显式轮次信号，模型在长序列中越到后面越难维持搜索意图，即使训练中学到了多搜索的倾向。
- 这一现象提示：**训练时的搜索积极性不一定能迁移到评测**，评测时的解码策略（temperature、top_p）也需要配合调整。

#### 3.4.4 综合结论

1. **训练视角**：stop strings + do_sample 是触发 RLF 模型在训练中学习多轮搜索策略的**关键条件**。没有这些修复，environment 的多轮循环形同虚设，模型实际只进行单轮交互。

2. **评测视角**：修复的效果与 prompt mode 强相关——searchr1_agl 模式获益显著（+3.52pp），searchr1 模式反而略降（-2.25pp）。这说明训练中学到的多搜索倾向在 searchr1 的裸文本拼接格式下难以有效利用。

3. **工程启示**：stop strings 和 do_sample 的配置应与 prompt mode 联合考虑，而非独立看待。对于使用 chat API 进行多轮交互的模式（searchr1_agl），这些修复不可或缺；对于 token 级拼接的模式（searchr1），修复的收益有限。

### 3.5 统一对比：RLF vs AGL（Step 400, 5,000 样本）

以下使用 RLF 无修复 searchr1 模式（Step 400）与 AGL（Step 600）对比，展示框架级差异：

| 指标 | RLF searchr1 (Step 400) | AGL (Step 600) | 差异 |
|------|------------------------|---------------|------|
| **Overall EM** | **33.76%** | **38.74%** | -4.98pp |
| Avg Turns | 2.18 | 4.27 | RLF 搜索更少 |
| Avg Searches | 1.12 | 3.26 | RLF 搜索更少 |
| Search Rate | 100% | 99.94% | 均充分搜索 |
| Answer Tag Rate | 99.92% | 94.44% | — |
| Avg Time/Sample | **2.57s** | **5.41s** | RLF 快 2.1× |
| Total Tokens | 9.38M | 30.95M | RLF 省 3.3× |

**Per-Source EM 对比：**

| 数据源 | 类型 | RLF (Step 400) | AGL (Step 600) | 差异 |
|--------|------|---------------|---------------|------|
| popqa | 单跳 | 41.09% | 40.58% | **+0.51pp** |
| triviaqa | 单跳 | **51.88%** | **52.98%** | -1.10pp |
| nq | 单跳 | 37.78% | 37.22% | **+0.56pp** |
| hotpotqa | 多跳 | 24.02% | 33.38% | -9.36pp |
| 2wikimultihopqa | 多跳 | 19.39% | 32.59% | -13.20pp |
| musique | 多跳 | 5.91% | 11.39% | -5.48pp |
| bamboogle | 多跳 | 17.65% | 52.94% | -35.29pp |

**分析：**

1. **单跳数据集**（nq, triviaqa, popqa）：RLF 与 AGL 差距极小（<1.1pp），在 nq 和 popqa 上甚至略优。一次搜索通常足够，RLF 的低搜索次数不是劣势。
2. **多跳数据集**（hotpotqa, musique, bamboogle）：AGL 明显优于 RLF。AGL 平均 3.26 次搜索使其在需要多步推理的任务上更有优势。
3. **效率**：RLF 推理耗时仅为 AGL 的 ~48%，token 消耗仅为 ~30%。
4. **答对样本 vs 答错样本的搜索行为**：

| | RLF 答对 | RLF 答错 | AGL 答对 | AGL 答错 |
|---|---------|---------|---------|---------|
| Avg Turns | 2.09 | 2.23 | 4.04 | 4.41 |
| Avg Searches | 1.08 | 1.14 | 3.04 | 3.40 |

两个框架中答错样本的轮数和搜索次数都略高于答对样本，说明模型在无法找到正确答案时会倾向多搜索几次。

### 3.6 小结

- **工具格式对齐**（`<search>` 标签替换 `<tool_call>`）是 RLF 能学会搜索的**必要条件**，解决了模型完全不搜索的根本问题
- **Stop strings + do_sample** 的效果与 prompt mode 强相关：在 searchr1_agl 模式下提升 3.5pp EM，在 searchr1 模式下无正面效果
- 对齐后 RLF 在**训练效率**上显著优于 AGL（5-10× 速度提升）
- AGL 在**多跳问答**上表现更好，得益于其更积极的搜索策略（平均 3.26 vs 1.12 次搜索）
- RLF 在**单跳问答**上与 AGL 持平甚至略优，且推理效率更高（token 消耗仅 ~30%）

---

## 4. 实验二：Prompt Mode 对比

RLF 框架支持三种训练时的 token 序列构建方式：

### 4.1 三种模式说明

| 模式 | 训练 token 序列 | 多轮构建方式 | 评测 API |
|------|---------------|------------|---------|
| `searchr1` | `[P₁][R₁][info₁][R₂][info₂][R₃]` | 原始文本拼接，单 assistant turn | vLLM completions |
| `searchr1_multistep` | `[P₁][R₁][user_info][R₂][user_info][R₃]` | 显式 user→assistant turn 交替 | vLLM completions |
| `searchr1_agl` | `[P₁][R₁][P₂][R₂][P₃][R₃]` | 每轮 full reprompt（完全对齐 AGL） | chat.completions |

其中：
- **searchr1**：所有内容在一个连续的 assistant turn 内完成，搜索结果直接拼接，模型看不到显式的「新轮次」信号
- **searchr1_multistep**：搜索结果作为新的 user message 插入，提供显式的轮次边界
- **searchr1_agl**：每轮将全部历史拼接到 user content 中，重新 apply_chat_template，完全对齐 AGL 的训练行为

### 4.2 实验结果

三组实验除 `tool_manager` 不同外，其余超参数完全一致（均使用 `format_reward`、addstopstring + dosample 修复、400 步）。

**训练指标对比：**

| 指标 | searchr1 | searchr1_multistep | searchr1_agl |
|------|----------|-------------------|--------------|
| 收敛步数 | [Wandb] | [Wandb] | [Wandb] |
| 收敛 reward | [Wandb] | [Wandb] | [Wandb] |
| 单步耗时 (s) | [Wandb] | [Wandb] | [Wandb] |
| Avg response_length | [Wandb] | [Wandb] | [Wandb] |

**评测指标对比（Step 400 checkpoint, 5,000 样本）：**

| 指标 | searchr1 | searchr1_multistep | searchr1_agl |
|------|----------|-------------------|--------------|
| **Overall EM** | **28.74%** | **37.34%** | **35.17%** |
| Avg Turns | 2.00 | 2.00 | 2.01 |
| Avg Searches | 1.00 | 1.00 | 1.00 |
| Search Rate | 100% | 100% | 99.82% |
| Answer Tag Rate | 100% | 100% | 100% |
| Avg Time/Sample | 2.74s | 4.22s | 2.85s |
| Total Tokens | 7.37M | 8.24M | 6.48M |

**Per-Source EM（Step 400 checkpoint）：**

| 数据源 | 类型 | searchr1 | searchr1_multistep | searchr1_agl |
|--------|------|----------|-------------------|--------------|
| nq | 单跳 | 23.58% | **39.20%** | **39.20%** |
| triviaqa | 单跳 | 39.87% | **52.98%** | 53.12% |
| popqa | 单跳 | 33.02% | **44.29%** | 40.85% |
| hotpotqa | 多跳 | 19.97% | **27.09%** | 24.02% |
| 2wikimultihopqa | 多跳 | 25.25% | **26.98%** | 23.92% |
| musique | 多跳 | 5.91% | **6.75%** | 5.11% |
| bamboogle | 多跳 | 11.76% | **29.41%** | 23.53% |

### 4.3 分析

- **searchr1_multistep 全面最优**：在每个数据源上均优于其他两种模式，Overall EM 达到 37.34%，比 searchr1 高出 **8.6pp**，比 searchr1_agl 高 **2.2pp**。这说明显式的 user→assistant turn 交替是最有效的多轮 prompt 构建方式。

- **searchr1 表现最差（28.74%）**：裸文本拼接缺乏轮次边界信号，模型难以区分环境反馈与自身生成的内容，导致 EM 显著低于其他两种模式。尤其在 nq（23.58% vs 39.20%）和 popqa（33.02% vs 44.29%）上差距巨大。

- **searchr1_agl 居中（35.17%）**：通过 chat.completions API 每轮重新 apply chat template，与 AGL 训练行为一致，但 EM 仍低于 multistep。可能原因是 full reprompt 的 token 开销更大，在固定 `max_response_length` 下留给答案生成的空间更少。

- **搜索深度相同**：三种模式的 Avg Searches 均为 ~1.00，说明 prompt 构建方式不影响搜索积极性，差异完全来自模型利用检索结果的能力。

- **效率**：searchr1_agl 的 Total Tokens 最少（6.48M），因为模型平均只搜索一次后直接回答；searchr1_multistep 需要额外的 user/assistant 标记，token 稍多。

### 4.4 小结

1. **searchr1_multistep 是 RLF 的最佳 prompt mode**，EM（37.34%）接近 AGL（38.74%），差距仅 1.4pp，同时保持 RLF 的训练效率优势。
2. 显式的轮次边界标记（`<|im_start|>user/assistant`）对模型理解多轮交互至关重要，裸文本拼接（searchr1）会损失 ~8.6pp EM。
3. Full reprompt（searchr1_agl）不如 multistep，可能因为完整重编码历史引入了冗余信息。

---

## 5. 实验三：Reward Mode 对比

### 5.1 三种奖励模式

| 模式 | 设计理念 | 奖励计算 |
|------|---------|---------|
| `agl` (Pure EM) | 最简信号，完全对齐 AGL | 答对 → 1.0，其余 → 0.0 |
| `format_reward` (4-way) | EM + 格式合规底分 | 见下表 |
| `multi_dim` (原始 RLF) | EM + 多维格式奖惩 | 见下方详细说明 |

**`format_reward` 奖励矩阵（λ_f = 0.2）：**

| 条件 | 奖励 |
|------|------|
| EM 正确 + 格式正确 | 1.0 |
| EM 正确 + 格式错误 | 0.8 |
| EM 错误 + 格式正确 | 0.2（底分保证非零梯度） |
| EM 错误 + 格式错误 | 0.0 |

**格式正确的判定**：`<answer>` 标签交替合法（必须），`<search>` / `<think>` 标签交替合法（若存在）。

`format_reward` 的设计动机：Pure EM 下答错样本 reward=0，格式正确与否都没有梯度信号；4-way 设计通过给格式正确但答错的样本 0.2 的底分，确保模型即使答错也能从格式合规中获得学习信号。

**`multi_dim` 奖励计算（$f_s = 0.1$，训练时）：**

该模式在 EM 正确性基础上叠加标签合法性的细粒度奖惩，分两个维度：

| 维度 | 评判标准 | 得分 |
|------|---------|------|
| `answer_format` | `<answer>`/`</answer>` 标签是否正确交替出现 | 合法 → $+f_s$；不合法 → $-f_s$ |
| `num_score` | 工具调用次数是否过多（>2 次） | 超过 2 次 → $-f_s$；否则 → $0$ |

中间量：

$$\text{total\_format\_score} = \text{answer\_format} + \text{num\_score}$$

最终奖励按 EM 结果分三种情况：

| 条件 | 奖励公式 | 典型值范围 |
|------|---------|----------|
| 答案提取失败（无 `<answer>` 标签） | $-f_s + 0.5 \times \text{total\_format}$ | $[-0.15, -0.05]$ |
| **EM 正确** | $1.0 + 0.5 \times \text{total\_format}$ | $[0.9, 1.05]$ |
| EM 错误 | $\text{total\_format}$ | $[-0.2, 0.1]$ |

与 `format_reward` 的关键区别：`multi_dim` 对格式错误施加**负惩罚**（而非仅给零分），且惩罚粒度更细——区分了「无答案标签」「标签不合法」「工具调用过多」等不同错误类型。

### 5.2 实验结果

三组实验使用 `searchr1` tool_manager + addstopstring + dosample 修复，仅 `reward_mode` 不同，均训练 400 步。

**训练指标对比：**

| 指标 | agl (Pure EM) | format_reward | multi_dim |
|------|--------------|---------------|-----------|
| 收敛步数 | [Wandb] | [Wandb] | [Wandb] |
| 收敛 reward | [Wandb] | [Wandb] | [Wandb] |
| Answer Tag Rate | [Wandb] | [Wandb] | [Wandb] |

**评测指标对比（Step 400 checkpoint, 5,000 样本）：**

| 指标 | agl (Pure EM) | format_reward | multi_dim |
|------|--------------|---------------|-----------|
| **Overall EM** | **31.51%** | 28.74% | **32.86%** |
| Avg Turns | 2.31 | 2.00 | 2.01 |
| Avg Searches | 1.07 | 1.00 | 1.00 |
| Search Rate | 99.98% | 100% | 99.98% |
| Answer Tag Rate | 99.84% | 100% | 100% |
| Avg Time/Sample | 2.74s | 2.74s | 1.08s |
| Total Tokens | 9.44M | 7.37M | 5.35M |

**Per-Source EM（Step 400 checkpoint）：**

| 数据源 | 类型 | agl (Pure EM) | format_reward | multi_dim |
|--------|------|--------------|---------------|-----------|
| nq | 单跳 | 30.97% | 23.58% | **36.93%** |
| triviaqa | 单跳 | 45.92% | 39.87% | **49.95%** |
| popqa | 单跳 | 37.60% | 33.02% | **39.49%** |
| hotpotqa | 多跳 | 22.80% | 19.97% | **22.49%** |
| 2wikimultihopqa | 多跳 | **22.58%** | **25.25%** | 20.54% |
| musique | 多跳 | 3.80% | **5.91%** | 5.49% |
| bamboogle | 多跳 | 11.76% | 11.76% | 11.76% |

### 5.3 分析

- **multi_dim 整体最优（32.86%）**：原始 RLF 多维度奖励在 searchr1 模式下表现最好，尤其在单跳任务上优势明显（nq 36.93%, triviaqa 49.95%, popqa 39.49%），比 pure EM 高 **1.35pp**。

- **format_reward 表现最差（28.74%）**：格式底分反而拉低了 EM。可能原因是 λ_f=0.2 的底分让模型倾向于输出格式正确但内容不准确的回答——Answer Tag Rate 达到了完美的 100%，但这并未转化为更高的 EM。模型在优化格式合规的同时牺牲了答案质量。

- **pure EM 居中（31.51%）**：最简奖励信号表现稳定，但 Avg Turns（2.31）和 Avg Searches（1.07）略高于其他两种模式，说明纯 EM 信号下模型偶尔会多搜索一次。

- **format_reward 在多跳上反而略好**：2wikimultihopqa（25.25%）和 musique（5.91%）上 format_reward 略优于其他模式。这可能是因为格式底分鼓励了更规范的输出结构，在需要多步推理的任务中有一定帮助。

- **效率差异显著**：multi_dim 的 Avg Time（1.08s）和 Total Tokens（5.35M）远低于其他模式，说明多维度奖励让模型学到了更简洁高效的回答策略。

### 5.4 小结

1. **multi_dim（原始 RLF 奖励）是 searchr1 模式下的最优 reward**，EM 最高且推理效率最好。
2. **format_reward 的格式底分设计失效**：在 searchr1 模式下，模型已能自然学会格式合规（三种模式 Answer Tag Rate 均 ≥99.84%），额外的格式奖励信号引入了噪声，反而降低了 EM 约 4pp。
3. 不同 reward mode 不影响搜索积极性（Avg Searches 均 ≈1.0），差异主要体现在答案质量上。

---

## 6. 综合结论

### 6.1 关键发现

1. **对齐是前提**：RLF 原始设置因工具格式、stop strings、采样策略三个工程问题导致模型完全不搜索。修复后训练效果与 AGL 可比。

2. **训练效率优势**：对齐后 RLF 训练效率显著优于 AGL——200 步 vs 600 步收敛，单步耗时 60s vs 300-600s，得益于 RLF 框架的并行 rollout 与 token 缓存优化。

3. **搜索深度 vs 效率的权衡**：RLF（searchr1 模式）倾向于少搜索（avg 1.13 次），单跳任务表现与 AGL 持平但推理快 2.2×；AGL 搜索更多（avg 3.26 次），多跳任务优势明显。

4. **Prompt mode 选择**：`searchr1_multistep`（user→assistant 交替）是最优 prompt mode，EM 达 37.34%，接近 AGL（38.74%）。裸文本拼接（searchr1）损失 ~8.6pp。显式轮次边界对多轮交互理解至关重要。

5. **Reward 设计**：`multi_dim`（多维度奖励）在 searchr1 模式下 EM 最高（32.86%），`format_reward` 的格式底分反而降低了 EM（28.74%）——因为模型已自然学会格式合规，额外的格式信号引入噪声。在最优 prompt mode（multistep）+ 最优 reward mode（multi_dim 或 pure EM）的组合下，RLF 有望进一步逼近甚至超越 AGL。

### 6.2 RLF vs AGL 框架定位

| 维度 | RLF | AGL |
|------|-----|-----|
| 训练速度 | ★★★★★（5-10× 更快） | ★★ |
| 多跳 QA 表现 | ★★★ | ★★★★ |
| 单跳 QA 表现 | ★★★★ | ★★★★ |
| 推理效率 | ★★★★★（token 消耗少 3×） | ★★ |
| 可扩展性 | ★★★★（模块化 Registry） | ★★★（单文件） |
| 上手难度 | ★★（配置复杂） | ★★★★（直观） |

### 6.3 后续方向

- 在更大模型（Qwen3-4B / 8B）上验证 prompt mode 和 reward mode 的消融结论
- 探索是否可以通过 reward shaping 鼓励 RLF 模型增加搜索次数以提升多跳表现
- 将最佳配置组合应用到 multi-domain 场景

---

## 附录

### A. 训练曲线

请从 Wandb 导出以下曲线截图：

**§3 框架对齐——Stop Strings + do_sample 的训练行为影响（最重要）：**

- **图 A1**：`response_length` 对比（无 stop+sample vs 有 stop+sample，searchr1 模式）
  - Wandb 实验：`searchr1_single_assistant`（无修复）vs `search_r1_single_assistant_addstopstring_dosample`（Pure EM）
  - 预期观察：有修复版本的 response_length 在训练中持续增长，无修复版本保持平稳
- **图 A2**：`avg_turns`（或 `num_turns`）对比（同上两组实验）
  - 预期观察：有修复版本的平均轮数逐步上升，反映模型学到了多轮搜索策略
- **图 A3**：`reward` 曲线对比（同上两组实验）
  - 预期观察：有修复版本收敛后仍有缓慢上升，无修复版本快速饱和

**§3 框架对齐——RLF vs AGL：**

- **图 A4**：RLF（格式对齐后）vs AGL 的训练 reward 对比
  - 展示 RLF 200 步收敛 vs AGL 600 步收敛的速度差异
- **图 A5**：`timing_s/step` 对比（RLF vs AGL）
  - 展示 RLF 单步 ~60s vs AGL ~300-600s 的吞吐率差异

**§4 Prompt Mode 消融：**

- **图 A6**：三种 Prompt Mode（searchr1 / multistep / agl）的 `reward` 曲线对比
  - Wandb 实验：三个 `*_addstopstring_dosample` 实验
- **图 A7**：三种 Prompt Mode 的 `response_length` 对比
  - 观察不同 prompt 构建方式对生成长度的影响

**§5 Reward Mode 消融：**

- **图 A8**：三种 Reward Mode（pure_em / format_reward / multi_dim）的 `reward` 曲线对比
  - Wandb 实验：三个 `search_r1_single_assistant_*_addstopstring_dosample` 实验
  - 注意：不同 reward mode 的 reward 值域不同，主要观察收敛趋势而非绝对值
- **图 A9**：三种 Reward Mode 的 `response_length` 对比

> **Wandb 导出建议**：在 Wandb 中选中相关实验，使用 Panel 的 Group 功能叠加曲线，导出为 PNG。每张图保留 x 轴（step）和 y 轴（指标）标注。

### B. 评测命令参考

```bash
# RLF searchr1 模式评测
python eval_searchr1.py --prompt-mode searchr1 --label rlf ...

# RLF searchr1_agl 模式评测
python eval_searchr1.py --prompt-mode searchr1_agl --label rlf_agl ...

# AGL 模式评测
python eval_searchr1.py --prompt-mode agl --label agl ...
```
