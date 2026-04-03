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

### 3.3 对齐后训练效果

修复上述三个问题后，RLF 训练表现显著改善：

| 指标 | RLF 对齐后 (Step 200) | AGL (Step 600) |
|------|----------------------|---------------|
| 训练 reward | ~0.4（收敛） | ~0.4（收敛） |
| 单步耗时 | **~60s** | ~300–600s |
| response_length | ~700 tokens | — |

对齐后 RLF 仅需 200 步即收敛至与 AGL 600 步相近的 reward 水平，且单步耗时仅为 AGL 的 1/5 — 1/10，体现了 RLF 框架在架构层面的吞吐率优势。

### 3.4 对齐后 5,000 样本评测对比

| 指标 | RLF (Step 200) | AGL (Step 600) | 差异 |
|------|---------------|---------------|------|
| **Overall EM** | **33.62%** | **38.74%** | -5.12pp |
| Avg Turns | 2.19 | 4.27 | RLF 搜索更少 |
| Avg Searches | 1.13 | 3.26 | RLF 搜索更少 |
| Search Rate | 100% | 99.94% | 均充分搜索 |
| Answer Tag Rate | 99.92% | 99.44% | — |
| Avg Time/Sample | **2.41s** | **5.41s** | RLF 快 2.2× |
| Total Tokens | 9.87M | 30.95M | RLF 省 3.1× |

**Per-Source EM 对比：**

| 数据源 | 类型 | RLF (Step 200) | AGL (Step 600) | 差异 |
|--------|------|---------------|---------------|------|
| popqa | 单跳 | 41.53% | 48.58% | -7.05pp |
| triviaqa | 单跳 | **50.41%** | **52.98%** | -2.57pp |
| nq | 单跳 | 38.07% | 37.22% | **+0.85pp** |
| hotpotqa | 多跳 | 24.30% | 33.38% | -9.08pp |
| 2wikimultihopqa | 多跳 | 10.72% | 12.59% | -1.87pp |
| musique | 多跳 | 4.64% | 11.39% | -6.75pp |
| bamboogle | 多跳 | 11.76% | 52.94% | -41.18pp |

**分析：**

1. **单跳数据集**（nq, triviaqa, popqa）：RLF 与 AGL 差距较小（<7pp），在 nq 上甚至略优。这说明一次搜索通常足够，RLF 的低搜索次数不是劣势。
2. **多跳数据集**（hotpotqa, musique, bamboogle）：AGL 明显优于 RLF。AGL 平均 3.26 次搜索使其在需要多步推理的任务上更有优势。
3. **效率**：RLF 推理耗时仅为 AGL 的 45%，token 消耗仅为 32%。
4. **答对样本 vs 答错样本的搜索行为**：

| | RLF 答对 | RLF 答错 | AGL 答对 | AGL 答错 |
|---|---------|---------|---------|---------|
| Avg Turns | 2.09 | 2.24 | 4.04 | 4.41 |
| Avg Searches | 1.08 | 1.15 | 3.04 | 3.40 |

两个框架中答错样本的轮数和搜索次数都略高于答对样本，说明模型在无法找到正确答案时会倾向多搜索几次。

### 3.5 小结

- 对齐修复解决了 RLF 模型不搜索的根本问题，三个 bug 均为工程层面的配置问题
- 对齐后 RLF 在**训练效率**上显著优于 AGL（5-10× 速度提升）
- AGL 在**多跳问答**上表现更好，得益于其更积极的搜索策略（平均 3.26 vs 1.13 次搜索）
- RLF 在**单跳问答**上与 AGL 持平，且推理效率更高

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

> 以下数据需从 Wandb 补充具体数值。三组实验除 `tool_manager` 不同外，其余超参数完全一致。

**训练指标对比：**

| 指标 | searchr1 | searchr1_multistep | searchr1_agl |
|------|----------|-------------------|--------------|
| 收敛步数 | [TODO] | [TODO] | [TODO] |
| 收敛 reward | [TODO] | [TODO] | [TODO] |
| 单步耗时 (s) | [TODO] | [TODO] | [TODO] |
| Avg response_length | [TODO] | [TODO] | [TODO] |

**评测指标对比（收敛后 checkpoint）：**

| 指标 | searchr1 | searchr1_multistep | searchr1_agl |
|------|----------|-------------------|--------------|
| Overall EM | [TODO] | [TODO] | [TODO] |
| Avg Turns | [TODO] | [TODO] | [TODO] |
| Avg Searches | [TODO] | [TODO] | [TODO] |
| Search Rate | [TODO] | [TODO] | [TODO] |

**Per-Source EM（收敛后 checkpoint）：**

| 数据源 | searchr1 | searchr1_multistep | searchr1_agl |
|--------|----------|-------------------|--------------|
| nq | [TODO] | [TODO] | [TODO] |
| triviaqa | [TODO] | [TODO] | [TODO] |
| popqa | [TODO] | [TODO] | [TODO] |
| hotpotqa | [TODO] | [TODO] | [TODO] |
| 2wikimultihopqa | [TODO] | [TODO] | [TODO] |
| musique | [TODO] | [TODO] | [TODO] |

### 4.3 分析

[TODO: 根据 Wandb 数据填充以下分析]

- **收敛速度**：searchr1（flat concat）是否收敛最快？
- **搜索深度**：searchr1_agl（full reprompt）是否搜索更多轮？
- **多跳表现**：多跳数据集（hotpotqa, musique）上哪种模式优势最大？
- **效率**：searchr1_agl 由于每轮需重新编码完整历史，单步耗时是否更长？

### 4.4 小结

[TODO: 根据具体结果补充 2-3 条结论]

---

## 5. 实验三：Reward Mode 对比

### 5.1 三种奖励模式

| 模式 | 设计理念 | 奖励计算 |
|------|---------|---------|
| `agl` (Pure EM) | 最简信号，完全对齐 AGL | 答对 → 1.0，其余 → 0.0 |
| `format_reward` (4-way) | EM + 格式合规底分 | 见下表 |
| `multi_dim` (原始 RLF) | EM + 多维格式奖惩 | EM 基础上叠加标签合法性奖励 |

**`format_reward` 奖励矩阵（λ_f = 0.2）：**

| 条件 | 奖励 |
|------|------|
| EM 正确 + 格式正确 | 1.0 |
| EM 正确 + 格式错误 | 0.8 |
| EM 错误 + 格式正确 | 0.2（底分保证非零梯度） |
| EM 错误 + 格式错误 | 0.0 |

**格式正确的判定**：`<answer>` 标签交替合法（必须），`<search>` / `<think>` 标签交替合法（若存在）。

`format_reward` 的设计动机：Pure EM 下答错样本 reward=0，格式正确与否都没有梯度信号；4-way 设计通过给格式正确但答错的样本 0.2 的底分，确保模型即使答错也能从格式合规中获得学习信号。

### 5.2 实验结果

> 以下数据需从 Wandb 补充。三组实验使用 `searchr1` tool_manager，仅 `reward_mode` 不同。

**训练指标对比：**

| 指标 | agl (Pure EM) | format_reward | multi_dim |
|------|--------------|---------------|-----------|
| 收敛步数 | [TODO] | [TODO] | [TODO] |
| 收敛 reward | [TODO] | [TODO] | [TODO] |
| Answer Tag Rate | [TODO] | [TODO] | [TODO] |

**评测指标对比（收敛后 checkpoint）：**

| 指标 | agl (Pure EM) | format_reward | multi_dim |
|------|--------------|---------------|-----------|
| Overall EM | [TODO] | [TODO] | [TODO] |
| Avg Turns | [TODO] | [TODO] | [TODO] |
| Search Rate | [TODO] | [TODO] | [TODO] |
| Answer Tag Rate | [TODO] | [TODO] | [TODO] |

### 5.3 分析

[TODO: 根据 Wandb 数据填充以下分析]

- **格式学习速度**：format_reward 是否加速了模型学会正确输出 `<answer>` / `<search>` 标签？
- **EM 影响**：格式底分的引入是否影响最终 EM？
- **搜索行为**：不同 reward mode 下模型的搜索积极性是否有差异？

### 5.4 小结

[TODO: 根据具体结果补充 2-3 条结论]

---

## 6. 综合结论

### 6.1 关键发现

1. **对齐是前提**：RLF 原始设置因工具格式、stop strings、采样策略三个工程问题导致模型完全不搜索。修复后训练效果与 AGL 可比。

2. **训练效率优势**：对齐后 RLF 训练效率显著优于 AGL——200 步 vs 600 步收敛，单步耗时 60s vs 300-600s，得益于 RLF 框架的并行 rollout 与 token 缓存优化。

3. **搜索深度 vs 效率的权衡**：RLF（searchr1 模式）倾向于少搜索（avg 1.13 次），单跳任务表现与 AGL 持平但推理快 2.2×；AGL 搜索更多（avg 3.26 次），多跳任务优势明显。

4. **Prompt mode 选择**：[TODO: 根据消融实验结果补充]

5. **Reward 设计**：[TODO: 根据消融实验结果补充]

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

[TODO: 从 Wandb 导出以下曲线截图]

- 图 A1：对齐前后 RLF 训练 reward 对比
- 图 A2：RLF 对齐后 vs AGL 训练 reward 对比
- 图 A3：三种 Prompt Mode 训练 reward 对比
- 图 A4：三种 Reward Mode 训练 reward 对比
- 图 A5：各实验 timing_s/step 对比

### B. 评测命令参考

```bash
# RLF searchr1 模式评测
python eval_searchr1.py --prompt-mode searchr1 --label rlf ...

# RLF searchr1_agl 模式评测
python eval_searchr1.py --prompt-mode searchr1_agl --label rlf_agl ...

# AGL 模式评测
python eval_searchr1.py --prompt-mode agl --label agl ...
```
