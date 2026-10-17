# SkillOpt SearchQA 复现与补丁记录

本仓库记录基于本地 Transformers 模型运行 SkillOpt SearchQA 的复现，以及后续的嵌套 Slow Update 标记解析修复。模型权重不包含在仓库中。

## 复现实验

两次训练均使用 Qwen2.5-7B-Instruct 作为 Target，训练集 400 条、batch size 40、4 个 epoch、seed 42（共 40 步）。Optimizer 配置不同：

| 实验 | Optimizer | 精度 | 初始 Test Hard | 最终 Test Hard | 最终 Test Soft | Accept / Reject / Skip |
|---|---|---|---:|---:|---:|---:|
| 1 | Llama-3-8B-Instruct | BF16 | 61.00% | 60.57% | 0.6932 | 2 / 20 / 18 |
| 2 | Qwen2.5-32B-Instruct | NF4 4-bit，double quant，BF16 compute | 60.79% | 62.79% | 0.7197 | 5 / 35 / 0 |

实验 2 的最佳 Skill 出现在 Step 20；Final 指标来自第 4 个 epoch 后的 Skill。两次运行使用的 Test 集此前已用于固定 Skill 重复评测，因此这些分数是描述性结果，不是独立锁定测试集上的最终验证。

配置、训练 history/CSV、Skill 版本快照、模型版本元数据及分析见 [`outputs/reproduction_20261017/`](outputs/reproduction_20261017/)。完整逐样本轨迹保留在本地 `outputs/experiment_archive_20261017/`，未纳入此仓库材料包。

## Slow Update 解析补丁

第二次训练生成的 Final Skill 含重复嵌套的 `SLOW_UPDATE_START` / `SLOW_UPDATE_END` 标记。旧解析器只匹配第一对边界，可能把内层结束标记之后的内容误判为未保护文本。补丁改为按嵌套深度匹配，并覆盖 Slow Update 提取/替换、编辑保护范围和 semantic-density 门控。原始实验快照保持不变；Benchmark 使用的规范化快照也保存在复现材料中。

回归测试共 5 项，覆盖嵌套范围、尾部保护、门控排除、多个区块和不闭合标记：

```bash
python -m unittest tests.test_nested_slow_update_markers -v
```

补丁细节见 [`docs/nested_slow_update_fix.md`](docs/nested_slow_update_fix.md)。
