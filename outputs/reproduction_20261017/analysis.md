# SkillOpt SearchQA: 两次训练归档与分析

## 1. 归档内容和实验配置

两个完整训练输出目录已复制到本归档，每个包含运行时完整配置、40 步 history、逐步记录、trajectory digest、评估输出和所有 Skill 版本。额外保存了初始/最佳/最终 Skill 快照、数据 split manifest、模型快照 commit、权重分片 SHA-256 元信息和结构化 `training_log.csv`（由 history.json 展开）。原始终端 stdout 未保存在训练目录中；没有伪造控制台日志。准确配置以各实验的 `complete_run_output/config.json` 为准。第二次运行使用的 YAML 也已复制。

| 实验 | Optimizer | Target | 精度/量化 | 训练设置 |
|---|---|---|---|---|
| 1 | Meta-Llama-3-8B-Instruct | Qwen2.5-7B-Instruct | 两模型 BF16，无量化 | 400 训练样本，batch 40，4 epoch，seed 42 |
| 2 | Qwen2.5-32B-Instruct | Qwen2.5-7B-Instruct | optimizer NF4 4-bit、double quant、BF16 compute；target BF16 | 相同的训练样本数、batch、epoch、seed 和 split |

模型的 HF snapshot commits、配置架构信息、分片大小/哈希见 `model_versions.json`。模型权重仍保留在原本地路径，没有再次复制。

## 2. Qwen2.5-32B 优化器的规则变化

Step 20 是最佳 Skill（`best_skill.md` 指向 Step 20）；Final 是第 4 个 epoch 后的 `skill_v0040.md`。两者 Slow Update 保护区之前的规则正文相同。Final 新增的 Slow Update 内容主要重申：按问题要求的具体程度返回实体；避免额外限定词和解释；核对格式并选择上下文支持的最短形式。它补充三个轨迹中的纠错例子：`Brunei` 而不是 `Hassanal Bolkiah`、`Angela Bassett` 而不是一串电影名、`the manatee` 而不是 `dugong`。这些具体例子没有构成新的通用推理策略。

Step 20 已经形成的主要规则是：从上下文抽取精确实体；匹配问题的具体程度；优先精确、简短的答案；不加多余描述；提交前对照问题和上下文检查，并使用 `<answer>...</answer>` 格式。Skill 中多处标题和句子反复表达同一组原则。Final 的例子更具体，但也增加重复。Final 原始文件包含重复嵌套的 `SLOW_UPDATE_START` / `SLOW_UPDATE_END` 标记；Slow Update 正文完整，原始文件已保留用于实验审计。检查发现旧解析器使用首个结束标记，会在区块内层结束标记之后、外层结束标记之前存在文本时错误缩小保护范围；本次实际 Final 的这段间隙为空，所以已归档内容没有发生规则泄漏。

## 3. Candidate 提案与门控分布

| 统计 | 实验 1：Llama3-8B | 实验 2：Qwen2.5-32B 4-bit |
|---|---:|---:|
| 有 patch 的步骤 | 22/40 | 40/40 |
| Reflection patches（失败 + 成功） | 23（10 + 13） | 206（80 + 126） |
| 合并 edits | 44 | 211 |
| 排名 edits | 44 | 118 |
| Accept | 2 | 5 |
| Reject | 20 | 35 |
| Skip（无 patch） | 18 | 0 |
| 在进入 gate 的候选步骤中 Accept 比例 | 9.1% | 12.5% |

第二次提案产出明显改善：40 步都有 patch（第一次只有 22 步），生成 206 个 patch（第一次 23 个，约 9.0 倍），Skip 从 18 降到 0。Accept 从 2 增至 5。Reject 的绝对数量从 20 增至 35，是因为更多步骤产生候选并进入 gate；按进入 gate 的候选步骤计算，Accept 比例仅从 9.1% 小幅升至 12.5%，仍有多数候选被拒。结论是提案供给和可用候选数改善，门控通过率只小幅改善；每种配置仅一次运行，不能单独归因于模型规模。

第二次训练的初始/Step20/Final test Hard 分别为 60.786%、62.214%、62.786%；Soft 为 0.6991、0.7088、0.7197。这 1,400 道题与之前 fixed-Skill 复测测试集完全相同，因此属于描述性结果，不是新的锁定测试集验证。


## 5. Slow Update 标记解析核查与迁移准备

重复嵌套标记不只是格式冗余：旧实现的字段提取会把内层开始标记当作文本传给优化器；保护范围识别和语义密度统计则只匹配第一个结束标记，未来若嵌套区块尾部出现内容，该内容可能被当成普通 Skill 文本。对本次 Final 检查确认，内层结束与外层结束之间没有正文，因此当前 1,672 字符的 Slow Update 规则均仍位于保护区。

已将标记匹配改为按嵌套层级配对，覆盖 Slow Update 提取/替换、逐步编辑的受保护范围和 semantic-density 门控；替换时会清理新内容中误带的结构标记。原始 `final_skill_v0040.md` 保持不变，另生成 `experiment_2_qwen25_32b_nf4_bf16/skill_snapshots/final_skill_v0040_for_benchmark.md`，其中规则正文不变，标记规范为单一平衡的开始/结束标记。对嵌套尾部保护、受保护文本不计入语义密度、字段提取和替换后只保留一对标记做了针对性检查，并对修改的 Python 文件执行了 compileall。
