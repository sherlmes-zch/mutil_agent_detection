# PrimeVul 多样本人工评测基线

## 目的与状态

本评测集用于建立当前风险路径分析员和保护依据分析员的首轮根因分析基线，不代表 PrimeVul 全测试集上的总体性能。样本来自 `datasets/primevul_test_paired.jsonl`，按固定配对索引选取 10 组，共 20 个函数：每组包含一个漏洞版本和一个配对非漏洞版本。样本选择在查看本次 Agent 输出前固定。

当前完成状态：

- 已固定 10 组、20 条样本，并生成可直接运行的 JSONL 子集。
- 已为每组填写暂定人工根因、源码差异证据、关键触发条件及保护检查点。
- 已尝试使用 Kimi K3 运行 20 条；免费额度耗尽前，6 条样本有完整且可解析的双 Agent 输出，1 条样本的两个回答无法解析，另外 13 条因额度失败。部分运行和暂定评分见 `workspace/manual_eval_baseline_kimi_k3_v2_review.md`。这不是完整 20 条基线，不得外推为总体性能。
- 标注状态为 `provisional_manual_review`。有些触发条件依赖仓库宏、API 或状态语义，正式研究报告前应按下文复核，不能把暂定标注说成双人仲裁结果。

## 样本构成

| 配对索引 | 漏洞样本 idx | 非漏洞样本 idx | CVE | CWE | 项目 | 主要根因类别 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 194963 | 217569 | CVE-2022-32547 | CWE-704 | ImageMagick6 | 浮点指针强转/未对齐访问 |
| 2 | 194994 | 218852 | CVE-2022-23578 | CWE-401 | tensorflow | 错误路径资源泄漏 |
| 7 | 195022 | 219947 | CVE-2021-45379 | CWE-287 | glewlwyd | 认证失败路径仍更新会话 |
| 11 | 195029 | 220201 | CVE-2022-23592 | CWE-125 | tensorflow | 生产构建缺失边界检查 |
| 16 | 195056 | 220841 | CVE-2022-23557 | CWE-369 | tensorflow | 零长度导致模零/错误循环步长 |
| 19 | 195069 | 221079 | CVE-2021-40568 | CWE-120 | gpac | 解析索引缺少下界校验 |
| 26 | 195095 | 221678 | CVE-2021-44273 | CWE-295 | e2guardian | TLS 缺失证书主机名校验 |
| 27 | 195220 | 222666 | CVE-2021-44512 | CWE-362 | tmate-ssh-server | 临时会话目录权限过宽 |
| 30 | 195233 | 222906 | CVE-2022-23566 | CWE-787 | tensorflow | 输出端口索引缺失范围检查 |
| 58 | 195410 | 226433 | CVE-2022-21736 | CWE-476 | tensorflow | 稀疏张量形状前置条件缺失 |

选择覆盖了内存安全、数值边界、资源生命周期、认证授权、TLS 配置和文件权限等多种分析模式；同时保留 10 组 paired 正负样本，便于检查 Agent 是否能区分修复前后的关键差异。由于这是有意分层的小样本集，不能据此估计数据集自然分布下的 Accuracy、Precision 或 Recall。

## 文件

- 冻结运行子集：`datasets/primevul_manual_eval_20.jsonl`
- 逐组人工标注及评分字段：`doc/primevul_manual_eval_annotations.jsonl`
- 可重复生成上述文件：`scripts/build_manual_eval_baseline.py`
- 原数据集 SHA-256 记录在每条标注记录的 `source_dataset_sha256` 字段中。

重新生成并校验：

```bash
PYTHONPATH=src python scripts/build_manual_eval_baseline.py
python - <<'PY'
import json
from collections import Counter
rows = [json.loads(line) for line in open("datasets/primevul_manual_eval_20.jsonl")]
print("samples:", len(rows), "labels:", dict(Counter(row["target"] for row in rows)))
PY
```

## 冻结运行配置

基线定义为当前代码和提示词下的 `function_only` 首轮运行：Qwen `qwen3.8-27b`、temperature `0.1`、completion `8192` tokens、thinking 关闭；Joern 使用 CLI 默认路径和当前默认配置。开始正式运行前应记录代码版本（当前工作区不是可识别的 Git checkout 时，记录文件 SHA-256）、模型配置、运行时间、Joern 版本、依赖版本和上下文模式。不要在这 20 条上调提示词后仍称为同一个基线；改动后应另建版本号并在同一冻结样本上比较。

在配置 `DASHSCOPE_API_KEY` 后运行：

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_manual_eval_20.jsonl \
  --context-mode function_only \
  --limit 20 \
  --max-tokens 8192 \
  --output workspace/manual_eval_baseline_v1.jsonl
```

CLI 默认同时生成 `workspace/manual_eval_baseline_v1.agent_io.json`。保留这两个原始文件，不要只保留人工打分后的汇总。若部分样本失败，应保留失败项并单独报告运行成功率；不能把失败项静默删除后只报告成功样本指标。

## 人工评分规程

先由评审者只看函数源码、配对差异、CVE 描述和标注表完成根因判断，再看 Agent 输出。当前 annotations 是单人暂定标注；建议第二位评审者独立复核全部 10 个漏洞根因，至少复核所有 `confidence=medium` 项和任何有争议项。仲裁后将 `annotation_status` 更新为 `adjudicated`，并记录复核者及依据。

对每个漏洞样本，分别评分两个 Agent：

| 维度 | 0 分 | 1 分 | 2 分 |
| --- | --- | --- | --- |
| 风险根因 | 未识别或归因错误 | 识别到相关操作/类别，但因果解释不完整 | 识别到真实根因及关键对象/安全属性 |
| 位置定位 | 没有有效位置或位置无关 | 锚点在相关代码附近但未指向关键操作 | 锚点指向标注的根因操作/检查 |
| 触发条件 | 错误或臆造 | 部分正确，关键前提遗漏/未知 | 与标注条件一致且没有把未知说成事实 |
| 保护匹配 | 把无关检查说成充分保护，或漏掉关键差异 | 找到相关检查但覆盖范围判断不完整 | 说明保护对象、路径，并正确判断能否阻断该风险 |

非漏洞配对样本不做“必须找出某个保护机制”的惩罚式评分。记录 Agent 是否错误地复述对应漏洞根因、是否把真实修复检查遗漏，以及是否无证据地产生漏洞主张。每条不支持的关键事实另记 `unsupported_claims`，不与锚点格式错误混为一类。

将根因覆盖定义为风险 Agent 的 `root_cause_score == 2`；另报宽松覆盖（分数至少 1），位置命中率、触发条件完全命中率和保护匹配完全命中率。分别在 10 个漏洞样本上报告这四项的计数/分母；正负样本判别结果和运行成功率在 20 条上报告。不要把候选有正确源码锚点等同于根因命中。

建议逐条填写 `primevul_manual_eval_annotations.jsonl` 中的评分字段：

- `agent_risk_root_cause_score`
- `agent_risk_location_score`
- `agent_risk_trigger_score`
- `agent_protection_match_score`
- `unsupported_claims`
- `reviewer_notes`

保护 Agent 对真实风险根因的覆盖并不等同于风险 Agent 命中；它的主要评分是保护机制是否相关、保护属性是否匹配、覆盖范围是否正确。正负样本整体判断仅作为补充观察，因为当前完整仲裁/最终裁决节点尚未接入真实模型处理器。

## 当前可得结果与限制

本次 Kimi K3 的 20 条运行记录和部分复核位于 `workspace/manual_eval_baseline_kimi_k3_v2.agent_io.json`、`workspace/manual_eval_baseline_kimi_k3_v2.jsonl` 和 `workspace/manual_eval_baseline_kimi_k3_v2_review.md`。在已评分的 4 条漏洞样本中，风险 Agent 对 CWE-704 未对齐读取只在 Top-3 命中；对 CWE-401 内存泄漏和 CWE-287 认证失败路径更新会话均未命中；对 CWE-125 生产构建中 `DCHECK` 不提供边界保护命中。保护 Agent 只在最后一例正确指出 debug-only 检查的风险。

正式基线汇总应在 20 条都完成当前配置运行后进行。至少报告：成功/失败数、风险根因严格/宽松覆盖、位置命中、触发条件命中、保护属性匹配、无依据关键主张数，以及逐样本失败分析。该试点评测只能用于迭代诊断，不足以支持总体性能或统计显著性结论。
