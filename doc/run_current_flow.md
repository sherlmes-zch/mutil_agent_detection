# 运行当前 PrimeVul 检测流程

本文说明当前仓库已经实现的流程如何运行。当前 CLI 支持阿里云百炼和新的 OpenAI 兼容平台 `xem8k5`，并调用 Joern 和 LangGraph；真实运行需要所选平台的可用 API Key。

## 1. 已实现的流程

每条 PrimeVul 样本大致经过以下节点：

```text
读取 JSONL 样本
  -> 准备函数上下文或精确仓库版本
  -> Joern 生成/复用 CPG 并提取局部骨架
  -> 风险路径分析员
  -> 保护依据分析员
  -> 分歧诊断与查询编排
  -> 最终报告
```

两个首轮 Agent 使用同一个完整函数、程序生成的关键操作索引和同一份 Joern skeleton，但使用不同的角色 Prompt。完整函数会按声明的行号基准加稳定行号前缀；前缀只是定位元数据，Agent 引用时只返回源码片段和证据来源，不返回行号。Joern skeleton 除调用/控制结构外，还查询 CPG 中的强转、解引用、数组/指针索引、加减乘除/移位及比较节点，并收集函数形参、`recv`/`read` 等输入 API 候选，以及形参到运算节点的候选数据流。每条结构线索带文件、行号和代码，类型信息可用时一并提供。Python 文本索引标出强转、解引用/成员指针访问、数组索引、指针步进和条件所在行，并附短源码行；它和 Joern 结果都只是定位线索，Agent 仍需检查完整函数并独立判断，不能将线索直接当成漏洞结论。

仓库模式下，程序会先确认提供的仓库 `HEAD` 与样本 `commit_id` 一致，再在 `file_name` 中用函数 token 序列定位函数的绝对起止行。匹配唯一时，Agent 的源码锚点、函数范围和关键操作索引均采用该 commit 中的 1-based 绝对行号。若无法验证 commit、文件缺失、函数未找到或匹配不唯一，不会猜测绝对行号：输入会标注映射原因，并退回相对于完整函数片段的行号。函数片段模式始终使用函数内相对行号。

## 2. 环境检查

在项目根目录执行：

```bash
cd /home/zch/kaiti/mutil_agent_detection
export PYTHONPATH="$PWD/src"
```

确认 Python 依赖已经安装：

```bash
python -m pytest -q
python -m compileall -q src tests
```

当前测试应全部通过。没有 API Key 时仍可运行这些测试，但不能运行真实 CLI。

确认 Joern：

```bash
/home/zch/joern/joern-cli/joern --help
/home/zch/joern/joern-cli/joern-parse --help
```

默认配置使用：

```text
joern:       /home/zch/joern/joern-cli/joern
joern-parse: /home/zch/joern/joern-cli/joern-parse
```

也可以在 CLI 中用 `--joern` 指定其他 Joern 入口。若 `joern-parse` 不在同一目录，需要后续通过配置指定解析器路径。

## 3. 选择模型 API 平台

CLI 通过 `--api-platform` 选择模型平台：

```text
bailian  阿里云百炼（默认）
xem8k5   https://ai.xem8k5.top/v1 OpenAI 兼容接口
```

`--base-url` 可以覆盖所选平台的默认地址，`--api-key-env` 可以覆盖默认的密钥环境变量名。密钥只从环境变量读取，不会写入代码、数据集或实验记录。

### 3.1 阿里云百炼 API Key

推荐使用环境变量，不要把密钥写入代码或数据文件：

```bash
export DASHSCOPE_API_KEY='你的百炼API密钥'
```

CLI 默认读取 `DASHSCOPE_API_KEY`。也可以指定其他环境变量名：

```bash
export MY_BAILIAN_KEY='你的百炼API密钥'
```

然后运行时增加：

```bash
--api-key-env MY_BAILIAN_KEY
```

默认模型和接口地址为：

```text
model:    qwen3.8-27b
base-url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

默认每个 Agent 最多生成 8192 个 token，并关闭 Qwen thinking 模式，避免推理过程耗尽结构化答案的输出额度。可通过 `--max-tokens` 调整上限；只有确实需要思考模式时才使用 `--enable-thinking`。可通过 `--model` 和 `--base-url` 覆盖模型和接口地址。

### 3.2 xem8k5 OpenAI 兼容平台

新平台对应的请求地址是 `/v1/chat/completions`，CLI 的 `--base-url` 应填写上一级 `/v1` 地址：

```text
base-url: https://ai.xem8k5.top/v1
```

默认读取 `XEM8K5_API_KEY`，默认模型为 `deepseek-v4.1-flash`：

```bash
export XEM8K5_API_KEY='你的新平台API密钥'
```

如果使用其他环境变量名，运行时加上 `--api-key-env ENV_NAME`。新平台不发送百炼专用的 `enable_thinking` 参数；模型是否支持 LangChain 的结构化输出，需要以该平台的 OpenAI 兼容实现为准。

## 4. 推荐的第一次运行：函数片段模式

函数片段模式不会自动克隆仓库，只使用 PrimeVul 的 `func`、文件元数据和函数级 Joern 输入。先用 paired test 数据集中的少量样本：

```bash
PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_test_paired.jsonl \
  --context-mode function_only \
  --limit 2 \
  --output workspace/test_function_only.jsonl

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_test_paired.jsonl \
  --model qwen3.8-max-0902 \
  --context-mode function_only \
  --repository-source-cache .cache/primevul_source_repositories \
  --repository-cache .cache/primevul_repositories \
  --limit 1 \
  --max-tokens 8192 \
  --output workspace/first_sample_full_flow.jsonl \
  --agent-trace-output workspace/first_sample_full_flow.agent_io.json
```

也可以使用未配对测试集：

```bash
PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_test.jsonl \
  --context-mode function_only \
  --limit 2 \
  --output workspace/test_function_only.jsonl
```

运行结束后，终端会打印类似以下摘要：

```json
{
  "dataset": "primevul_test_paired",
  "requested": 2,
  "completed": 2,
  "failed": 0,
  "predictions": {"unknown": 2},
  "output": "workspace/test_function_only.jsonl"
}
```

实际预测结果取决于模型调用和后续编排，不应预期所有样本都得到确定的 vulnerable 或 benign 结论。当前系统对证据不足的样本会保留 `unknown`。

### 4.1 xem8k5 平台批量运行（limit 10+）

下面命令使用新平台、Balanced 判定策略运行前 10 条样本，并把主结果和 Agent 专用记录写到新的文件：

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_test_paired.jsonl \
  --api-platform xem8k5 \
  --model deepseek-v4.1-flash \
  --decision-policy balanced \
  --context-mode function_only \
  --max-query-rounds 2 \
  --repository-source-cache .cache/primevul_source_repositories \
  --repository-cache .cache/primevul_repositories \
  --limit 10 \
  --max-tokens 8192 \
  --agent-retries 1 \
  --output workspace/xem8k5_limit10_balanced.jsonl \
  --agent-trace-output workspace/xem8k5_limit10_balanced.agent_io.json
```

建议先用 `--limit 1` 做连通性检查，再扩大到 10 或更多；如平台提供不同模型，只需替换 `--model`，如提供不同的兼容地址，再增加 `--base-url`。

`--agent-retries` 对首轮分析、分歧诊断、查询规划和事实调查的模型失败做有限重试，默认命令行值为 1；重试不会改变候选的程序分配 ID。查询执行器会按“后端、查询类型、问题、函数和仓库快照”缓存成功的证据，避免相同查询重复运行 Joern 或源码扫描。

`--max-query-rounds` 控制每条样本最多执行多少轮“查询执行 -> 事实调查”。当前默认值为 2。第一轮即使成功回答了一个 premise，只要诊断器仍留下决定性未决 premise，工作流会继续规划下一轮；如果所有决定性 premise 已解决，则提前结束。若只需要一次事实调查，可设置 `--max-query-rounds 1`。

命令行默认在每条样本完成后打印一行进度 JSON，首字段 `progress` 使用 `1/4`、`2/4` 等格式，随后包含 `idx`、`sample_id`、`target`、`prediction`、置信度、证据等级和简短原因。批量日志只需要最终摘要时，可增加 `--quiet-progress`。

当前候选在进入诊断器前统一为 `risk-1`、`risk-2`、`risk-3`，保护项统一为 `protection-1` 等。模型偶尔返回的临时 `claim_*`/旧 ID 会被安全映射或丢弃，不会让其他有效关系整体失效。Balanced 不会仅凭“源码锚点存在且未看到保护”判定漏洞：除结构化锚点和属性不匹配外，还要求查询证据支持该候选，或要求类型化访问/对齐冲突能够由地址来源、访问类型和对齐要求等已验证源码事实直接建立。Benign 必须逐候选获得证据反驳，或由同一对象、同一安全属性且完整覆盖路径的有效保护覆盖。Strict 仍要求决定性 premise 被查询证据完整支持。

风险候选同时标记为 `direct_violation`、`conditional_interprocedural` 或 `speculative`。首轮操作索引和 Joern skeleton 会显式列出整数窄化转换、源表达式、目标类型、符号/宽度与表示范围检查要求。仓库 SOURCE 查询除相关匹配行外，还返回完整宏定义、符号所在函数或调用点局部函数切片。查询规划优先处理直接风险关联的决定性 premise，并禁止下一轮仅换措辞重复相同 premise/backend/kind；事实调查器会裁剪超长锚点引用并归一化常见兼容字段。Balanced 的 benign 采用候选级闭合：直接风险必须被反驳或被同属性完整保护，跨过程/探索性候选必须至少经过证据调查且必要条件未被建立，未查询候选不能被静默忽略。

## 5. 仓库上下文模式

仓库模式会根据每条样本的 `project_url` 和 `commit_id`：

1. 复用已有项目缓存；
2. 必要时自动克隆或获取远程仓库；
3. 为目标 commit 创建 detached worktree；
4. 将该 worktree 交给 Joern；
5. 再校验 worktree 的 `HEAD` 与样本提交号一致。

使用命令：

```bash
PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_test_paired.jsonl \
  --context-mode repository \
  --limit 2 \
  --repository-cache .cache/primevul_repositories \
  --output workspace/test_repository.jsonl
```

默认仓库缓存目录为：

```text
.cache/primevul_repositories
```

如果已经准备了本地源码缓存，可以指定：

```bash
PYTHONPATH=src python -m mutil_agent_detection.cli \
  datasets/primevul_test_paired.jsonl \
  --context-mode repository \
  --repository-source-cache /path/to/source-cache \
  --limit 2
```

本地源码缓存目前按以下目录形式查找：

```text
/path/to/source-cache/PROJECT/
/path/to/source-cache/PROJECT/COMMIT/
```

默认只允许 HTTPS GitHub 地址。仓库没有 `project_url`、没有有效 `commit_id`、远程不可访问或提交不存在时，该样本会失败并在报告中记录错误，不会静默退回 `function_only`。

## 6. 结果文件

实验记录只保留样本标识和工作流中各 Agent 的输入/输出，不再重复保存最终报告、完整锚点校验明细、原始模型响应和常规重试元数据；若发生定向锚点修正，会额外记录修正反馈和修正后的结构化回答。使用 `.jsonl` 后缀时，每行对应一条样本：

```json
{
  "sample_id": "...",
  "idx": 123,
  "workflow": [
    {
      "agent": "risk_analyst",
      "status": "parsed",
      "input": {
        "task": "...",
        "response_schema": "RiskAnalysisDraft",
        "sample": {"function_name": "...", "file_name": "..."},
        "source": "...",
        "joern_skeleton": {}
      },
      "output": {"summary": "..."}
    }
  ],
  "error": null
}
```

每个 `workflow` 项按执行顺序对应一个 Agent。`input` 展示角色任务、schema、样本定位、函数源码和 Joern skeleton；`output` 是解析后的结构化回答。失败时 `output` 为 `null`，并保留简短的 `error` 或 `fallback` 状态。

需要直接人工阅读时，将输出路径命名为 `.json`，CLI 会写成带缩进的 JSON 文档；`.jsonl` 继续用于逐样本流式处理。例如把命令中的 `workspace/test_function_only.jsonl` 改为 `workspace/test_function_only.json`。

除保持原有 `--output` 文件结构外，CLI 会额外生成 Agent 专用输入/输出记录，默认命名为 `workspace/test_function_only.agent_io.json`，采用缩进 JSON 便于阅读。其中每个 Agent 项按调用启动顺序记录 `order`、`agent`、`input`、`output` 和 `status`；失败时附 `error`/`fallback`。可用 `--agent-trace-output PATH` 自定义专用记录路径。该文件不包含最终报告或内部校验明细。

### 结构化响应被截断

若 Agent 的 `status` 为 `failed`，查看对应记录的 `error`。`workflow` 不保存原始响应和逐次重试详情，以保持实验记录聚焦于 Agent 输入/输出；详细诊断需在程序日志或调试模式中查看。

当前 CLI 输出聚焦于 Agent 输入/输出。完整的 LangGraph 审计状态仍不写入实验记录。

两个首轮 Agent 使用不同的严格输出 schema。源码锚点由 Agent 提供精确代码片段和来源类型，程序在原始函数或 Joern skeleton 中解析文件与行号，不采信 Agent 自报行号。若关键片段不存在或定位不唯一，程序会追加一次定向修正；修正后仍不匹配的风险候选会标记 `evidence_status=unverified`，其置信度不进入 Claim，相关无效锚点也不会支持状态为 supported 的前提。此校验验证的是片段及位置存在，不等于语义证明。

风险 Agent 会先扫描全函数，再按证据强度和可能影响输出最多 3 个彼此不同、可证伪的风险候选；不要求凑满数量。每个候选独立携带危险操作、对象、路径、触发前提、源码锚点和置信度。为兼容当前后续诊断/查询节点，候选 1 还会投影到 `Claim.analysis` 的旧顶层风险字段，完整候选列表仍保存在 `Claim.analysis.candidates`。旧版单候选模型响应仍可解析。

实验记录里的 Agent `input` 会包含行号基准和关键操作索引，便于对照源码审阅；函数源码仍完整提供。`output` 是 schema 解析后的完整 Agent 回答（风险 Agent 包括 `candidates`）；运行时仍会逐候选执行源码锚点校验，并将所有候选的触发前提纳入 Claim 前提集合。保护覆盖校验照旧执行。这些内部校验明细不写入 Agent I/O 记录文件。

## 7. 调试和常见错误

### API Key 缺失

错误：

```text
Missing API key. Export DASHSCOPE_API_KEY before making model calls.
```

处理：

```bash
export DASHSCOPE_API_KEY='你的百炼API密钥'
```

使用 xem8k5 时，错误提示中的环境变量应为 `XEM8K5_API_KEY`：

```bash
export XEM8K5_API_KEY='你的新平台API密钥'
```

如果密钥曾经直接出现在聊天记录、终端历史或日志中，应先在平台侧撤销并重新生成。

### 数据集路径错误

确认文件存在：

```bash
ls -l datasets/primevul_test.jsonl datasets/primevul_test_paired.jsonl
```

### Joern 不可用

确认两个程序都存在：

```bash
test -x /home/zch/joern/joern-cli/joern
test -x /home/zch/joern/joern-cli/joern-parse
```

如果只想先验证 Python 和 LangGraph 结构，不需要启动真实 CLI，执行：

```bash
python -m pytest -q
```

### 仓库提交不匹配

不要手动切换正在使用的源码仓库分支。清理或补充本地源码缓存后重新运行；系统会为目标 commit 创建独立 worktree。

## 8. 推荐执行顺序

```text
1. pytest 验证代码和协议
2. function_only + paired 数据集 + --limit 1/2
3. 检查 JSONL 中的 report 和 error
4. repository + paired 数据集 + --limit 1
5. 确认 Joern、仓库缓存和提交校验正常
6. 扩大 limit，再运行完整 test 数据集
```

不要一开始直接对完整数据集启用仓库模式：首次运行可能触发多个项目的网络下载、提交获取、CPG 构建和模型调用。
