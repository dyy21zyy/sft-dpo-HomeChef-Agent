# HomeChef Booking Agent 小模型后训练与本地部署技术规范

## 1. 项目背景

本项目面向上门私厨预约场景。Booking Agent 中存在一组高频、边界明确、协议要求严格的模型任务：用户预约意图理解、槽位抽取、多轮状态继承、相对时间解析、候选厨师处理、用户确认、Tool Calling 决策和 Tool Result 理解。

这些任务不负责开放式咨询、真实检索、真实数据库写入或支付流程。项目目标是把预约决策能力迁移到可本地部署的小模型中，降低对远程大模型的依赖，并建立后训练、评估、量化和本地部署链路。

## 2. 项目目标

项目覆盖以下工程目标：

- 以 `BOOKING_MACHINE_CONTRACT_v1.md`、`contracts/`、Pydantic schema 和 JSON Schema 固定机器合同。
- 使用 Frozen Evaluation、Diagnostic Dev、manifest 和 hash 约束评估集。
- 建立 Base Model Benchmark、Structured / Unstructured 推理评估和确定性 scorer。
- 从 Raw 样本派生 SFT 与 DPO 数据，并执行 contamination 检查和数据 manifest 记录。
- 使用 LLaMA-Factory、PEFT / LoRA 进行 SFT 与 DPO。
- 支持 LoRA merge、HF merged model、F16 GGUF、imatrix、Q4_K_M 与 llama.cpp CPU Serving。

## 3. 非目标

当前仓库不是完整的 HomeChef 多 Agent 系统，不负责真实支付、真实订单创建、真实数据库生产写入、Chef RAG 服务实现、Supervisor ReAct、长期记忆或开放式菜谱咨询。

仓库不保存模型权重、真实 LoRA Adapter、训练 checkpoint、GGUF、实验结果、benchmark 输出、运行日志或本机缓存。

## 4. Booking Machine Contract

Booking Machine Contract 是模型输入输出的唯一机器边界。核心输入为 `history`、`current_state`、`user_input`、`current_time` 和 `available_tools`。模型输出只能是 `ToolCallDecision` 或 `FinalDecision`。

合同由以下文件共同约束：

- `BOOKING_MACHINE_CONTRACT_v1.md`：人类可读的冻结合同。
- `contracts/booking_machine_contract_v1.schema.json`：Runtime Input、Decision、Tool Call、Final 等 JSON Schema。
- `contracts/find_chefs_v1.schema.json`：`find_chefs` Tool Result schema。
- `contracts/contract_manifest.yaml`：合同源文件与 schema 的 manifest。
- `src/homechef_booking/schemas/`：Pydantic 类型。
- `src/homechef_booking/validation/contract_validator.py`：合同校验 CLI 与业务规则校验。

合同固定了 slot、状态、缺失信息、Tool Fact 来源、候选顺序、确认条件和 out-of-scope 行为。模型不得自行编造 Tool 才能提供的厨师事实。

## 5. 总体架构

```mermaid
flowchart LR
  A["业务合同"] --> B["Frozen Evaluation"]
  B --> C["Base Benchmark"]
  C --> D["Data Pipeline"]
  D --> E["SFT"]
  E --> F["DPO"]
  F --> G["Evaluation"]
  G --> H["LoRA Merge"]
  H --> I["F16 GGUF"]
  I --> J["imatrix"]
  J --> K["Q4_K_M"]
  K --> L["llama.cpp"]
  L --> M["Local Agent Backend"]
```

代码实现分布在 `src/homechef_booking/`。数据构建脚本位于 `scripts/data/`，评估脚本位于 `scripts/eval/`，训练脚本位于 `scripts/train/`，GGUF 准备脚本位于 `scripts/models/` 和根目录 Phase05 脚本。

## 6. 任务体系

当前任务体系来自合同、schema、scorer、生成器和测试用例，主要包括：

- missing required slots：`service_date`、`start_time`、`people`、`address` 不完整时追问。
- semantic slot extraction：抽取菜系、菜单、预算、忌口、场景、采购责任等业务 slot。
- relative time：依据 `current_time` 规范化今天、明天、后天、周几等表达。
- multi-turn / state inheritance：保留未被用户修改的有效状态。
- state invalidation：查询依赖字段变化后清空旧 Tool Fact。
- tool call：信息完整且缺少实时厨师事实时调用 `find_chefs`。
- tool result：消费 matched、available、unavailable、not_found、no_match、out_of_service_area、error 等结果。
- confirmation：区分候选选择与最终授权预约。
- candidate selection / order：保留 Tool 返回的候选顺序并按用户选择绑定 `chef_id`。
- unrelated input：非预约请求进入 handoff。

## 7. Evaluation First

Frozen Test 用于正式模型选择，冻结后不得进入训练集。`data/eval/frozen_test.manifest.json` 保存冻结集 manifest，相关代码通过 SHA256、case 数、原始输出和 scorer 结果建立可追溯评估。

评估指标由代码定义并分开记录：

- Protocol：JSON 可解析、字段、枚举、union、类型和多余字段检查。
- Task Correctness：Tool Call 参数、Final 状态、候选、缺失信息、reply_type 和语义回复检查。
- Effective Pass：Protocol 通过、任务正确性达到阈值且无 critical error。
- Critical Error：厨师事实编造、错误授权、缺必需 slot 却调用工具、候选顺序破坏、忌口丢失等高风险错误。
- 性能指标：latency、TTFT、tokens per second 等只作为评估输出，不写入公开文档结论。

训练数据必须通过 contamination 检查，避免 Frozen Test 与 Diagnostic Dev 污染训练。

## 8. 数据流水线

数据实现位于 `src/homechef_booking/data/` 与 `scripts/data/`。Raw 样本是事实源，SFT 与 DPO 数据从 Raw 确定性派生。

```text
Raw
→ validation
→ contamination check
→ SFT rendering
→ DPO pair generation
→ manifest / data card
```

`scripts/data/validate_dataset.py` 校验 Raw schema、重复样本、train/val overlap、SFT/DPO 重复、Frozen/Diagnostic contamination，并写出 manifest 与 data card。v0.3 正式训练数据位于 `data/processed/sft/v0.3/` 与 `data/processed/dpo/v0.3/`，配置文件直接引用这些路径。

## 9. Base Model Benchmark

Base Benchmark 先评估未后训练模型在同一合同和同一 Frozen Test 上的表现，防止训练前没有可比较基线。配置位于 `configs/evaluation/` 和 `configs/inference/`，执行入口包括 `homechef-eval`、`scripts/eval/run_base_benchmark.py`、`scripts/eval/run_all_base_benchmarks.py` 与 Phase02 相关脚本。

Structured 与 Unstructured 是推理模式差异，不是不同任务。两者需要在相同合同和相同样例下分别评估。

## 10. SFT

SFT 使用 LoRA，目标是学习协议、业务格式、状态继承、slot 标准化、Tool Call 格式、Tool Result 消费和业务边界。

正式配置位于 `configs/training/phase04_sft_qwen3_1_7b.yaml` 和 `configs/training/phase04_sft_qwen3_4b.yaml`。核心字段来自配置：`stage: sft`、`experiment_class: formal`、`finetuning_type: lora`、`lora_target: all`、`lora_rank: 16`、`lora_alpha: 32`、`lora_dropout: 0.05`、`cutoff_len: 2048`、`learning_rate: 1.0e-4`、`num_train_epochs: 3`、batch size 为训练与评估各 2、`gradient_accumulation_steps: 8`、`lr_scheduler_type: cosine`、`warmup_ratio: 0.1`、`bf16: true`。

1.7B 配置使用 `template: qwen3` 和 `enable_thinking: false`。4B Instruct-2507 配置使用 `template: qwen3_nothink`。训练输出目录位于 `experiments/phase04/`，该目录下的 checkpoint 和 adapter 不进入 Git。

## 11. DPO

DPO 使用 chosen/rejected 偏好对齐，必须从对应 SFT adapter/checkpoint 继续训练，不能从 Base 直接开始。`src/homechef_booking/training/formal_matrix.py` 与 `validate_dpo_source_checkpoint` 对该边界执行 fail-closed 校验。

正式 DPO 配置位于 `configs/training/phase04_dpo_qwen3_*_beta_*.yaml`。核心字段包括 `stage: dpo`、`experiment_class: formal`、`pref_loss: sigmoid`、`pref_beta`、`learning_rate: 5.0e-6`、`num_train_epochs: 1`、训练与评估 batch size 各 1、`gradient_accumulation_steps: 8`、`bf16: true`。DPO 数据来自 `data/processed/dpo/v0.3/`。

DPO 是否改善必须通过同一 Frozen Evaluation、Diagnostic Dev 和失败归因判断，公开文档不记录实测数值。

## 12. Structured / Unstructured 推理

Unstructured 模式让模型自由生成文本，随后由 Protocol scorer 和 Task Correctness scorer 判断是否合法。

Structured 模式在 HF backend 中使用 `lm-format-enforcer` 和 Decision schema 构建 token 约束，提升输出结构约束。实现位于 `src/homechef_booking/inference/structured_output.py` 与 `src/homechef_booking/inference/phase04_hf_backend.py`。

两种模式共享同一 base model 与 adapter；差异只在运行时 schema 约束，不能混为一个评估结果。

## 13. 模型合并与 GGUF

Phase05 的合并路线：

```text
LoRA Adapter
→ PEFT merge_and_unload
→ merged HF model
→ convert_hf_to_gguf
→ F16 GGUF
```

`phase05_merge.py` 执行 PEFT merge。`scripts/models/prepare_phase02_base_gguf.ps1` 展示 HF snapshot、GGUF 转换和 llama.cpp quantize 的 base model 准备流程。`deployment/llama_cpp/` 只保留仓库需要的轻量脚本边界；本地 llama.cpp 二进制和下载源码不进入 Git。

## 14. imatrix + Q4_K_M

imatrix 用训练侧或校准侧语料生成量化校准信息，校准语料不得来自 Frozen Test。`phase05_build_calibration.py` 从 SFT 数据生成本地 calibration 文本，输出到 `data/calibration/`，该目录不纳入 Git。

量化是 PTQ / weight-only 过程。Q4_K_M 用于 CPU 部署候选，相关 GGUF 文件保存在 `models/gguf/`，不纳入 Git。

## 15. llama.cpp 本地 CPU 部署

llama.cpp 本地部署采用 OpenAI-compatible API：

```text
POST /v1/chat/completions
```

运行配置位于 `configs/runtime/llama_cpp_cpu.yaml` 和 `configs/inference/*llama_cpp*.yaml`，关键字段包括 `host`、`port`、`ctx_size`、`threads`、`gpu_layers: 0`、`base_url`、`model_format: gguf`、`quantization`、`max_model_length`、`max_new_tokens`、`temperature: 0`。

`src/homechef_booking/inference/llama_cpp_backend.py` 封装 llama.cpp server 调用。`phase05_cpu_benchmark.py` 可调用本地 `127.0.0.1:8080` 的 OpenAI-compatible 服务并把本地报告写入 `reports/generated/phase05/`，该输出目录不纳入 Git。

## 16. 评估与 Gate

当前代码支持以下 Gate：

- component：schema、manifest、prompt、dataset adapter、backend、training config 的单元测试。
- protocol：原始模型输出必须是合法 Decision JSON。
- task：Tool Call 与 Final Decision 的业务字段校验。
- effective：Protocol、Task Correctness 和 Critical Error 的组合判定。
- critical：高风险错误直接失败。
- performance：latency、TTFT、throughput 等性能字段在 evaluation runner 和 benchmark 脚本中记录。

评估遵循 Fail Closed：adapter 缺失不回退 Base，structured constraint 初始化失败不回退 unstructured，generation error 或空输出不写成成功结果。

## 17. 项目目录结构

```text
configs/               evaluation、inference、training、runtime、model 配置
contracts/             Booking Machine Contract JSON Schema 与 manifest
data/dev/              Diagnostic Dev
data/eval/             Frozen Test 与 manifest
data/raw/              Raw 合成样本
data/processed/        SFT / DPO 训练数据、manifest 与 data card
deployment/            llama.cpp 相关轻量脚本与本地部署边界
experiments/           本地训练输出目录，Git 默认排除运行产物
project-log/           审批与阶段记录
reports/               本地评估和 benchmark 输出目录，Git 默认排除 generated
scripts/data/          数据构建、校验、冻结完整性脚本
scripts/eval/          Base Benchmark、Phase04 matrix、llama.cpp smoke 脚本
scripts/models/        GGUF 准备脚本
scripts/train/         训练配置渲染、dry-run、云端制品打包脚本
src/homechef_booking/  Python 包源码
tests/                 fixtures、unit、integration 测试
```

## 18. 配置说明

`configs/evaluation/` 保存 `homechef-eval` 和矩阵脚本使用的评估配置。`configs/inference/` 保存 mock、HF、Phase04 HF、llama.cpp server 等 backend 配置。`configs/training/` 保存 SFT/DPO 训练配置和 dataset registry。`configs/runtime/` 保存本地 llama.cpp runtime 参数。`configs/models/` 保存 base model 准备所需的模型元数据。

配置中不得依赖用户机器绝对路径。模型、adapter、GGUF 和报告路径保持仓库相对路径或由调用者在本地环境中指定。

## 19. 运行流程

安装与基础校验：

```powershell
uv run ruff check .
uv run pytest -q
uv run homechef-contract-validate --root . --fixtures tests/fixtures/contracts
```

运行 evaluation：

```powershell
uv run homechef-eval --config configs/evaluation/phase01_mock.yaml --output-dir reports/generated/smoke
```

数据校验：

```powershell
uv run python scripts/data/validate_dataset.py --raw data/raw/phase03_smoke_v0.1.jsonl --sft-train data/processed/phase03_smoke_sft_train.jsonl --sft-val data/processed/phase03_smoke_sft_val.jsonl --dpo-train data/processed/phase03_smoke_dpo_train.jsonl --dpo-val data/processed/phase03_smoke_dpo_val.jsonl --frozen data/eval/frozen_test.jsonl --diagnostic data/dev/diagnostic_dev.jsonl --manifest-out data/processed/phase03_smoke_manifest.v0.1.json --data-card-out data/processed/phase03_smoke_data_card.v0.1.json
```

渲染训练配置：

```powershell
uv run python scripts/train/render_config.py --config configs/training/phase04_sft_qwen3_1_7b.yaml --output experiments/phase04/configs/phase04_sft_qwen3_1_7b.resolved.yaml
```

审批门控 dry-run：

```powershell
uv run python scripts/train/dryrun.py --stage sft --config configs/training/phase04_sft_qwen3_1_7b.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json --local-model-path models/hf_cache/Qwen_Qwen3-1.7B-Base
uv run python scripts/train/dryrun.py --stage dpo --config configs/training/phase04_dpo_qwen3_1_7b_beta_0_1.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json --local-model-path models/hf_cache/Qwen_Qwen3-1.7B-Base
```

Phase04 matrix 配置生成：

```powershell
uv run python scripts/eval/run_phase04_matrix.py --write
```

Phase04 Frozen Evaluation：

```powershell
uv run python scripts/eval/run_phase04_frozen_matrix.py --output-root reports/generated/phase04/exploratory_currentdata --skip-complete
```

Phase05 本地部署链路：

```powershell
uv run python phase05_merge.py
uv run python phase05_build_calibration.py
.\scripts\models\prepare_phase02_base_gguf.ps1
uv run python phase05_cpu_benchmark.py
```

上述命令会在本地生成模型、adapter、GGUF、calibration 和报告文件；这些产物不进入 Git。

## 20. 可复现性

项目通过以下机制维护可复现性：

- `contracts/contract_manifest.yaml` 固定合同源文件。
- Frozen Test manifest 与 SHA256 固定评估集。
- 数据 manifest 与 data card 记录数据版本、派生关系和污染检查。
- `TrainingRunSpec` 对训练配置执行严格字段校验。
- `render_config.py` 只把 LLaMA-Factory 原生参数写入 resolved config。
- Structured / Unstructured matrix 由代码生成并校验 checkpoint 共享关系。
- 模型制品与运行结果留在本地或外部制品存储，不进入 Git。

## 21. 测试

当前测试位于 `tests/`，覆盖合同 schema、Pydantic 类型、业务不变量、evaluation scorer、Phase02 benchmark、Phase03 数据流水线、Phase04 训练配置、structured runtime、formal matrix 和 validator regression。

常用测试命令：

```powershell
uv run ruff check .
uv run pytest -q
uv run homechef-contract-validate --root . --fixtures tests/fixtures/contracts
```

外部模型、llama.cpp 二进制、HF 权重或 adapter 缺失时，相关训练和推理命令按环境阻塞处理，不把阻塞写成通过。

## 22. 仓库发布边界

仓库包含：源代码、配置、合同、测试、文档、小型合成数据、manifest、data card、可复现脚本和本地部署说明。

仓库不包含：模型权重、LoRA adapter、训练 checkpoint、GGUF、HF cache、merged model、imatrix 文件、calibration 输出、Phase04/Phase05 评估结果、benchmark 输出、scorecard、case_results、failure_analysis、日志、本地临时目录、Python 虚拟环境、llama.cpp 编译产物和下载源码。
