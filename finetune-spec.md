# HomeChef Booking Agent 本地小模型后训练项目 Spec

> **Repository**: `dyy21zyy/sft-dpo-HomeChef-Agent`  
> **Document**: `finetune-spec.md`  
> **Status**: Approved Design v1.0  
> **Contract**: `BOOKING_MACHINE_CONTRACT_v1.md`  
> **Last updated**: 2026-08-08

---

# 前言

本项目面向 HomeChef Booking 场景建立一条完整的本地小模型后训练工程链路：**Agent Evaluation → 数据生成 → SFT → DPO → 模型选择**，并覆盖 **模型合并、GGUF 量化、llama.cpp 本地部署、失败驱动数据迭代和最终集成**。

技术目标不是简单做：

```text
technician → chef
massage → cooking
```

而是：

```text
读取 HomeChef Booking 子系统
        ↓
提取 Booking Model 外部合同
        ↓
冻结 BOOKING_MACHINE_CONTRACT_v1
        ↓
建立 HomeChef 专用训练/评估框架
        ↓
重新构建 HomeChef Frozen Eval
        ↓
重新生成 HomeChef SFT/DPO 数据
        ↓
重新训练和评估
        ↓
量化、本地部署、失败迭代
        ↓
接回 HomeChef-Agent
```

本项目与 HomeChef 主项目保持独立仓库。主项目定义业务事实与真实 Tool；本仓库负责训练一个能够稳定执行 Booking 状态转移和 Tool Calling 决策的小模型。

---

# 1. 项目背景

## 1.1 为什么选择 Booking Agent 做本地小模型后训练

HomeChef-Agent 的完整能力包括 Supervisor、咨询、RAG、长期记忆、Skill、Booking、数据库和订单服务。并不是所有 LLM 调用都适合迁移成本地小模型。

Booking Agent 中存在一类高频但相对封闭的模型调用：

```text
历史对话
+
上一轮 Booking 状态
+
当前用户输入
+
当前时间
+
当前允许工具
        ↓
模型
        ↓
结构化状态更新
+
Tool Call 或 Final Decision
```

它具有以下特征：

- 高频调用；
- 业务范围窄；
- 输入输出协议固定；
- 需要中文口语理解，但不需要大量开放世界知识；
- 需要多轮状态继承和覆盖；
- 需要相对日期/时间标准化；
- 需要轻量 Tool Calling；
- 对 JSON 稳定性和业务边界要求高；
- 对厨师档期、服务区域、实时价格等事实必须依赖 Tool，而不能自由生成。

因此，这一调用点比开放式咨询、RAG 问答、复杂 Supervisor Planning 更适合迁移到 0.6B～1.7B 级别的小模型中。

## 1.2 本项目要替换什么

本项目最终替换的是 HomeChef Booking Agent 中的“结构化 Booking Decision LLM”，而不是整个 Booking Agent。

目标职责：

```text
Extract
从当前用户输入抽取 Booking Slot

Merge
与 current_state 做最小状态合并

Normalize
规范化日期、时间、人数、预算等字段

Validate
判断查询厨师所需信息是否齐全

Decide
决定 Tool Call 还是 Final

Consume
消费 find_chefs Tool Result

Confirm
处理候选选择、最终确认、拒绝、修改
```

明确不负责：

```text
Chef RAG 检索实现
数据库档期查询
实时价格
服务区域判定
订单写入
支付
RAG 问答
Memory
Skill
Supervisor ReAct
开放式菜谱咨询
```

## 1.3 项目独立性

完成 `BOOKING_MACHINE_CONTRACT_v1` 后，本项目原则上可以独立完成：

```text
数据生成
Frozen Evaluation
Base Benchmark
SFT
DPO
Failure Analysis
量化
llama.cpp 本地评估
```

只有两个阶段需要重新依赖 HomeChef 主项目：

1. **Phase 01 前置合同提取/兼容确认**；
2. **Phase 07 最终端到端集成测试**。

---

# 2. 核心机器合同

> 本章给出 `BOOKING_MACHINE_CONTRACT_v1.md` 的摘要。若冲突，以独立 Contract 文档为唯一权威来源。

## 2.1 模型任务定义

```python
def booking_decision(
    history,
    current_state,
    user_input,
    current_time,
    available_tools,
) -> ToolCallDecision | FinalDecision:
    ...
```

模型每次只能输出 `ToolCallDecision` 或 `FinalDecision`，不允许同时输出两套结构。

## 2.2 Runtime Input

正式输入：

```json
{
  "history": [],
  "current_state": {
    "booking_state": {
      "service_date": null,
      "start_time": null,
      "people": null,
      "address": null,
      "cuisine": null,
      "budget_min": null,
      "budget_max": null,
      "menu": [],
      "chef_id": null,
      "chef_name": null,
      "ingredient_purchase": null,
      "dietary_constraints": [],
      "occasion": null,
      "confirmation": null
    },
    "chef_query_status": "not_checked",
    "candidate_chefs": [],
    "awaiting_confirmation": false
  },
  "user_input": "周六晚上六点，6个人，在杨浦，预算800到1200，家宴，想吃川菜",
  "current_time": "2026-08-08 17:00",
  "available_tools": [
    {
      "type": "function",
      "function": {
        "name": "find_chefs",
        "description": "根据预约约束查询可服务厨师；指定 chef_name 时执行指定厨师验证，否则执行条件匹配。",
        "parameters": {}
      }
    }
  ]
}
```

五个顶层字段必须且只能是：

```text
history
current_state
user_input
current_time
available_tools
```

## 2.3 History

采用 OpenAI-style Tool Calling 消息。

允许：

```json
{"role":"user","content":"周六晚上六点，6个人，在杨浦"}
```

```json
{"role":"assistant","content":"请问服务地址在哪里？"}
```

```json
{
  "role":"assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_001",
    "type": "function",
    "function": {
      "name": "find_chefs",
      "arguments": "{\"chef_name\":null,\"service_date\":\"2026-08-15\",\"start_time\":\"18:00\",\"people\":6,\"address\":\"杨浦\",\"cuisine\":\"川菜\",\"budget_min\":800,\"budget_max\":1200,\"menu\":[],\"ingredient_purchase\":null,\"dietary_constraints\":[],\"occasion\":\"家庭聚餐\"}"
    }
  }]
}
```

```json
{
  "role": "tool",
  "tool_call_id": "call_001",
  "name": "find_chefs",
  "content": "{\"mode\":\"search\",\"status\":\"matched\",\"candidates\":[{\"chef_id\":\"C003\",\"chef_name\":\"张伟\"}]}"
}
```

约束：

- `history` 不包含 system prompt；
- 新 User Turn 第一次调用时 `user_input != null`；
- Tool Result continuation 时当前 user 消息已经进入 history，`user_input = null`；
- `function.arguments` 固定为 JSON string；
- v1 一次模型输出最多一个 Tool Call；
- Tool Result 必须与 `tool_call_id` 对应；
- 不把整份内部 FinalDecision JSON 当普通 assistant content 写入 History。

## 2.4 BookingSlot

| 字段 | JSON 类型 | 说明 |
|---|---|---|
| `service_date` | `string|null` | `YYYY-MM-DD` |
| `start_time` | `string|null` | `HH:MM` |
| `people` | `integer|null` | 明确人数 |
| `address` | `string|null` | 用户提供的服务地址文本 |
| `cuisine` | `string|null` | 菜系偏好 |
| `budget_min` | `number|null` | 总预算下界 |
| `budget_max` | `number|null` | 总预算上界 |
| `menu` | `array[string]` | 菜单/菜品要求 |
| `chef_id` | `string|null` | 必须来自 Tool |
| `chef_name` | `string|null` | 可来自用户指定或 Tool |
| `ingredient_purchase` | `boolean|null` | true=厨师采购；false=用户准备 |
| `dietary_constraints` | `array[string]` | 忌口、过敏等 |
| `occasion` | `string|null` | 场景/宴请用途 |
| `confirmation` | `boolean|null` | 对当前已验证方案的确认状态 |

## 2.5 Required Slot

查询厨师前最低必需字段：

```text
service_date
start_time
people
address
```

唯一公式：

```python
info_complete = (
    service_date is not None
    and start_time is not None
    and people is not None
    and address is not None
)
```

`missing_info` 只允许上述四个字段，并按固定顺序输出。

## 2.6 用户事实与 Tool Fact 边界

模型可从用户输入/有效历史状态获得：

```text
service_date
start_time
people
address
cuisine
budget_min
budget_max
menu
指定 chef_name
ingredient_purchase
dietary_constraints
occasion
confirmation 表达
```

必须来自 Tool：

```text
推荐 chef_name
chef_id
chef availability
service area validity
candidate_chefs
实时 query result status
实时价格（若未来 Tool 暴露）
```

禁止模型自行编造业务事实。

---

# 3. Slot 规范化规则

## 3.1 日期

相对日期基于 `current_time`：

```text
今天 → current date
明天 → +1 day
后天 → +2 days
周X → 当前日期之后最近一次该星期几
下周X → 下一自然周对应星期几
无法唯一解析 → null
```

如果当前日期本身是周六，用户说“周六”，v1 解释为下一次周六。

## 3.2 时间

模糊 daypart 不补具体时刻：

```text
晚上 → null
下午 → null
午饭时间 → null
晚饭时间 → null
```

明确时刻才可标准化：

```text
晚上六点 → 18:00
下午三点 → 15:00
上午十点半 → 10:30
```

只说“6点”且 AM/PM 不唯一：

```text
start_time = null
```

## 3.3 地址

v1 不根据常识隐式补城市。

```text
用户：“杨浦”
→ address = "杨浦"
```

除非未来 Runtime 显式加入 `business_context.service_city`。

## 3.4 人数

```text
“6个人” → 6
“我们一家人” → null
“我爸妈和我” → null
```

v1 不做隐式人数推理。

## 3.5 ingredient_purchase

```text
true  = 厨师负责采购
false = 用户自行准备
null  = 未确定
```

“需要买菜”没有明确采购责任时为 `null`。

## 3.6 Budget

`budget_min / budget_max` 表示本次预约总预算区间。

| 用户表达 | budget_min | budget_max |
|---|---:|---:|
| 800 到 1200 | 800 | 1200 |
| 不超过 1000 | null | 1000 |
| 最多 1000 | null | 1000 |
| 至少 1000 | 1000 | null |
| 预算 1000 | null | 1000 |
| 1000 左右 | null | null |
| 大概 1000 | null | null |

v1 不支持人均预算自动换算。

## 3.7 dietary_constraints

属于高风险强继承字段。只有用户明确纠正才能删除或替换。

---

# 4. Tool 设计

## 4.1 为什么只保留一个主要 Tool

v1 给小模型一个主要查询工具：

```text
find_chefs
```

不拆成多个查询工具，原因是减少小模型 Tool Selection 难度，并把 Chef RAG、数据库、档期与服务区域等复杂逻辑封装在 Tool 内。

## 4.2 Tool Call Schema

```json
{
  "action": "tool_call",
  "tool_name": "find_chefs",
  "arguments": {
    "chef_name": null,
    "service_date": "2026-08-15",
    "start_time": "18:00",
    "people": 6,
    "address": "杨浦",
    "cuisine": "川菜",
    "budget_min": 800,
    "budget_max": 1200,
    "menu": [],
    "ingredient_purchase": null,
    "dietary_constraints": [],
    "occasion": "家庭聚餐"
  }
}
```

12 个 argument key 全部出现。

禁止增加：

```text
reason
thought
analysis
reply
booking_state
candidate_chefs
```

## 4.3 Tool 调用条件

只有 `service_date/start_time/people/address` 全部存在，且当前查询条件下没有有效 Tool Fact 时才允许调用。

> **需要新的外部业务事实 → Tool Call；已有足够事实、只需向用户展示/追问/确认 → Final。**

## 4.4 条件搜索

`chef_name == null`。

合法状态：

```text
matched
no_match
out_of_service_area
error
```

`matched` 返回 `candidates[]`，模型必须保持 Tool 返回顺序，不能自行重排。

## 4.5 指定厨师

`chef_name != null`。

合法状态：

```text
available
unavailable
not_found
out_of_service_area
error
```

`unavailable` 可以返回 `alternatives[]`，但模型不能自动选 Top-1。

---

# 5. Final Decision

正式结构：

```json
{
  "action": "final",
  "booking_state": {
    "service_date": "2026-08-15",
    "start_time": "18:00",
    "people": 6,
    "address": "杨浦",
    "cuisine": "川菜",
    "budget_min": 800,
    "budget_max": 1200,
    "menu": [],
    "chef_id": null,
    "chef_name": null,
    "ingredient_purchase": null,
    "dietary_constraints": [],
    "occasion": "家庭聚餐",
    "confirmation": false
  },
  "chef_query_status": "matched",
  "candidate_chefs": [
    {"chef_id":"C003","chef_name":"张伟"},
    {"chef_id":"C007","chef_name":"李明"}
  ],
  "info_complete": true,
  "unrelated": false,
  "missing_info": [],
  "reply_type": "present_chef_candidates",
  "reply": "找到两位符合当前条件的厨师：张伟和李明。请选择你希望预约的厨师。"
}
```

## 5.1 chef_query_status

```text
not_checked
matched
available
unavailable
not_found
no_match
out_of_service_area
error
```

## 5.2 reply_type

```text
handoff
ask_service_date
ask_start_time
ask_people
ask_address
ask_multiple_required_fields
present_chef_candidates
confirm_specific_chef
present_alternatives
inform_not_found
inform_no_match
inform_out_of_service_area
booking_authorized
booking_paused
acknowledge_result
```

## 5.3 reply 的评估原则

`reply` 不做逐字 exact match。严格 Gold 主要包括：

```text
action
tool_name
arguments
booking_state
chef_query_status
candidate_chefs
info_complete
unrelated
missing_info
reply_type
```

`reply` 只检查语义动作和事实一致性。

---

# 6. 状态转移与确认

## 6.1 最小状态合并

用户没有修改的有效用户事实继续保留，仅更新被明确修改的字段。

## 6.2 Tool Fact 失效

以下任一字段变化：

```text
service_date
start_time
people
address
cuisine
budget_min
budget_max
menu
ingredient_purchase
dietary_constraints
occasion
chef_name
```

统一失效：

```text
chef_id = null
chef_query_status = not_checked
candidate_chefs = []
confirmation = false
awaiting_confirmation = false
```

## 6.3 候选选择不等于最终确认

用户说“第二个”或“第二个，就他”，只完成候选选择：

```text
chef_id = 对应 Tool candidate chef_id
chef_name = 对应 chef_name
confirmation = false
reply_type = confirm_specific_chef
```

## 6.4 最终确认

仅当：

```text
awaiting_confirmation = true
具体 chef 已被 Tool 核实
用户本轮明确 affirmative
本轮没有修改任何查询相关 Slot
```

才允许：

```text
confirmation = true
reply_type = booking_authorized
```

## 6.5 mutation dominates confirmation

“可以，不过改成七点”中，修改优先于确认词，必须让旧 Tool Fact 失效并重新查询。

---

# 7. v1 Out of Scope

明确不训练：

- 修改已经存在的订单；
- 取消已有订单；
- 支付/退款；
- Booking Model 自己执行 `create_order`；
- 模型声称订单已经创建成功；
- 一次创建多个预约；
- Tool timeout 自动重试策略；
- Parallel Tool Calls；
- 模糊时间自动补具体时刻；
- 隐式补城市；
- 人均预算自动换算总预算；
- 模糊预算范围推断；
- 隐式人数推理；
- 用 RAG 静态内容证明实时可用性；
- 模型自行生成 chef_id；
- 模型修改 Tool 返回候选顺序。

---

# 8. 基座模型选择

## 8.1 任务能力需求

优先关注：

```text
中文口语理解
多轮状态理解
结构化 JSON
Tool Calling
相对时间解析
规则遵循
幻觉抑制
```

## 8.2 第一轮候选

正式比较：

```text
Qwen3-0.6B
Qwen3-1.7B
```

不在设计阶段预先宣布冠军。若两者都明显不足，再扩展更大模型，而不是机械堆 DPO。

---

# 9. 后训练方案

## 9.1 总体路线

```text
Base
 ↓
SFT
 ↓
Frozen Eval
 ↓
SFT 已基本学会任务？
 ├─ No → 数据/合同/模型容量分析
 └─ Yes
      ↓
     DPO
      ↓
Failure-driven Iteration
```

## 9.2 SFT

SFT 负责学习：

```text
Machine Contract
JSON Schema
Slot Normalization
State Merge
Tool Call 格式
Tool Result 消费
reply_type
确认流程
```

## 9.3 DPO

DPO 只针对 SFT 残余边界错误：

```text
Chef Hallucination
Wrong Tool Timing
State Corruption
Dietary Constraint Violation
Relative Time Error
Confirmation Error
Wrong Tool Fact
```

启动门槛：

```text
Protocol >= 90%
Task Correctness >= 85%
```

---

# 10. 训练框架与默认参数

## 10.1 框架

主线：

```text
LLaMA-Factory + PEFT LoRA
```

未来若进入 GRPO、自定义 reward 或魔改 loss，再下沉 TRL；v1 不引入。

## 10.2 SFT 默认配置

```yaml
stage: sft
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
template: qwen3
enable_thinking: false
train_on_prompt: false
mask_history: true
cutoff_len: 2048
learning_rate: 1.0e-4
num_train_epochs: 3
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 8
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
fp16: false
eval_strategy: epoch
save_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
save_total_limit: 2
plot_loss: true
```

## 10.3 DPO 默认配置

```yaml
stage: dpo
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
template: qwen3
enable_thinking: false
pref_loss: sigmoid
cutoff_len: 2048
learning_rate: 5.0e-6
num_train_epochs: 1
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
fp16: false
```

DPO 从对应 SFT Adapter 继续训练。

---

# 11. 数据构造

## 11.1 数据来源

v1 首轮不依赖真实用户日志。主线：

```text
BOOKING_MACHINE_CONTRACT_v1
+
Scenario Specs
+
强模型数据生成
+
Schema Validator
+
Business Validator
+
人工抽检
```

强模型只能在合同约束下生成实例，不能自行决定业务规则。

## 11.2 数据生成模型

默认配置化为：

```yaml
backend: openai_responses
model: gpt-5.6-sol
max_tokens: 4096
timeout_s: 180
max_attempts: 3
```

正式生成前先生成 25 条 Smoke Samples，人工 Review 后才扩大。

## 11.3 Raw Sample

```json
{
  "id": "phase03-tool-001",
  "output_kind": "tool_call",
  "conversation_kind": "multi_turn",
  "tags": ["tool_call", "relative_time", "dietary"],
  "input": {},
  "expected": {},
  "dpo_targets": ["H2", "H4"]
}
```

Raw 是唯一事实源，SFT/DPO 从 Raw 确定性派生。

## 11.4 v0.1 Pilot Scale

首轮：

```text
Raw ≈ 600
```

| 类别 | 数量 |
|---|---:|
| 缺失信息 / 追问 | 130 |
| Tool Call | 140 |
| Tool Result → Final | 140 |
| 确认 / 拒绝 / 修改 | 110 |
| unrelated / handoff | 80 |
| 合计 | 600 |

SFT：

```text
train ≈ 540
val ≈ 60
```

DPO v1 只在 SFT Gate 通过后构建，目标约 210 对。

## 11.5 Broad Underfitting Gate

若第一轮 SFT：

```text
Protocol < 90%
或 Task Correctness < 85%
或 Effective Pass < 60%
或 >= 4 类主要错误同时大量出现
```

则进入：

```text
600 Raw → 1200 Raw → 重新 SFT
```

若仍不足，再考虑 1800～2400 Raw 或升级 Base Model。

## 11.6 数据校验

每条样本必须过：

```text
JSON Schema
Runtime Input Schema
Final/ToolCall Union Schema
History Tool Call 配对
Slot 类型
日期/时间格式
info_complete
missing_info
Tool Fact evidence
confirmation legality
state invalidation
candidate order
Out-of-Scope exclusion
eval overlap
```

## 11.7 数据隔离

```text
train ∩ val = 0
train ∩ diagnostic_dev = 0
train ∩ frozen_test = 0
```

正式 Data Card 必须记录 `frozen_eval_overlap = 0`。

---

# 12. DPO 数据

## 12.1 H1-H7

```text
H1 Chef Hallucination
H2 Wrong Tool Timing
H3 State Corruption
H4 Dietary Constraint Violation
H5 Relative Time Error
H6 Confirmation Error
H7 Wrong Tool Fact
```

## 12.2 Hard Negative

Rejected 必须保持 Schema 合法，只改变业务行为。禁止用乱码、非法 JSON 或完全无关文本。

## 12.3 Real Failure DPO

Phase 06 重点加入：

```text
chosen = 正确 Gold
rejected = SFT 在 Diagnostic Dev 上的真实错误输出
```

Frozen Test Prediction 不允许直接回流训练。

---

# 13. Agent Evaluation

## 13.1 两套评估数据

Frozen Test：正式模型选择，建议约 120 条，冻结后不可用于训练。

Diagnostic Dev：错误分析与 Phase 06 数据闭环，建议约 80 条，可版本化。

## 13.2 Eval Case

```json
{
  "id": "tool-001",
  "output_kind": "tool_call",
  "conversation_kind": "single_turn",
  "input": {},
  "expected": {},
  "assertions": [],
  "tags": ["tool_call", "relative_time"]
}
```

## 13.3 Protocol

确定性检查：

```text
JSON 可解析
无 Markdown code fence
action 合法
Final/ToolCall 字段精确
类型合法
enum 合法
无多余字段
```

## 13.4 Task Correctness

Tool Call 严格检查 `action/tool_name/12个arguments`。

Final 检查：

```text
booking_state
chef_query_status
candidate_chefs
info_complete
unrelated
missing_info
reply_type
```

建议：

```text
task_correctness = 0.70 * structured_score + 0.30 * reply_score
```

## 13.5 Semantic Slot

对于 `cuisine/menu/dietary_constraints/occasion` 的语义等价，可采用：

```text
NFKC / 标点归一化
→ 高置信别名
→ 否定极性门控
→ 本地 Embedding
```

可沿用 `BAAI/bge-small-zh-v1.5`，阈值初始 `0.70`。饮食限制的否定极性冲突必须直接判错。

## 13.6 Critical Error

以下错误直接判 case fail：

```text
编造 Tool 未返回的 chef
unavailable → available
丢失/反转过敏或明确忌口
未经确认 booking_authorized
声称订单已经创建
错误继承已失效 chef_id
缺 Required Slot 却调用 find_chefs
改变 Tool candidate 顺序并改变选择语义
```

## 13.7 Effective Pass

```text
protocol = pass
AND task_correctness >= 0.95
AND critical_error = false
```

## 13.8 Scorecard

至少输出：

```text
Protocol
Task Correctness
Effective Pass
Critical Error Count
Slot Accuracy
Tool Timing Accuracy
Tool Arguments Accuracy
State Inheritance Accuracy
Relative Time Accuracy
Confirmation Accuracy
Tool Fact Grounding Accuracy
Dietary Constraint Recall
P50/P95 latency
TTFT
prompt tok/s
generation tok/s
model size
peak RSS
```

质量指标与资源指标分开记录。

---

# 14. 训练实验矩阵

## 14.1 Phase 4A：SFT 优先

第一轮：

```text
Qwen3-0.6B-SFT
Qwen3-1.7B-SFT
```

两者都跑 Frozen Test + Diagnostic Dev。

## 14.2 Phase 4B：条件式 DPO

如果最佳 SFT 达到门槛，再跑：

```text
Best-SFT → DPO beta=0.1
Best-SFT → DPO beta=0.3
```

如果 0.6B 与 1.7B 接近，才考虑两边都做 DPO。

## 14.3 Champion Selection

顺序：

1. Critical Error 最低；
2. Effective Pass 最高；
3. Task Correctness；
4. Protocol；
5. 接近时优先小模型；
6. DPO 无提升就保留 SFT。

---

# 15. 训练算力

正式训练建议 AutoDL 或等价按小时 GPU 平台，参考 RTX 4090 24GB。

本地负责：数据、校验、配置、CPU dry-run、评估、量化、llama.cpp。

正式训练前必须做：

```text
1 条 SFT
1 条 DPO
max_steps = 2
```

只验证工程链路，不代表训练有效。

---

# 16. 量化方案

## 16.1 路线

```text
Base + LoRA
→ Merge
→ F16 GGUF
→ imatrix
→ Q8_0 / Q5_K_M / Q4_K_M
→ llama.cpp
```

采用 PTQ，不做 QAT。

## 16.2 imatrix

从训练/验证语料抽约 128 条代表性 Prompt，禁止使用 Frozen Test。

## 16.3 Quantization Release Gate

```text
Critical Error 不增加
Protocol 相对 F16 下降 <= 1pp
Task Correctness 相对 F16 下降 <= 2pp
```

Q4_K_M 是首选候选，不是强制最终选择。

---

# 17. 推理框架与 SLO

主线：`llama.cpp / llama-server`。

主项目通过 OpenAI-compatible API 调用。

参考目标环境：

```text
x86_64
8 physical cores / 16 threads+
AVX2+
32 GB RAM
CPU-only
Windows 11 或 Ubuntu
```

Release Gate 目标：

```text
Protocol >= 98%
Task Correctness >= 90%
Effective Pass >= 75%
Critical Error = 0
P95 total latency <= 5.0s
P95 TTFT <= 2.0s
Peak RSS <= 4GB
Integration fallback rate <= 2%
```

这些是目标，不是预先宣称已经达到的结果。

---

# 18. Failure-driven Iteration

一级 Failure Taxonomy：

```text
F1 Protocol
F2 Missing Information
F3 Relative Time
F4 Tool Timing
F5 Tool Arguments
F6 State Inheritance
F7 Chef Hallucination
F8 Tool Fact
F9 Dietary Constraint
F10 Confirmation
F11 Reply Semantics
F12 Unrelated Routing
```

从 Diagnostic Dev 导出失败，按类别定向补数据。v0.2 不覆盖 v0.1，保留完整数据谱系。

---

# 19. HomeChef 合同工程正确性

## 19.1 Contract Freeze Anchor

唯一业务机器合同起点：

```text
BOOKING_MACHINE_CONTRACT_v1.md
```

## 19.2 Contract Baseline

正式数据生成和训练前，先跑通本仓库的 Schema、Contract Validator、Mock Eval 和最小 scorecard，确认 HomeChef Booking 合同可以被机器校验。

## 19.3 Scorer Consistency Anchor

在进入正式 Frozen Eval 和训练前，用同一组 HomeChef mock predictions 重复运行 scorer，要求：

```text
scorecard 一致
error tags 一致
effective pass 一致
```

非 HomeChef 合同数据不能进入 HomeChef 正式训练、Frozen Eval 或最终成绩。

---

# 20. 文件级实现策略

动作：

```text
NEW
ADAPT
DO_NOT_CREATE_AS_PHASE00_OUTPUT
```

通用工程模块按 HomeChef 命名空间重新实现：

```text
src/homechef_booking/inference/
src/homechef_booking/utils/jsonl.py
src/homechef_booking/evaluation/runner.py
src/homechef_booking/evaluation/scorecard.py
scripts/train/collect_artifacts.py
scripts/train/dryrun.py
scripts/train/package_cloud_artifacts.py
scripts/train/render_config.py
scripts/train/run_matrix.sh
scripts/eval/run_eval.py
scripts/eval/collect_analysis.py
scripts/eval/diff_runs.py
```

业务相关 NEW/ADAPT：

```text
raw_sample.py
raw_schema.py
raw_validator.py
scenario_specs.py
generator.py
sft_render.py
dpo_perturb.py
coverage_audit.py
tag_audit.py
dataset_build.py
isolation.py
evaluation/assertions.py
evaluation/scorer.py
evaluation/scenarios.py
evaluation/scorers/*
prompts/*
schemas/*
```

以下内容不得作为 HomeChef 正式资产直接进入训练或最终成绩：

```text
非 HomeChef Contract Frozen Eval
非 HomeChef Raw/SFT/DPO Data
非 HomeChef LoRA
非 HomeChef Champion
非 HomeChef scorecard 作为 HomeChef 成绩
```

新增 NEW：

```text
BOOKING_MACHINE_CONTRACT_v1.md
contracts/booking_machine_contract_v1.schema.json
contracts/find_chefs_v1.schema.json
contracts/contract_manifest.yaml
scripts/quantize/*
scripts/iteration/*
scripts/report/*
```

---

# 21. 目标仓库结构

```text
sft-dpo-HomeChef-Agent/
├── README.md
├── finetune-spec.md
├── BOOKING_MACHINE_CONTRACT_v1.md
├── pyproject.toml
├── requirements-train.txt
├── contracts/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── dev/
│   ├── eval/
│   └── calibration/
├── deployment/llama_cpp/
├── experiments/
├── models/
├── project-log/
├── reports/
├── scripts/
│   ├── data/
│   ├── eval/
│   ├── train/
│   ├── quantize/
│   ├── iteration/
│   ├── report/
│   └── serve/
├── src/homechef_booking/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── prompts/
│   ├── schemas/
│   └── utils/
└── tests/
    ├── fixtures/
    ├── unit/
    └── integration/
```

---

# 22. 实施阶段

## Phase 00 — Contract Baseline

冻结 HomeChef Booking 合同，建立 Schema、Contract Validator、Mock Eval 和最小 scorecard。

## Phase 01 — Scaffold + Contract

完善 HomeChef 专用工程框架，冻结 HomeChef Machine Contract。

DoD：Scorer Replay Anchor 通过、Contract FROZEN、Schema/Prompt/Unit Test/Mock Eval 通过。

## Phase 02 — Frozen Eval + M0

建立约 120 条 Frozen Test + 约 80 条 Diagnostic Dev，跑 0.6B/1.7B Base Benchmark。

## Phase 03 — Dataset

25 条 Smoke → 正式 Raw ≈ 600 → SFT → 条件式 DPO；如 Broad Underfitting，扩到 1200 甚至 1800～2400。

## Phase 04 — Training

先 0.6B/1.7B SFT，再决定是否 DPO。必须输出 Champion Selection。

## Phase 05 — Quantization & Deployment

Champion → Merge → F16 GGUF → imatrix → Q8/Q5/Q4 → llama-server → CPU Benchmark。

## Phase 06 — Failure-driven Iteration

Dev Failures → Taxonomy → Targeted Data → SFT/DPO v2 → Regression Diff。

## Phase 07 — HomeChef Integration & Final Report

本地模型接入 Booking Agent，真实执行 find_chefs，Tool Result continuation，最终完成 Final Benchmark 与 Deployment Runbook。

---

# 23. Runtime Fallback

本地模型遇到：

```text
timeout
invalid JSON
Schema invalid
llama-server unavailable
一次 repair retry 后仍失败
```

允许 fallback 到远程通用 LLM，并记录：

```text
fallback_reason
model_id
quantization
latency
protocol_error
```

---

# 24. 数据安全与复现性

公开仓库使用合成用户、地址、厨师和业务状态，不提交真实 PII。

每个正式 Run 保存：

```text
git_sha
contract_version
contract_hash
dataset_version
dataset_sha
seed
base_model
adapter
training_config
dependency_snapshot
hardware
llama.cpp_version
inference_config
```

---

# 25. 项目里程碑

```text
M0  Base Benchmark
M1  SFT
M2  DPO（条件式）
M3  Quantized Local Model
M4  Failure-driven Iteration
M5  HomeChef Integrated Release
```

---

# 26. 最终 Benchmark

| Model | Protocol | Task Correctness | Effective Pass | Critical Errors | Size | P95 | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Base | | | | | | | |
| Qwen3-0.6B SFT | | | | | | | |
| Qwen3-1.7B Base | | | | | | | |
| Qwen3-1.7B SFT | | | | | | | |
| Best DPO | | | | | | | |
| Iteration v2 | | | | | | | |
| Final Quantized | | | | | | | |

最终必须诚实回答：SFT 是否有效、DPO 是否有效、失败迭代解决了哪些问题、量化损失多少、最终为何选择该模型/量化档位、CPU 是否满足 SLO、还有哪些已知限制。

---

# 27. 最终项目定位

完成后的项目不是“对 Qwen 做一次 SFT + DPO”，而是一条完整的 Agent 小模型后训练工程链：

```text
业务合同提取
→ Machine Contract
→ Agent Evaluation
→ 合成数据
→ SFT
→ DPO
→ Failure-driven Data Flywheel
→ LoRA Merge
→ GGUF
→ imatrix
→ Quantization
→ llama.cpp
→ CPU Deployment
→ HomeChef Agent Integration
```

最终定义：

> **面向 HomeChef Booking Agent 的业务专用小模型后训练、Agent Evaluation、失败驱动迭代、量化与本地 CPU 部署项目。**
