# BOOKING_MACHINE_CONTRACT_v1

> **Contract ID**: `homechef-booking-v1`  
> **Status**: FROZEN  
> **Effective date**: 2026-08-08  
> **Purpose**: HomeChef Booking Small Model 的唯一业务机器合同。  
> **Rule**: SFT、DPO、Prompt、Frozen Eval、Dataset Validator、Runtime Parser 必须共享本合同。破坏性变化不直接修改 v1，应创建 v2。

---

# 1. Model Function

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

模型一次只能输出 `ToolCallDecision` 或 `FinalDecision`。

# 2. Runtime Input

固定五个顶层字段：

```text
history
current_state
user_input
current_time
available_tools
```

`available_tools` 传完整 Tool JSON Schema。

# 3. BookingState

```json
{
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
}
```

# 4. Decision State

```json
{
  "booking_state": {},
  "chef_query_status": "not_checked",
  "candidate_chefs": [],
  "awaiting_confirmation": false
}
```

`chef_query_status`：

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

# 5. Required Slot

```text
service_date
start_time
people
address
```

# 6. Normalization

- 日期基于 `current_time`；
- 模糊 daypart 不补时刻；
- 地址不补用户没说的城市；
- 只抽取明确人数；
- `ingredient_purchase=true` 表示厨师采购；
- `budget_min/max` 表示总预算；
- 不自动换算人均预算；
- `dietary_constraints` 强继承。

# 7. find_chefs Tool Call

严格 12 个 arguments：

```text
chef_name
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
```

# 8. find_chefs Tool Result Contract (CG-04 CLOSED)

每个 mode × status 对具有完全确定的字段集：

## search/matched

```json
{"mode":"search","status":"matched","candidates":[{"chef_id":"C003","chef_name":"张伟"}]}
```

candidates required, minItems=1

## search/no_match

```json
{"mode":"search","status":"no_match","candidates":[]}
```

candidates required, exactly []

## search/out_of_service_area

```json
{"mode":"search","status":"out_of_service_area","candidates":[]}
```

candidates required, exactly []

## search/error

```json
{"mode":"search","status":"error","error_code":"SERVICE_ERROR","retryable":false,"message":"查询服务暂时不可用"}
```

error_code: string, retryable: boolean, message: string — all required

## specific/available

```json
{"mode":"specific","status":"available","chef":{"chef_id":"C003","chef_name":"张伟"}}
```

chef required, exact CandidateChef

## specific/unavailable

```json
{"mode":"specific","status":"unavailable","requested_chef":"张伟","alternatives":[]}
```

requested_chef: string (required), alternatives: required, may be []

## specific/not_found

```json
{"mode":"specific","status":"not_found","requested_chef":"张伟","alternatives":[]}
```

requested_chef: string (required), alternatives exactly []

## specific/out_of_service_area

```json
{"mode":"specific","status":"out_of_service_area","requested_chef":"张伟","alternatives":[]}
```

requested_chef: string (required), alternatives exactly []

## specific/error

```json
{"mode":"specific","status":"error","error_code":"CHEF_QUERY_ERROR","retryable":false,"message":"查询厨师服务暂时不可用"}
```

error_code: string, retryable: boolean, message: string — all required

## CandidateChef v1

```json
{"chef_id":"C003","chef_name":"张伟"}
```

additionalProperties=false

# 9. Tool Fact Boundary (see section 8 for exact Tool Result contract)

必须由 Tool 提供：

```text
推荐 chef_name
chef_id
candidate_chefs
chef availability
service area validity
query result status
```

# 9. Tool Fact Invalidation

任一 Chef Query Dependency 变化后统一：

```text
chef_id = null
chef_query_status = not_checked
candidate_chefs = []
confirmation = false
awaiting_confirmation = false
```

# 10. Confirmation

候选选择不等于最终确认。

只有：

```text
awaiting_confirmation=true
+ specific chef 已被 Tool 核实
+ 用户明确 affirmative
+ 当前轮没有修改查询条件
```

才允许：

```text
confirmation=true
reply_type=booking_authorized
```

# 11. Final Decision

固定：

```text
action
booking_state
chef_query_status
candidate_chefs
info_complete
unrelated
missing_info
reply_type
reply
```

`reply_type`：

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

# 12. Out of Scope

v1 不支持：已存在订单改约、取消订单、支付退款、create_order、声称订单成功、批量多预约、自动 Tool Retry、Parallel Tool Calls、模糊时间补具体时刻、隐式补城市、人均预算换算、模糊预算范围推断、隐式人数推理、模型自行生成 chef_id、修改 Tool 候选顺序。

# 13. Versioning

以下变化必须创建 v2：字段增删/重命名、Tool 参数破坏性变化、Tool Result 状态语义变化、Confirmation 语义变化、状态失效规则变化、Final/ToolCall union 变化。
