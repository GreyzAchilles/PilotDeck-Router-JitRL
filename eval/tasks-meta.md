# 评测任务集元数据（v0）

## 1. 族表与 GT 分布

本任务集包含 24 条主任务（6 族 × 4 变体）和 5 条续轮探针，共 29 条，文件行序即呈现序列。

| family | 主题 | 变体 | GT 分布 |
|---|---|---:|---|
| `chat_qa` | 闲聊/简单问答 | 4 | simple ×4 |
| `code_gen` | 代码生成（消息内含代码块） | 4 | medium ×4 |
| `doc_writing` | 文档写作 | 4 | medium ×2, reasoning ×2 |
| `data_analysis` | 数据分析（消息内含数据） | 4 | reasoning ×4 |
| `refactor` | 代码重构（消息内含代码块） | 4 | medium ×2, reasoning ×2 |
| `info_retrieval` | 信息检索/资料综合 | 4 | simple ×1, reasoning ×1, complex ×2 |
| **主任务合计** |  | **24** | **simple 5, medium 8, reasoning 9, complex 2** |
| `continuation` | 续轮继承探针 | 5 | medium ×2, reasoning ×3 |

GT 依据：`complex` 只用于并行编排或专职代理委派；不涉及编排的多文件、数据分析、多步骤或多来源工作归为 `reasoning`；1–2 文件读写、短文档或单次代码生成归为 `medium`；寒暄、确认、记规则和单步抽取归为 `simple`。

## 2. 变体与同签名原理

每个主任务族的 4 个变体复用逐字符相同的请求样板，只替换样板后的内联实例内容。代码块、数字和路径经过签名归一化后会变成 `<code>`、`<num>`、`<path>` 等占位形式，因此同族任务会得到近似签名，后出现的变体可以检索到前面变体的经验。

具体例子：`code_gen` 的 T05、T06、T07、T08 都使用完全相同的样板：

````text
给下面这个函数写单元测试：
```python
<不同的小函数>
```
````

四条任务只替换函数代码，分别使用 `clamp`、`slugify`、`is_leap_year` 和 `chunk`。同理，`data_analysis` 固定使用“请分析下面这组数据：”、`refactor` 固定使用“请重构下面这段代码并说明改动：”、`info_retrieval` 固定使用“请处理下面这个资料任务：”。

## 3. 误导面任务

主任务中恰有 8 条标记为 `misleading_surface: true`，两个方向各 4 条。另有 5 条 continuation 按要求全部标记为 true，但不计入“双向误导面 8 条”的主任务配额。

| id | 方向 | 设计 |
|---|---|---|
| T01 | long-but-simple | 超过 100 字的礼貌寒暄，实际只需友好回应 |
| T02 | long-but-simple | 以大量限制包装“收到并确认” |
| T03 | long-but-simple | 长规则陈述，实际只需记住并确认 |
| T21 | long-but-simple | 长篇内联资料，答案是单步抽取 |
| T13 | short-but-deep | 极短数据仍需识别峰值、异常和解释边界 |
| T22 | short-but-deep | 短句要求三方资料检索与结构化比较 |
| T23 | short-but-deep | 短句明确要求三个代理并行委派 |
| T24 | short-but-deep | 短句明确要求四个专职代理并行核查 |

## 4. 呈现序列

| 文件行 | 主任务位置 | id | family | gt_tier |
|---:|---:|---|---|---|
| 1 | 1 | T01 | chat_qa | simple |
| 2 | 2 | T05 | code_gen | medium |
| 3 | — | P01 | continuation | medium |
| 4 | 3 | T09 | doc_writing | medium |
| 5 | 4 | T13 | data_analysis | reasoning |
| 6 | — | P02 | continuation | reasoning |
| 7 | 5 | T17 | refactor | medium |
| 8 | 6 | T21 | info_retrieval | simple |
| 9 | 7 | T02 | chat_qa | simple |
| 10 | 8 | T06 | code_gen | medium |
| 11 | 9 | T10 | doc_writing | medium |
| 12 | — | P03 | continuation | medium |
| 13 | 10 | T14 | data_analysis | reasoning |
| 14 | 11 | T18 | refactor | medium |
| 15 | 12 | T22 | info_retrieval | reasoning |
| 16 | 13 | T03 | chat_qa | simple |
| 17 | 14 | T07 | code_gen | medium |
| 18 | 15 | T11 | doc_writing | reasoning |
| 19 | 16 | T15 | data_analysis | reasoning |
| 20 | 17 | T19 | refactor | reasoning |
| 21 | — | P04 | continuation | reasoning |
| 22 | 18 | T23 | info_retrieval | complex |
| 23 | 19 | T04 | chat_qa | simple |
| 24 | 20 | T08 | code_gen | medium |
| 25 | 21 | T12 | doc_writing | reasoning |
| 26 | — | P05 | continuation | reasoning |
| 27 | 22 | T16 | data_analysis | reasoning |
| 28 | 23 | T20 | refactor | reasoning |
| 29 | 24 | T24 | info_retrieval | complex |

忽略 continuation 后，每个族的主任务位置形成固定步长 6：

- `chat_qa`：1, 7, 13, 19
- `code_gen`：2, 8, 14, 20
- `doc_writing`：3, 9, 15, 21
- `data_analysis`：4, 10, 16, 22
- `refactor`：5, 11, 17, 23
- `info_retrieval`：6, 12, 18, 24

所以任意相邻同族变体之间相隔 6 个主任务位置，满足“间隔 ≥3”。最后三分之一（主任务位置 17–24）覆盖六族，其同族兄长均已出现在前三分之一（位置 1–8），用于观察后段记忆收益。

## 5. 续轮探针放置

| 探针 | 紧邻前项 | 继承档位 | 消息 |
|---|---|---|---|
| P01 | T05 | medium | 继续 |
| P02 | T13 | reasoning | 好的 |
| P03 | T10 | medium | 嗯嗯 |
| P04 | T19 | reasoning | 可以 |
| P05 | T12 | reasoning | 接着来 |

每个探针都直接放在对应主任务之后，且前项均为 medium 或 reasoning。探针的 GT 与前一行完全相同，用于检查路由器能否继承上下文档位，而不是按短消息表面降为 simple。

## 6. Schema 文档

`eval/tasks.jsonl` 每行是一个独立 JSON 对象，字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 唯一任务编号；主任务使用 T01–T24，探针使用 P01–P05 |
| `family` | string | 主任务族标识，探针固定为 `continuation` |
| `variant` | integer | 主任务族内变体编号 1–4；探针固定为 0 |
| `gt_tier` | enum | `simple`、`medium`、`complex`、`reasoning` 之一 |
| `message` | string | 完全自含的自然中文用户消息 |
| `quality_checklist` | string[] | 3–5 条具体、可核验的回答质量要求 |
| `misleading_surface` | boolean | 是否具备误导性的长度或表面形式 |
| `notes` | string | 中文设计意图；探针注明所跟主任务 |

## 7. 扩展指南

### 增加新变体

1. 选择既有 `family`，复制该族请求样板，保证样板文字逐字符不变。
2. 只替换内联实例，如代码块、数据或主题。
3. 按任务实际执行方式标 GT，不按消息长度标注。
4. 提供 3–5 条针对新实例的可核验 checklist。
5. 将变体插入与同族上一条至少相隔 3 个主任务位置的位置；计算间隔时忽略 continuation。
6. 若插入 continuation，必须紧跟 medium/reasoning 主任务，且 GT 与前一行相同。

### 增加新族

1. 定义稳定、可复用的请求样板和清晰主题边界。
2. 至少设计多个仅实例不同的变体，以形成同签名学习信号。
3. 依据路由标准逐条确定 GT 分布；只有明确并行委派/子代理编排才使用 complex。
4. 更新族表、GT 合计、误导面清单和完整序列表。
5. 重新运行逐行 JSON、行数、ID、档位、checklist、族/变体、间隔和探针继承校验。
