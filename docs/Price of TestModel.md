# CPA 实际计费口径（Test Models）

> 版本：v1.0（2026-09-11，项目所有者提供）｜消费方：`project-decisions.md` D11 定价表；会话 1 写入 `eval/pricing.json`
> 单位：$/Mtok（输入 / cache 输入 / 输出）；MiniCPM5-1B 本地 Judge 成本记 0（本地推理无边际计费）

|      |                         |                                     |                                                   | Price/$（输入/M，cache输入/M，输出/M） |
| ---- | ----------------------- | ----------------------------------- | ------------------------------------------------- | -------------------------------------- |
| 1    | CPA/glm-5.3-flash       | simple 档执行                       | episode 成本                                      | (0.15, 0.03, 0.50)                     |
| 2    | `CPA/opencode-v4-flash` | medium 档执行                       | episode 成本                                      | (0.3, 0.006, 1.20)                     |
| 3    | `CPA/glm-5.3`           | reasoning 档执行 + **A 组静态路由** | episode 成本 + **cost_saving 的参考基线（分母）** | (1.4, 0.26, 4.40)                      |
| 4    | `CPA/OpenBMB-5.3`       | complex 档执行                      | episode 成本                                      | (1.4, 0.26, 4.40)                      |
| 5    | `CPA/gpt-5.6-sol`       | Evaluator（三组对称调用）           | 评估成本（计入预算核算，不计入 episode 路由成本） | (4.00, 0.40, 20.00)                    |
