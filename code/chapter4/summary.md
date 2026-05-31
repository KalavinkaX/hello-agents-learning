用一个生活场景贯穿三种方式：**假设你是一个学生，要做一份"关于特斯拉公司最新动态"的调研报告。**

---

## 1. ReAct — 边想边做

> 核心思想：**想一步，做一步，再想一步，再做一步**……循环往复

类比：**像侦探破案**，不是提前知道所有答案，而是每找到一条线索，就思考下一步该查什么。

```
第1步:
  Thought: 用户要了解特斯拉最新动态，我需要搜索一下
  Action:  Search[特斯拉2025年最新消息]
  Observation: 特斯拉发布了新款Model Y，Q1营收增长...

第2步:
  Thought: 已经知道新车型了，用户还问了公司动态，我再搜搜
  Action:  Search[特斯拉2025年股价和市场表现]
  Observation: 特斯拉股价波动，Robotaxi计划推迟...

第3步:
  Thought: 我已经收集了足够的信息，可以回答了
  Action:  Finish[特斯拉最新动态包括：1.发布新款Model Y...]
```

对应你项目代码的循环（`ReAct.py` 第 40-95 行）：

```python
while current_step < self.max_steps:
    # 1. 把"之前做了什么"告诉 LLM，让它想下一步
    prompt = REACT_PROMPT_TEMPLATE.format(
        tools=tools_desc,       # 有哪些工具可以用
        question=question,      # 用户的问题
        history=history_str     # 之前做了什么（Action + Observation）
    )
    
    # 2. LLM 思考，返回 Thought + Action
    response_text = self.llm_client.think(messages=messages)
    
    # 3. 如果是 Finish → 结束；否则 → 执行工具，把结果记入 history
    if action.startswith("Finish"):
        return final_answer
    
    observation = tool_function(tool_input)  # 真正调用搜索工具
    self.history.append(f"Action: {action}")
    self.history.append(f"Observation: {observation}")
```

**特点**：能调用外部工具（搜索、计算等），适合需要**实时信息**的任务。

---

## 2. Plan-and-Solve — 先规划，后执行

> 核心思想：**先把大象切成块，再一块一块处理**

类比：**像项目经理写方案**，先列出所有步骤，然后按顺序一个个执行。

以项目里的任务为例（`Plan_and_solve.py` 第 156 行的任务）：

```
问题: 周一卖了15个苹果，周二是周一两倍，周三比周二少5个，三天总共卖了多少？

第一步 - 规划（Planner 生成计划）:
  ["计算周一的销量",
   "计算周二的销量",
   "计算周三的销量", 
   "计算三天总销量"]

第二步 - 逐步执行（Executor 逐步执行）:
  步骤1: 计算周一 → 结果: 15个
  步骤2: 计算周二 → 结果: 15×2=30个
  步骤3: 计算周三 → 结果: 30-5=25个
  步骤4: 计算总和 → 结果: 15+30+25=70个
```

对应代码，分两个角色：

**Planner**（规划器）— 第 32-64 行：
```python
class Planner:
    def plan(self, question):
        # 让 LLM 输出一个 Python 列表作为计划
        # 输入: "周一卖15个，周二两倍，周三少5个..."
        # 输出: ["计算周一销量", "计算周二销量", ...]
        plan_str = response_text.split("```python")[1].split("```")[0].strip()
        plan = ast.literal_eval(plan_str)  # 字符串转列表
        return plan
```

**Executor**（执行器）— 第 87-120 行：
```python
class Executor:
    def execute(self, question, plan):
        for i, step in enumerate(plan):
            # 每一步都会把"原始问题 + 完整计划 + 之前的结果"一起给 LLM
            # 这样 LLM 知道上下文，能正确执行当前步骤
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
```

**特点**：结构清晰，适合**逻辑推理、数学计算**等有明确步骤的任务。但**不能调用外部工具**。

---

## 3. Reflection — 写稿 → 审稿 → 改稿

> 核心思想：**先写初稿，然后自己当评委批评自己，再根据批评修改，反复迭代**

类比：**像程序员写代码 + Code Review**，写了代码后让同事审查，审查出问题就改，改完再审，直到没问题。

```
任务: 写一个找素数的函数

第一轮 - 写代码:
  LLM生成: 用试除法判断每个数是否为素数  ← 能用但效率低

第二轮 - 反思(当评委):
  评审员: "试除法时间复杂度O(n√n)，应该用埃拉托斯特尼筛法O(nloglogn)"

第三轮 - 优化(根据反馈改):
  LLM生成: 改用筛法  ← 效率大幅提升

第四轮 - 再次反思:
  评审员: "算法已最优，无需改进"  ← 停止！
```

对应代码（`Reflection.py`），三个提示词模板扮演不同角色：

```python
# 角色1: 程序员（写代码）
INITIAL_PROMPT_TEMPLATE = "你是一位资深的Python程序员，请编写..."

# 角色2: 评审员（挑毛病）
REFLECT_PROMPT_TEMPLATE = "你是一位极其严格的代码评审专家...分析时间复杂度..."

# 角色3: 程序员根据反馈改代码
REFINE_PROMPT_TEMPLATE = "根据评审员的反馈，生成优化后的代码..."
```

核心循环（第 119-142 行）：
```python
for i in range(self.max_iterations):
    # 1. 反思：让 LLM 扮演评审员，审查最新代码
    feedback = self._get_llm_response(reflect_prompt)
    
    # 2. 如果评审员说"无需改进"，结束
    if "无需改进" in feedback:
        break
    
    # 3. 优化：让 LLM 根据反馈重写代码
    refined_code = self._get_llm_response(refined_prompt)
```

**特点**：能**自我改进**，代码/文本质量会随迭代提升。但不能调用外部工具。

---

## 三种方式对比总结

|              | ReAct                   | Plan-and-Solve               | Reflection                |
| ------------ | ----------------------- | ---------------------------- | ------------------------- |
| **一句话**   | 边想边做                | 先规划后执行                 | 写了改，改了写            |
| **流程**     | 思考→行动→观察→思考→... | 规划→执行步骤1→执行步骤2→... | 写初稿→评审→优化→评审→... |
| **能用工具** | ✅ 能调用搜索等          | ❌                            | ❌                         |
| **自我改进** | ❌                       | ❌                            | ✅ 多轮迭代优化            |
| **适合场景** | 需要实时信息的问题      | 步骤明确的推理任务           | 代码/文本生成与优化       |
| **对应类比** | 侦探破案                | 项目经理排计划               | 程序员 + Code Review      |