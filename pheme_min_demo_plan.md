# PHEME 最小可行 Demo 实验流程

## 1. 目标

在 **最少代码量** 前提下，用 PHEME 快速验证这条思路是否可行：

- 小模型先做 **事件级快速判断**
- 只把 **低置信样本** 送给 LLM
- LLM 读取 **压缩后的传播上下文**，输出结构化判决
- 对比 baseline，看是否能在 **有限调用率** 下提升 Macro-F1 / Recall

> 第一版建议只做 **binary rumor detection（rumor / non-rumor）**。
> 不建议一开始就做 rumor veracity 三分类（true / false / unverified），因为实现和数据清洗都会更复杂。

---

## 2. 推荐方案

### 方案 A：首选（最推荐）

- **Stage 1 baseline**：BiGCN
- **Stage 2 LLM**：API 模型（优先 GPT-5.4，调试阶段可先用 GPT-5.4-mini）
- **Routing**：按小模型置信度阈值
- **Evidence packer**：先用启发式，不训练 selector

优点：

- 和你的传播结构思路一致
- baseline 有论文和公开代码
- 不需要你先把完整 selector / distillation 都做完
- 只做一层最薄的“路由 + 证据压缩 + LLM判决”就能证明方向可行

### 方案 B：兜底

如果 BiGCN 老代码跑不通，就退一步：

- Stage 1 改成 `deberta-v3-base` 或 `roberta-base` 的文本分类器
- 输入只用 `source post + 简单统计特征`
- Stage 2 仍然保留 LLM 证据推理

这样说服力会弱一点，但非常容易跑通。

---

## 3. 建议对比实验

### Exp-0：Text-only baseline（可选）

- 输入：source tweet
- 模型：DeBERTa / RoBERTa
- 输出：rumor / non-rumor

### Exp-1：Propagation baseline（主 baseline）

- 输入：source tweet + propagation tree
- 模型：BiGCN
- 输出：rumor / non-rumor

### Exp-2：你的最小 demo

- Stage 1：BiGCN
- Routing：仅将低置信事件送入 LLM
- LLM 输入：source + 压缩后的 replies + propagation summary
- 输出：结构化 verdict
- 最终结果：
  - 高置信样本：直接用 BiGCN
  - 低置信样本：用 LLM 输出覆盖

### Exp-3：阈值消融

- threshold = 0.55 / 0.65 / 0.75
- 报告：
  - Macro-F1
  - Accuracy
  - LLM invocation rate
  - avg tokens per event

---

## 4. 推荐目录结构

```text
project/
├── data/
│   ├── raw/PHEME/                  # 你下载的原始 PHEME
│   ├── processed/
│   │   ├── events.jsonl
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
├── baselines/
│   ├── BiGCN/                      # 官方或可运行实现
│   └── text_cls/                   # 兜底文本分类器
├── scripts/
│   ├── 01_build_events.py
│   ├── 02_make_splits.py
│   ├── 03_train_small_model.sh
│   ├── 04_predict_small_model.py
│   ├── 05_pack_evidence.py
│   ├── 06_run_llm.py
│   ├── 07_merge_predictions.py
│   └── 08_eval.py
├── prompts/
│   └── rumor_verdict.txt
├── outputs/
│   ├── small_model/
│   ├── packed_events/
│   ├── llm_outputs/
│   └── metrics/
├── configs/
│   └── api_config.yaml
└── README.md
```

---

## 5. 数据处理：只做事件级样本，不做复杂图增强

PHEME 原始目录里通常有：

- `source-tweet/`
- `reactions/`
- `structure.json`
- `annotation.json`

### 你第一版只抽这几类字段

```json
{
  "event_id": "...",
  "label": 0,
  "source_text": "...",
  "replies": [
    {"tweet_id": "...", "text": "...", "parent": "...", "time": "..."}
  ],
  "structure": {...},
  "meta": {
    "num_replies": 23,
    "max_depth": 4,
    "num_branches": 6
  }
}
```

### `01_build_events.py` 要做的事情

1. 遍历 PHEME 全部事件目录
2. 读取 source tweet
3. 读取 reactions
4. 读取 `structure.json`
5. 生成 event-level 样本
6. 输出 `events.jsonl`

---

## 6. 标签建议

### 第一版：binary rumor detection

你可以直接把：

- `rumours/` -> 1
- `non-rumours/` -> 0

这样最简单。

### 不建议第一版就做的内容

- true / false / unverified 三分类
- stance 多任务
- 用户图
- 外部检索
- teacher-student 蒸馏

这些都留到第二轮。

---

## 7. Stage 1：小模型 baseline

## 7.1 方案 A：BiGCN

如果你直接用公开实现，尽量少改，只做：

- 数据适配到 repo 需要的输入格式
- 训练 / 推理
- 导出每个 event 的：
  - `pred_label`
  - `prob`
  - `margin`

### 需要导出的最小预测文件

```json
{
  "event_id": "...",
  "gold": 1,
  "pred": 0,
  "prob": [0.42, 0.58],
  "confidence": 0.58,
  "margin": 0.16
}
```

### routing 建议

```python
route_to_llm = confidence < threshold
```

binary 任务里：

```python
confidence = max(prob)
```

---

## 7.2 方案 B：文本分类器兜底

如果 BiGCN 旧代码环境太折腾，就直接用 HuggingFace：

- `microsoft/deberta-v3-base`
- 或 `roberta-base`

输入：

```text
[CLS] source_text [SEP]
```

这样当天就能跑出一个 baseline。

---

## 8. Routing：最简单就用置信度阈值

先不要做 learned router。

### 推荐阈值

- 开始先试：`0.65`
- 然后做三档：`0.55 / 0.65 / 0.75`

### 路由逻辑

```python
if confidence >= threshold:
    final_pred = small_model_pred
else:
    final_pred = llm_pred
```

你第一版只需要证明：

- 小模型已经能处理大部分容易样本
- LLM 只处理少量难样本
- 总体效果提升 or 某些关键类别 Recall 提升

---

## 9. Evidence Packer：第一版不要训练，直接启发式

为了最低代码量，**先不训练 selector**。

### 第一版 packer 直接选这些内容

1. `source_text`
2. 最早出现的 3 条 replies
3. 最长的 3 条 replies
4. 来自不同 branch 的 2 条 replies
5. propagation summary

如果 reply 不够，就按实际数量补齐。

### propagation summary 建议字段

```json
{
  "num_replies": 23,
  "max_depth": 4,
  "num_branches": 6,
  "early_reply_count": 5
}
```

### `05_pack_evidence.py` 输出格式

```json
{
  "event_id": "...",
  "label": 1,
  "source_text": "...",
  "selected_replies": [
    "reply a ...",
    "reply b ...",
    "reply c ..."
  ],
  "propagation_summary": {
    "num_replies": 23,
    "max_depth": 4,
    "num_branches": 6
  },
  "small_model": {
    "pred": 0,
    "confidence": 0.58
  }
}
```

---

## 10. LLM 选择建议

## API 方案：优先推荐

### 推荐顺序

1. `gpt-5.4`：正式实验
2. `gpt-5.4-mini`：调试 / 小规模验证

### 原因

- 不用自己部署
- 模型新
- 代码量最低
- 你现在重点不是省钱，而是证明 idea 可行

---

## 本地方案：只作为备选

可以考虑：

- `Qwen3-32B`
- 或 `Qwen3-30B-A3B`

部署方式：

- `vLLM`

适合场景：

- API 不方便
- 你后面要大量批量推理
- 你想可控地反复调 prompt

但第一版我仍建议 **先 API，后本地**。

---

## 11. Prompt：直接用结构化输出

`prompts/rumor_verdict.txt`

```text
You are a misinformation detection assistant.
Given a social media source post, several representative replies, and propagation statistics,
judge whether this event is a rumor.

Return valid JSON only.

Fields:
- label: "rumor" or "non-rumor"
- confidence: float between 0 and 1
- evidence_for: list of short strings
- evidence_against: list of short strings
- propagation_summary: short string
- final_rationale: short paragraph

Input:
[Source Post]
{source_text}

[Representative Replies]
{selected_replies}

[Propagation Summary]
{propagation_summary}

[Small Model Hint]
small_model_pred={small_model_pred}
small_model_confidence={small_model_confidence}
```

### 注意

- 第一版就让模型输出 JSON
- 不要 free-form 长文
- 方便后处理和错误分析

---

## 12. `06_run_llm.py` 伪代码

```python
for event in routed_events:
    prompt = render_prompt(event)
    rsp = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=0
    )
    save_json(event_id, rsp)
```

如果你走兼容 OpenAI 的中转：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)
```

---

## 13. 合并结果

`07_merge_predictions.py`

```python
if small_model_confidence >= threshold:
    final = small_model_pred
else:
    final = map_llm_label(llm_json["label"])
```

保留以下字段：

```json
{
  "event_id": "...",
  "gold": 1,
  "small_pred": 0,
  "small_conf": 0.58,
  "used_llm": true,
  "llm_pred": 1,
  "final_pred": 1
}
```

---

## 14. 评价指标

### 必报

- Accuracy
- Macro-F1
- rumor 类 Recall
- LLM invocation rate
- avg tokens per routed sample

### 如果你想更像论文

再加：

- cost per 100 events
- coverage-risk curve

但第一版不是必须。

---

## 15. 最小可交付结果

你第一轮只要交出这 4 个表，基本就能说明问题：

### 表 1：主结果

| Method | Accuracy | Macro-F1 | Rumor Recall |
|---|---:|---:|---:|
| Text-only |  |  |  |
| BiGCN |  |  |  |
| BiGCN + LLM Cascade |  |  |  |

### 表 2：成本

| Threshold | LLM Rate | Avg Tokens | Macro-F1 |
|---|---:|---:|---:|
| 0.55 |  |  |  |
| 0.65 |  |  |  |
| 0.75 |  |  |  |

### 表 3：错误案例

- small model 错，LLM 改对
- small model 对，LLM 改错
- 两者都错

### 表 4：case study

展示 2~3 个事件：

- source
- selected replies
- small model 低置信
- LLM structured verdict
- 最终为何修正成功

---

## 16. 最小开发顺序

### Day 1

1. 跑通 PHEME -> `events.jsonl`
2. 跑通 baseline（优先 BiGCN）
3. 导出 test set 预测概率

### Day 2

1. 实现 evidence packer
2. 接 API 跑 routed events
3. 合并结果并评估

### Day 3

1. 做 threshold ablation
2. 整理表格
3. 写方法图和实验说明

---

## 17. 我对这版 demo 的定义

这版不是完整论文系统，而是一个 **可辩护的 feasibility demo**：

- 有公开 baseline
- 有最薄路由逻辑
- 有最薄证据压缩
- 有结构化 LLM 判决
- 有成本-效果对比

只要 `BiGCN + selective LLM` 比 `BiGCN` 在合理调用率下更好，
你的核心叙事就站住了。

---

## 18. 你后续可以再升级的点

等第一版跑通后，再考虑：

- 把启发式 packer 换成 learned selector
- 把 binary 改成 veracity classification
- 增加 stance-aware packing
- 增加 consistency check
- 增加蒸馏回流

