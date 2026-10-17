# AI 字幕翻译服务质量评测

[![CI](https://github.com/peiyaoc48-bit/ai-subtitle-test/actions/workflows/ci.yml/badge.svg)](https://github.com/peiyaoc48-bit/ai-subtitle-test/actions/workflows/ci.yml)

为一个基于大模型（DeepSeek）的字幕翻译服务搭建质量保障体系：接口与格式测试、LLM-as-Judge 翻译质量评测、评测数据集构建、Prompt 迭代与回归、CI 自动化。

## 结果摘要

### Prompt 迭代（每个版本跑 3 次取均值）

| 版本 | 主要改动 | 单句评测集（45 条）| 上下文评测集（7 条）|
|---|---|---|---|
| v1 | 基础翻译指令 | 76%（1 次）| — |
| v2 | 加"口语化、简洁" | 76%（1 次），网络梗 ↑ 但语气情绪 ↓ | — |
| v3 | 去掉"简洁"，要求不漏译不添加、保留语气、意译网络梗 | **77.8%**（69% / 80% / 84%）| — |
| v4 | "直译 → 反思 → 意译"三步，JSON 输出 | **85.9%**（84% / 87% / 87%）| 23.8% |
| v5 | v4 + 上下文窗口（前 2 句、后 1 句）| 84.4%（80% / 87% / 87%），与 v4 差异在波动范围内 | **95.2%** |

| 成本（单句评测集，每条平均）| v3 | v4 | v5 |
|---|---|---|---|
| 输入 token | 143 | 484 | 646 |
| 输出 token | 16 | 105 | 104 |
| 耗时 | 0.93 s | 1.53 s | 1.24 s |

### 关键发现

1. **大模型输出失控缺陷**：62 个"呜"的字幕被翻译成原文 528 倍长的重复文本，并导致同条后半句"李娜真坏"漏译。新增译文长度比例校验用例复现后，通过 `max_tokens` 上限 + 连续重复字符预处理修复，单次耗时 46 s → 12 s。
2. **单次评测不可靠**：同一 v3 Prompt 三次运行结果为 84% / 69% / 80%，波动 15 个百分点。之后所有版本对比改为 3 次运行取均值；v4 的结构化思考将波动收窄到 3 个百分点。
3. **只看总分会掩盖问题**：v1 与 v2 总通过率相同（76%），按分类拆开后，v2 网络梗类 +15%、语气情绪类 −18%，定位到"简洁"指令导致信息压缩。
4. **逐条翻译丢失上下文**：不给上文时，"（我妈打电话来）说今晚不回来吃饭了"被译成 *He said…*。v5 引入上下文窗口后修复，上下文类 23.8% → 95.2%，单句集回归无显著退步。
5. **LLM 筛选器过度判定**：让大模型判断"是否需要上下文"，60 组判了 58 组（把"含网络梗"也算进去）；改用**差分测试**（同一句不给上文 / 给上文各译一次，只保留译文有实质差异的）后只剩 7 组，人工审核再采纳 4 组。
6. **预期结果本身也会错**：大模型生成的评测要求曾把"家里有矿"的直译 *mine at home* 写成标准答案。人工审核不仅判断用例有没有价值，也校验预期结果是否正确。

## 评测数据集构建

```
开源语料 zh-meme-sft-8k（MIT）1042 句
  → collect.py 长度筛选、去重、抽样 50 句
  → label.py 大模型分类并生成翻译要点
  → review_web.py 网页人工审核（采纳 40/50 = 80%，人工修改 3 条）
  → merge.py 并入评测集：45 条，7 个分类

对话语料 → context_candidates.py 大模型初筛（58/60）
  → context_filter.py 差分测试（7/58）→ 人工审核（4/7）
  → 加 3 条手写用例：上下文评测集 7 条
```

## 项目结构

```
app/                    被测服务（FastAPI 字幕翻译接口）
  main.py               接口：POST /translate、GET /health
  srt.py                SRT 解析与序列化
  translator.py         翻译器：mock / 真实大模型；JSON 输出解析、上下文窗口、成本统计
prompts/                Prompt v1 – v5
data/
  samples/              测试字幕（含超长重复、HTML 标签、多行等边界数据）
  eval/                 评测集：cases.json（单句）、context_cases.json（上下文）
  corpus/               语料抽取、标注、审核的中间结果
tests/
  test_api.py           接口测试：正常流程、参数校验、异常输入
  test_srt_format.py    格式测试：条数、序号、时间轴、换行、译文长度比例
  test_preprocess.py    预处理单元测试
  quality/              翻译质量评测（DeepEval，裁判模型为 DeepSeek）
scripts/                语料流水线、审核工具、评测报告
reports/                每次评测的 JUnit 结果、译文明细与成本统计
.github/workflows/      GitHub Actions：提交即回归，结果推送到微信
```

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # mock 模式，不调用大模型
uvicorn app.main:app --reload           # 启动服务，打开 http://127.0.0.1:8000/docs
```

### 接入真实大模型与质量评测

```bash
cp .env.example .env                    # 填入 DeepSeek API Key（.env 不会被提交）
set -a; source .env; set +a
pip install deepeval
PROMPT_VERSION=v5 pytest tests/quality --junitxml=reports/eval_v5.xml
python scripts/eval_report.py v4 v5     # 按分类对比各版本
```

## 进度

- [x] 被测服务、接口测试、格式测试、CI 与微信推送
- [x] 边界数据测试，发现并修复输出失控缺陷
- [x] 评测数据流水线与 Web 审核工具
- [x] Prompt v1 – v5 迭代，3 次运行取均值
- [x] 上下文窗口与差分测试
- [ ] promptfoo 对比报告（配置已写，尚未运行）
- [ ] 安全测试：garak 扫描 Prompt 注入、越狱
- [ ] 性能测试：Locust 压测并发与首字延迟

## 致谢

- 评测语料：[zh-meme-sft-8k](https://huggingface.co/datasets/GaryYang123/zh-meme-sft-8k)（MIT）
- v4 / v5 Prompt 设计参考：[VideoLingo](https://github.com/Huanshere/VideoLingo)（Apache-2.0）、[translation-agent](https://github.com/andrewyng/translation-agent)（MIT）
