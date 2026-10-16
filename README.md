# AI 字幕翻译服务质量评测

对一个基于大模型的字幕翻译服务做全方位质量保障：接口测试、格式正确性、翻译质量评测（LLM-as-Judge）、Prompt 版本对比、CI 自动回归。

## 项目结构

```
app/                    被测服务（FastAPI 字幕翻译接口）
  main.py               接口：POST /translate、GET /health
  srt.py                SRT 解析与序列化
  translator.py         翻译器（mock 模式 / 真实大模型模式）
prompts/                Prompt 版本（v1、v2）
data/samples/           测试字幕
tests/
  test_api.py           接口测试：正常流程、参数校验、异常输入
  test_srt_format.py    格式测试：条数、序号、时间轴、换行不能被翻译破坏
  quality/              翻译质量评测（DeepEval，需要真实调用大模型）
promptfoo/              Prompt v1 vs v2 对比评测
.github/workflows/      GitHub Actions：每次提交自动跑回归
```

## 快速开始

```bash
cd ~/Desktop/ai-subtitle-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest                                  # mock 模式，不花钱
uvicorn app.main:app --reload           # 启动服务，打开 http://127.0.0.1:8000/docs
```

## 接入真实大模型

```bash
cp .env.example .env                    # 填入 DeepSeek API Key
set -a; source .env; set +a
export TRANSLATOR_MODE=llm
pytest tests/test_srt_format.py         # 真实模型下格式还能 100% 正确吗？
pip install deepeval && pytest tests/quality
```

## Allure 报告

```bash
pytest --alluredir=allure-results
allure serve allure-results             # 需先安装：brew install allure
```

## 进度

- [x] 被测服务 + mock 模式
- [x] 接口测试、格式测试骨架
- [ ] 补充异常用例（见 test_api.py 末尾 TODO）
- [ ] 接入 DeepSeek，跑通真实翻译
- [ ] 评测集扩充到 100 条，v1/v2 对比
- [ ] promptfoo 对比报告
- [ ] 推到 GitHub，CI 跑绿
- [ ] （第二阶段）garak 安全扫描、Locust 性能测试
