#!/usr/bin/env bash
# 下载开放语料 zh-meme-sft-8k（MIT 协议）：https://huggingface.co/datasets/GaryYang123/zh-meme-sft-8k
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://huggingface.co/datasets/GaryYang123/zh-meme-sft-8k/resolve/main
curl -sL "$BASE/test.jsonl" -o data/corpus/raw/zh_meme_test.jsonl
curl -sL "$BASE/train.jsonl" -o data/corpus/raw/zh_meme_train.jsonl
wc -l data/corpus/raw/*.jsonl
