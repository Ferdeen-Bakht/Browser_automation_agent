# Browser Agent (Playwright + Ollama Qwen3.5 4B)

Single-page Bootstrap UI where you type a task and a local LLM drives a real Chromium browser.

## Setup
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m playwright install chromium
ollama pull qwen3.5:4b
```

## Run
```
.venv\Scripts\python server.py
```
Open http://localhost:8765

## How it works
Each step: page is scanned for visible interactive elements (numbered), the URL/text/elements are sent to the model,
which replies with one JSON action (goto/click/type/press/scroll/back/wait/done). Playwright executes it, a screenshot
streams to the UI over WebSocket, repeat until `done`.

A 4B model is small: keep tasks simple and specific, and mention a site (e.g. "on duckduckgo.com").
