import asyncio
import os
import shutil
import subprocess
from pathlib import Path
import base64
import json
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import db
from agent import Agent, BrowserSession, DEFAULT_MODEL, DEFAULT_ENGINE, chrome_running, launch_chrome

app = FastAPI()
session = BrowserSession()  # one persistent browser for the whole server process
current = {"task": None, "agent": None}
VIEW_W, VIEW_H = 1280, 800  # must match the browser viewport in agent.py
session.chat_id = None  # row in chats.db for the conversation on screen (created with its first message)


@app.get("/")
async def index():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})  # always serve the latest UI


@app.get("/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get("http://localhost:11434/api/tags")
        return {"models": [m["name"] for m in r.json()["models"]], "default": DEFAULT_MODEL, "engine": DEFAULT_ENGINE}
    except Exception:
        return JSONResponse({"models": [], "default": DEFAULT_MODEL, "engine": DEFAULT_ENGINE,
                             "error": "Ollama not reachable"})


@app.get("/chrome/status")
async def chrome_status():
    return {"running": await chrome_running()}


@app.post("/chrome/launch")
async def chrome_launch():
    try:
        await launch_chrome()
        return {"ok": True, "note": "Chrome is running. Sign in to your accounts in it once; they stay saved for next time."}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/chats")
async def chats():
    return {"chats": db.list_chats(), "current": session.chat_id}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    db.delete_chat(chat_id)
    if session.chat_id == chat_id:
        session.chat_id = None
        session.reset_chat()
    return {"ok": True}


def history_msg():
    msgs = db.get_messages(session.chat_id) if session.chat_id else []
    return {"type": "history", "messages": msgs, "busy": session.busy, "chat_id": session.chat_id}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()

    async def emit(msg):
        if msg.get("type") == "answer" and session.chat_id:  # persist every final answer
            db.add_message(session.chat_id, "bot", msg["text"], msg.get("links"))
        try:
            await sock.send_text(json.dumps(msg))
        except Exception:
            pass

    async def runner(a, goal):
        session.busy = True
        try:
            await a.run(goal)
        except asyncio.CancelledError:  # Stop pressed: the task was cancelled mid-step
            await emit({"type": "answer", "text": "Stopped by user.", "links": []})
        except Exception as e:
            await emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
        session.busy = False
        await emit({"type": "finished"})

    await emit(history_msg())
    remote = {"task": None}

    async def remote_stream():
        """While the 'take over' popup is open, stream screenshots so the user sees what they type into."""
        while True:
            p = session.live_page() if session.ctx else None
            if p:
                try:
                    b = await p.screenshot(type="jpeg", quality=60)
                    await emit({"type": "screenshot", "data": base64.b64encode(b).decode(), "url": p.url})
                except Exception:
                    pass
            await asyncio.sleep(0.7)

    def stop_remote():
        if remote["task"]:
            remote["task"].cancel()
            remote["task"] = None

    try:
        while True:
            m = json.loads(await sock.receive_text())
            t = current["task"]
            if m["type"] == "run" and not (t and not t.done()):
                if not session.chat_id:
                    session.chat_id = db.create_chat(m["task"])
                db.add_message(session.chat_id, "user", m["task"])
                session.mode = "chrome" if m.get("browser") == "chrome" else "test"
                a = Agent(session, emit, model=m.get("model") or DEFAULT_MODEL, headless=bool(m.get("headless")),
                          max_steps=int(m.get("max_steps") or 25), engine=m.get("engine") or DEFAULT_ENGINE,
                          location=m.get("location"))
                current["agent"] = a
                current["task"] = asyncio.create_task(runner(a, m["task"]))
            elif m["type"] == "answer":  # reply to an ask_user question from the agent
                session.provide_answer(m.get("text", ""))
            elif m["type"] == "remote_start":
                stop_remote()
                remote["task"] = asyncio.create_task(remote_stream())
            elif m["type"] == "remote_stop":
                stop_remote()
            elif m["type"].startswith("remote_") and session.ctx:
                # The user's own mouse/keyboard input, relayed to the page. Never logged, stored or shown to the model.
                try:
                    pg = session.live_page()
                    if session.page is not pg:
                        session.page = pg
                    # popup windows (Google sign-in...) have their own size: map the click to THIS page's viewport
                    vs = pg.viewport_size or await pg.evaluate("({width: innerWidth, height: innerHeight})")
                    if m["type"] == "remote_click":
                        await pg.mouse.click(float(m["x"]) * vs["width"], float(m["y"]) * vs["height"])
                    elif m["type"] == "remote_text":
                        await pg.keyboard.type(str(m["text"]))
                    elif m["type"] == "remote_key":
                        await pg.keyboard.press(str(m["key"]))
                    elif m["type"] == "remote_scroll":
                        await pg.mouse.wheel(0, float(m["dy"]))
                except Exception as e:  # tell the user instead of failing silently
                    await emit({"type": "status", "text": f"Browser input failed: {type(e).__name__}"})
            elif m["type"] == "stop":
                session.stop_flag = True  # lets a waiting ask_user return...
                tk = current["task"]
                if tk and not tk.done():
                    tk.cancel()  # ...and cancels a running model call / page action right away
            elif m["type"] == "new_chat":
                session.reset_chat()  # the finished conversation stays saved in chats.db
                session.chat_id = None
                await emit(history_msg())
            elif m["type"] == "load_chat" and not session.busy:
                session.reset_chat()
                session.chat_id = int(m["id"])
                msgs = db.get_messages(session.chat_id)  # rebuild the agent's context from the saved conversation
                last_user = None
                for mm in msgs:
                    if mm["role"] == "user":
                        last_user = mm["text"]
                    elif last_user is not None:
                        session.chat.append((last_user, mm["text"]))
                        last_user = None
                await emit(history_msg())
            elif m["type"] == "close_browser":
                session.stop_flag = True
                await session.close()
                await emit({"type": "status", "text": "Browser closed."})
    except WebSocketDisconnect:
        pass
    finally:
        stop_remote()  # keep the browser and any running task alive


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8766)
