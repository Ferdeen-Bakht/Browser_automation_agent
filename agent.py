"""Browser agent. A persistent Playwright session is driven by a local Ollama model (Qwen3.5 4B).

The browser stays open between chat prompts, so a follow-up continues on the page you are already on.
"""
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import datetime as dt
from urllib.parse import quote, quote_plus, urlparse, parse_qs

import httpx
from playwright.async_api import async_playwright

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
CDP_URL = "http://127.0.0.1:9222"  # the user's own Chrome, started with --remote-debugging-port=9222
CHROME_PROFILE = Path(__file__).with_name("chrome-profile")  # Chrome refuses remote debugging on its default profile


def find_chrome():
    for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        p = Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe" if base else None
        if p and p.exists():
            return str(p)
    return shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")


async def chrome_running():
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            return (await c.get(CDP_URL + "/json/version")).status_code == 200
    except Exception:
        return False


async def launch_chrome(wait=25):
    """Start the user's real Chrome with remote debugging (own profile folder, sign in once). No-op if it is already up."""
    if await chrome_running():
        return
    exe = find_chrome()
    if not exe:
        raise RuntimeError("Could not find Chrome. Install it or start it yourself with --remote-debugging-port=9222.")
    subprocess.Popen([exe, "--remote-debugging-port=9222", f"--user-data-dir={CHROME_PROFILE}", "--no-first-run",
                      "--no-default-browser-check", "about:blank"])
    for _ in range(wait * 2):
        if await chrome_running():
            return
        await asyncio.sleep(0.5)
    raise RuntimeError("Chrome was started but did not open its debugging port in time. Close all Chrome windows that use the "
                       "agent profile and try again.")


DEFAULT_ENGINE = "bing"
ENGINES = {
    "bing": ("Bing", "https://www.bing.com/", "https://www.bing.com/search?q={q}"),
    "duckduckgo": ("DuckDuckGo", "https://duckduckgo.com/", "https://duckduckgo.com/?q={q}"),
    "google": ("Google", "https://www.google.com/", "https://www.google.com/search?q={q}"),
    "brave": ("Brave Search", "https://search.brave.com/", "https://search.brave.com/search?q={q}"),
    "startpage": ("Startpage", "https://www.startpage.com/", "https://www.startpage.com/do/search?q={q}"),
}

COLLECT_JS = """
(base) => {
  base = base || 0;  // 0 for the main page, 100/200/... for iframes, so ids stay unique across frames
  document.querySelectorAll('[data-agent-id]').forEach(e => e.removeAttribute('data-agent-id'));
  const sel = 'a[href],button,input,textarea,select,summary,[role=button],[role=link],[role=tab],[role=menuitem],[onclick],[contenteditable=true]';
  const out = []; let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (r.width < 4 || r.height < 4 || st.visibility === 'hidden' || st.display === 'none') continue;
    if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
    if (el.type === 'hidden') continue;
    let label = (el.innerText || el.textContent || el.getAttribute('aria-label') || el.placeholder || el.title || el.value || el.alt || el.name || '').trim().replace(/\\s+/g, ' ').slice(0, 90);
    if (!label && el.tagName === 'A' && /popup/i.test(el.href)) label = '(icon/popup button - opens a popup, may do nothing visible)';
    const id = base + n;
    el.setAttribute('data-agent-id', id);
    out.push({id, tag: el.tagName.toLowerCase(), type: el.type || '', label, href: el.tagName === 'A' ? el.href : ''});
    if (++n >= 60) break;
  }
  return {elements: out, text: (document.body.innerText || '').replace(/\\s+/g, ' ').slice(0, 3500),
          scroll: Math.round(100 * (scrollY + innerHeight) / Math.max(document.body.scrollHeight, 1))};
}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string",
                   "enum": ["search", "maps_search", "maps", "use_my_location", "goto", "click", "type", "press", "scroll", "back", "wait",
                            "ask_user", "done"]},
        "id": {"type": "integer"},  # element number as shown in [brackets]; iframe elements are numbered 100 and up
        "text": {"type": "string"},
        "url": {"type": "string"},
        "key": {"type": "string"},
        "direction": {"type": "string", "enum": ["down", "up"]},
        "answer": {"type": "string"},
        "links": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["thought", "action"],
}

SYSTEM = """You are a browser automation agent chatting with a user. You control a real web browser that STAYS OPEN between
messages, so if the user's request relates to the page already open, continue from it instead of searching again.
Each turn you see the current URL, page text, and a numbered list of clickable/typeable elements.
Reply with ONE JSON action:
- {"thought": "...", "action": "search", "text": "query"}          (web search with the preferred engine)
- {"thought": "...", "action": "maps_search", "text": "parks near me"}  (Google Maps PLACES search: lists places, no route)
- {"thought": "...", "action": "maps", "text": "destination"}         (Google Maps DIRECTIONS page to one specific destination)
- {"thought": "...", "action": "use_my_location"}                    (on a Google Maps directions page: sets the starting point to my precise location)
- {"thought": "...", "action": "goto", "url": "https://..."}
- {"thought": "...", "action": "click", "id": N}
- {"thought": "...", "action": "type", "id": N, "text": "...", "key": "Enter"}   (key optional)
- {"thought": "...", "action": "press", "key": "Enter"}
- {"thought": "...", "action": "scroll", "direction": "down"}
- {"thought": "...", "action": "back"} / {"thought": "...", "action": "wait"}
- {"thought": "...", "action": "ask_user", "text": "question for the user"}   (waits for the user's reply in chat)
- {"thought": "...", "action": "done", "answer": "...", "links": ["https://..."]}
Booking / forms rule: for reservations, bookings, orders or any form needing personal details, first find the booking page or widget
(look for Reserve / Book / Reservations links; widgets may be inside an iframe, whose elements are numbered 100 and up). NEVER invent personal
details or dates: use "ask_user" to ask for whatever is missing (date, time, party size, name, phone, email) in ONE question, then
fill the form with the answers. Before the final submit/confirm click, use "ask_user" to get an explicit yes.
Language: write "answer" and any ask_user question in the SAME language and script the user wrote in (Roman Urdu -> Roman Urdu, Arabic -> Arabic,
French -> French, ...). Web searches and text typed into websites stay in English unless the site itself needs another language.
Login walls: if a site asks the user to log in or sign up, NEVER enter credentials or create accounts; use "ask_user" ONCE to tell the user to sign in
themselves in the login window that opens and press Continue. Do not offer to click buttons for them.
NEVER ask permission to click, open, scroll or navigate - just do it. Use "ask_user" only for information you truly lack (dates, names, choices,
login) or for the final confirmation of a purchase/booking/submission.
Form fields: for a dropdown (<select>) use "type" with the option you want (e.g. "2" picks "1 - 2 Guests"); for date fields use "type" with
"tomorrow", "next friday", "12 March" or 2026-03-12; for time fields e.g. "7:30pm". Fill every visible field, then click the submit button.
Pop-ups: for cookie banners choose Reject / Necessary only / Close, never "Accept all". If a click fails twice, pick another element.
Maps rules: for "find/near me/nearby" requests (e.g. "parks near me") use "maps_search", read the listed places from the page text and
answer with the LIST (names, ratings, addresses, what each is) -- do NOT start directions; offer to give a route to any of them.
Only when the user asks for a route/directions/how to get to ONE specific place use "maps". If they gave ONLY a destination (no
starting point), after "maps" use "use_my_location" so the route starts from where the user is. If they gave a starting point,
type it into the starting-point box instead.
Rules: use only element ids from the current list. Keep "thought" to one short sentence. Never repeat a failing action.
To research: search, open 1-3 relevant result pages, READ their page text (scroll if needed), then answer.
When finishing with done, write a DETAILED answer (several sentences or bullet points with concrete facts, names, numbers,
dates, and what you found on which site) -- not a one-liner. In "links" list the full URLs of the pages that support the answer
(only URLs you actually saw in the elements list or the current URL)."""


DIRECTIONS_RE = re.compile(r"\b(directions?|route|navigate|navigation|how (do|can|to) (i|we)? ?get to|way to|drive to|walk to|"
                           r"distance (to|from)|travel time|take me to)\b", re.I)
PLACES_RE = re.compile(r"\b(near me|nearby|near by|around me|close to me|closest|nearest|around here)\b", re.I)


def real_url(href: str) -> str:
    """Decode Bing's tracking redirect (bing.com/ck/a?...&u=a1<base64>) into the real destination."""
    try:
        u = urlparse(href)
        if u.netloc.endswith("bing.com") and u.path.startswith("/ck/a"):
            v = parse_qs(u.query).get("u", [""])[0]
            if v.startswith("a1"):
                v = v[2:]
                return base64.urlsafe_b64decode(v + "=" * (-len(v) % 4)).decode()
    except Exception:
        pass
    return href


class BrowserSession:
    """Owns the long-lived browser, plus the chat memory shared across prompts."""

    def __init__(self):
        self.pw = self.browser = self.ctx = None
        self.page = None
        self.headless = False
        self.mode = "test"  # "test" = Playwright's own Chromium, "chrome" = the user's real Chrome over CDP
        self.active_mode = None
        self.my_tab = None
        self.chat = []  # [(user, assistant)]
        self.busy = False
        self.applied_loc = None  # last geolocation given to the browser context
        self.bad_hosts = {}  # site host -> why we gave up on it (bot check, errors); never revisited this chat
        self.blocked_engines = set()  # engines that showed a bot check this session; skipped from then on
        self.stop_flag = False
        self.client = httpx.AsyncClient(timeout=300)

    async def ensure(self, headless):
        if self.mode == "chrome":
            headless = False  # a real Chrome window is always visible
        if self.browser and self.browser.is_connected() and self.headless == headless and self.active_mode == self.mode:
            if self.page is None or self.page.is_closed():
                self.page = await self.ctx.new_page()
            return
        await self.close()
        self.headless = headless
        self.active_mode = self.mode
        self.pw = await async_playwright().start()
        if self.mode == "chrome":
            # Attach to the user's own, already signed-in Chrome. Google etc. do not block it (it is not an automated browser).
            try:
                await launch_chrome()  # starts Chrome by itself when it is not already running with the debugging port
                self.browser = await self.pw.chromium.connect_over_cdp(CDP_URL)
            except Exception as e:
                await self.pw.stop()
                self.pw = self.browser = None
                self.active_mode = None
                raise RuntimeError(f"Cannot use your Chrome: {e}")
            self.ctx = self.browser.contexts[0]
            self.applied_loc = None
            self.page = self.my_tab = await self.ctx.new_page()  # work in a fresh tab, leave the user's tabs alone
            self.ctx.on("page", self._on_new_page)
            return
        self.browser = await self.pw.chromium.launch(headless=headless, args=["--disable-dev-shm-usage", "--disable-gpu"])
        self.ctx = await self.browser.new_context(viewport={"width": 1280, "height": 800}, permissions=["geolocation"])
        self.applied_loc = None
        self.page = await self.ctx.new_page()
        self.ctx.on("page", self._on_new_page)  # follow new tabs / sign-in popup windows

    def _on_new_page(self, pg):
        self.page = pg
        # when a popup (e.g. a Google sign-in window) closes, go back to the tab that is still open
        pg.on("close", lambda *_: setattr(self, "page", self.live_page()) if self.page is pg else None)

    def live_page(self):
        """The current page, or the most recent still-open one if the current page was closed."""
        if self.page is not None and not self.page.is_closed():
            return self.page
        open_pages = [p for p in (self.ctx.pages if self.ctx else []) if not p.is_closed()]
        return open_pages[-1] if open_pages else self.page

    async def close(self):
        try:
            if self.active_mode == "chrome":
                # the user's Chrome stays open: only close the tab we opened and disconnect
                if self.my_tab and not self.my_tab.is_closed():
                    await self.my_tab.close()
            elif self.browser:
                await self.browser.close()
            if self.pw:
                await self.pw.stop()
        except Exception:
            pass
        self.pw = self.browser = self.ctx = self.page = self.my_tab = None
        self.active_mode = None

    def reset_chat(self):
        self.chat = []
        self.bad_hosts.clear()

    async def wait_answer(self):
        """Block until the user answers the agent's question (or presses Stop). Returns the text, or None if stopped."""
        self.answer_future = asyncio.get_running_loop().create_future()
        try:
            while not self.stop_flag:
                done, _ = await asyncio.wait({self.answer_future}, timeout=1)
                if done:
                    return self.answer_future.result()
            return None
        finally:
            self.answer_future = None

    def provide_answer(self, text):
        fut = getattr(self, "answer_future", None)
        if fut and not fut.done():
            fut.set_result(text)
            return True
        return False


# Click targets that dismiss pop-ups without agreeing to anything (cookie "reject", "close", "no thanks"). Never "accept".
DISMISS_JS = """
() => {
  const ok = /^(reject all|reject|reject non-essential|decline|decline all|deny|refuse|only necessary|necessary only|only essential|essential only|use necessary cookies only|continue without accepting|no thanks|no, thanks|not now|maybe later|close|dismiss|skip|x|\\u00d7|\\u2715)$/i;
  for (const el of document.querySelectorAll('button,a,[role=button],[aria-label]')) {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (r.width < 4 || r.height < 4 || st.visibility === 'hidden' || st.display === 'none') continue;
    const t = (el.innerText || el.getAttribute('aria-label') || '').trim();
    if (t.length < 30 && ok.test(t)) { el.click(); return t; }
  }
  return '';
}
"""


FIND_RESERVE_JS = """
() => {
  const rx = /^(reserve( a)? (now|table)|book( a)? (now|table)|book|reserve|reservations?|make a reservation)$/i;
  for (const el of document.querySelectorAll('a[href],button,[role=button]')) {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (r.width < 20 || r.height < 12 || st.visibility === 'hidden' || st.display === 'none') continue;
    const t = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
    if (!rx.test(t) || /popup/i.test(el.getAttribute('href') || '')) continue;
    el.scrollIntoView({block: 'center'});
    return t;
  }
  return '';
}
"""


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_date(text):
    """'tomorrow', 'next friday', '12 March', '2026-03-12', '12/03/2026' -> datetime.date, or None."""
    s, today = text.strip().lower(), dt.date.today()
    if s in ("today", "tonight"):
        return today
    if s == "tomorrow":
        return today + dt.timedelta(days=1)
    if s == "day after tomorrow":
        return today + dt.timedelta(days=2)
    m = re.match(r"(?:next |this )?(" + "|".join(WEEKDAYS) + ")$", s)
    if m:
        ahead = (WEEKDAYS.index(m.group(1)) - today.weekday()) % 7 or 7
        return today + dt.timedelta(days=ahead)
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", s)
    if m:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d %B", "%d %b", "%B %d", "%b %d"):
        try:
            d = dt.datetime.strptime(re.sub(r"(?<=\d)(st|nd|rd|th)", "", s).replace(",", ""), fmt).date()
            if "%Y" not in fmt:
                d = d.replace(year=today.year)
                if d < today:
                    d = d.replace(year=today.year + 1)
            return d
        except ValueError:
            pass
    return None


def parse_time(text):
    """'7pm', '7:30 pm', '19:30' -> 'HH:MM', or None."""
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", text.strip().lower())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3) == "pm" and h < 12:
        h += 12
    if m.group(3) == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}" if h < 24 and mi < 60 else None



class Agent:
    def __init__(self, session, emit, model=DEFAULT_MODEL, headless=False, max_steps=25, engine=DEFAULT_ENGINE, location=None):
        self.location = location  # {"latitude", "longitude", "accuracy"} from the user's own browser
        self.s, self.emit, self.model, self.headless, self.max_steps = session, emit, model, headless, max_steps
        self.engine_key = engine if engine in ENGINES else DEFAULT_ENGINE
        self.engine_name, self.engine_home, self.engine_search = ENGINES[self.engine_key]
        self.last_query = None
        self.frames = {}    # frame index -> Frame, rebuilt by collect()
        self.seen = {}      # url -> label, every link shown to the model
        self.visited = []   # [(title, url)] pages the browser actually visited

    async def screenshot(self):
        try:
            b = await self.s.page.screenshot(type="jpeg", quality=60)
            await self.emit({"type": "screenshot", "data": base64.b64encode(b).decode(), "url": self.s.page.url})
        except Exception:
            pass

    async def ask(self, messages):
        r = await self.s.client.post(OLLAMA_URL, json={
            "model": self.model, "messages": messages, "stream": False, "think": False,
            "format": SCHEMA, "options": {"temperature": 0.2, "num_ctx": 12288},
        })
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    async def collect(self, page):
        """Scan the main page and every visible iframe (booking widgets often live in one). Iframe ids start at 100*index."""
        obs = await page.evaluate(COLLECT_JS, 0)
        self.frames = {}
        for i, fr in enumerate(page.frames):
            if fr == page.main_frame or len(obs["elements"]) >= 90:
                continue
            try:
                box = await (await fr.frame_element()).bounding_box()
                if not box or box["width"] < 40 or box["height"] < 20:
                    continue
                o = await fr.evaluate(COLLECT_JS, 100 * i)
            except Exception:
                continue
            if not o["elements"] and len(o["text"]) < 20:
                continue
            self.frames[i] = fr
            obs["elements"] += o["elements"]
            obs["text"] += f" [iframe {urlparse(fr.url).netloc}] {o['text'][:1500]}"
        return obs

    def locate(self, el_id):
        """Playwright locator for an element id from collect(); ids 100+ belong to iframes."""
        el_id = int(el_id)
        frame = self.frames.get(el_id // 100) if el_id >= 100 else None
        return (frame or self.s.page).locator(f'[data-agent-id="{el_id}"]').first

    async def click_element(self, el_id):
        """Click robustly. Returns None on success, or a short reason the model can act on."""
        page, loc = self.s.page, self.locate(el_id)
        if await loc.count() == 0:  # stale or invented number: fail fast with a reason instead of waiting for a timeout
            return f"no element numbered {el_id} on the current page; use a number from the Elements list"
        try:
            await loc.scroll_into_view_if_needed(timeout=3000)
            await loc.click(timeout=3000)
            return None
        except Exception as e:
            err = str(e).splitlines()[0][:80]
        try:  # what is sitting on top of it?
            blocker = await loc.evaluate(
                "e => { const r = e.getBoundingClientRect(); const t = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);"
                " if (!t || e.contains(t) || t.contains(e)) return ''; const c = typeof t.className === 'string' ? t.className.split(' ')[0] : '';"
                " return t.tagName.toLowerCase() + (t.id ? '#' + t.id : '') + (c ? '.' + c : ''); }", timeout=2000)
        except Exception:
            blocker = ""
        if blocker:
            try:
                dismissed = await page.evaluate(DISMISS_JS)
            except Exception:
                dismissed = ""
            if dismissed:
                await page.wait_for_timeout(600)
                try:
                    await loc.click(timeout=3000)
                    return None
                except Exception:
                    pass
        try:  # last resort: click through script (works when only an invisible layer is in the way)
            await loc.evaluate("e => e.click()", timeout=2000)
            return None
        except Exception:
            return f"covered by <{blocker}>, could not dismiss it" if blocker else err

    async def note_visit(self):
        try:
            url, title = self.s.page.url, await self.s.page.title()
        except Exception:
            return
        if url.startswith("http") and url not in [u for _, u in self.visited]:
            self.visited.append((title or url, url))

    def collect_links(self, answer_links):
        """Model-cited links (validated against what it really saw) + pages the browser visited."""
        out, urls = [], set()

        def add(title, url):
            if url not in urls:
                urls.add(url)
                out.append({"title": title, "url": url})

        engine_hosts = {urlparse(v[1]).netloc.replace("www.", "") for v in ENGINES.values()}
        is_engine = lambda u: urlparse(u).netloc.replace("www.", "") in engine_hosts
        base = lambda u: u.split("#")[0]
        titles = {base(u): t for t, u in self.visited}  # real page titles beat link labels
        for l in answer_links or []:
            for seen_url, label in self.seen.items():
                if seen_url == l or seen_url.startswith(l):
                    add(titles.get(base(seen_url)) or label or seen_url, base(seen_url))
                    break
            else:
                if base(l) in titles:
                    add(titles[base(l)], base(l))
        for t, u in self.visited:
            if not is_engine(u):
                add(t, base(u))
        return out[:10]

    async def run(self, task):
        s = self.s
        await s.ensure(self.headless)
        # Only touch the geolocation emulation when it changed: re-setting it while a Maps tab is open crashes the browser.
        if self.location and self.location != s.applied_loc:
            try:
                await s.ctx.set_geolocation(self.location)
            except Exception:  # a real Chrome may refuse emulation; it has its own location permission
                pass
            s.applied_loc = dict(self.location)
        s.stop_flag = False
        history = []
        s.blocked_engines.clear()  # every request starts with the user's chosen engine; a past (or false) block doesn't stick
        if s.page.url == "about:blank":
            try:
                await s.page.goto(self.engine_home, timeout=20000, wait_until="domcontentloaded")
            except Exception:
                pass
        convo = "\n".join(f"User: {u}\nYou answered: {a[:400]}" for u, a in s.chat[-4:]) or "(first message)"
        answer = None
        self.frames = {}
        fail_counts, hidden, consecutive_fails = {}, set(), 0  # stuck-loop guard
        refreshes = {}  # F5 count per URL
        asked_login = set()  # hosts where we already asked the user about a login wall
        details = []  # everything the user told us in answer to ask_user (the 4B model forgets it otherwise)
        bad_hosts = s.bad_hosts
        last_url, last_sig, last_click, scrolled_url = None, None, False, None
        last_site_url = None  # last normal (non sign-in) page, to return to after switching browsers
        is_booking = bool(re.search(r"\b(book|reserv\w*|table)\b", task, re.I)) and not DIRECTIONS_RE.search(task) and not PLACES_RE.search(task)
        for step in range(1, self.max_steps + 1):
            if s.stop_flag:
                await self.emit({"type": "status", "text": "Stopped."})
                answer = "Stopped by user."
                break
            page = s.page
            if s.mode == "test" and urlparse(page.url).netloc in ("accounts.google.com", "accounts.youtube.com"):
                # Google refuses to sign in to automated browsers: continue in the user's real Chrome instead.
                await self.emit({"type": "status", "text": "Google blocks automated browsers - switching to your own Chrome..."})
                s.mode = "chrome"
                try:
                    await s.ensure(False)
                    if last_site_url:
                        await s.page.goto(last_site_url, timeout=30000, wait_until="domcontentloaded")
                except Exception as e:
                    await self.emit({"type": "error", "text": f"Could not switch to your Chrome: {e}"})
                    return None
                await self.emit({"type": "browser_mode", "mode": "chrome"})
                note = ("switched the browser from the test browser to the user's real Chrome (Google blocks automated browsers); "
                        f"reopened {s.page.url}. Continue the sign-in flow there.")
                history.append(f"{step}. {note}")
                await self.emit({"type": "step", "n": step, "thought": "Google blocks the test browser.", "action": note})
                self.frames = {}
                continue
            if page.url.startswith("http") and not self.is_auth_host(page.url):
                last_site_url = page.url
            try:
                obs = await self.collect(page)
            except Exception:
                # Page is mid-navigation (e.g. after a click); retry the observation on the next loop.
                await asyncio.sleep(0.5)
                continue
            if not self.is_login_wall(page.url, obs) and await self.page_blocked(page, obs["text"]):  # a login wall is not a bot check
                host = urlparse(page.url).netloc.replace("www.", "")
                engine_key = next((k for k, v in ENGINES.items() if urlparse(v[1]).netloc.replace("www.", "") == host), None)
                if engine_key:
                    # A bot check on a search engine: stop using that engine and redo the search on the next one.
                    s.blocked_engines.add(engine_key)
                    await self.emit({"type": "status", "text": f"{ENGINES[engine_key][0]} showed a bot check - switching search engine"})
                    self.set_engine(next((k for k in ENGINES if k not in s.blocked_engines), engine_key))
                    desc = await self.search_with_fallback(self.last_query or task)
                else:
                    # A bot check on an ordinary site: leave it and let the model choose a different page.
                    blocked_url = page.url
                    bad_hosts[host] = "showed a bot check"
                    await self.emit({"type": "status", "text": f"{host} showed a bot check - going back"})
                    try:
                        await page.go_back(timeout=10000)
                    except Exception:
                        await self.search_with_fallback(self.last_query or task)
                    desc = f"back (page {host} showed a bot check; choose a different result, never solve it)"
                    history.append(f"{step}. {desc}")
                    await self.emit({"type": "step", "n": step, "thought": "Site showed a bot check.", "action": desc})
                    continue
                history.append(f"{step}. {desc}")
                await self.emit({"type": "step", "n": step, "thought": "Search engine showed a bot check.", "action": desc})
                continue
            if page.url != last_url:  # new page: close cookie/consent pop-ups by rejecting them (never accepting)
                last_url = page.url
                try:
                    if await page.evaluate(DISMISS_JS):
                        await page.wait_for_timeout(500)
                        obs = await self.collect(page)
                except Exception:
                    pass
            if is_booking and page.url != scrolled_url:
                # Booking request: the Reserve/Book button is often far down the page, and Elements lists only what is on screen.
                scrolled_url = page.url
                try:
                    found = await page.evaluate(FIND_RESERVE_JS)
                    if found:
                        await page.wait_for_timeout(400)
                        obs = await self.collect(page)
                        history.append(f"{step - 1 if step > 1 else 0}. scroll: scrolled down to the button '{found}'")
                except Exception:
                    pass
            sig = (page.url, obs["text"][:400])
            if last_click and sig == last_sig:
                history.append(f"{step - 1}. (that click changed nothing on the page - the button you want is probably further down: scroll down)")
            last_sig, last_click = sig, False
            await self.note_visit()
            await self.screenshot()
            lines = []
            for e in obs["elements"]:
                if f"{page.url}|{e['id']}" in hidden:  # failed repeatedly on this page: stop offering it
                    continue
                href = real_url(e["href"]) if e["href"] else ""
                if href and self.is_bad(href):  # a site we already gave up on: don't even offer it
                    continue
                if href:
                    self.seen[href] = e["label"]
                kind = {"date": " (date: type e.g. tomorrow or 2026-03-12)", "time": " (time: type e.g. 7:30pm)"}.get(e["type"], "")
                if e["tag"] == "select":
                    kind = " (dropdown: type the option you want)"
                lines.append(f'[{e["id"]}] <{e["tag"]}{" " + e["type"] if e["type"] else ""}> {e["label"]}{kind}'
                             + (f' -> {href[:110]}' if href else ""))
            els = "\n".join(lines) or "(none)"
            maps_hint = ""
            acts = [h.split(". ", 1)[1] for h in history]
            did = lambda prefix: any(a.startswith(prefix) for a in acts)
            wants_dir, wants_places = bool(DIRECTIONS_RE.search(task)), bool(PLACES_RE.search(task))
            if wants_places and not wants_dir:
                if not did("maps_search"):
                    maps_hint = ('HINT: this is a "find places near me" request. Your next action MUST be '
                                 '{"action": "maps_search", "text": "<what to find>, e.g. parks near me"}. No directions.\n\n')
                elif not did("FAILED maps_search"):
                    maps_hint = ("HINT: Google Maps is showing the list of places. Read the Page text and the Elements list and reply NOW "
                                 "with action done: a detailed LIST of the places (name, rating, type/address as shown), and say the user can "
                                 "ask for a route to any one of them. Put the place URLs from the Elements list in links.\n\n")
            elif wants_dir:
                if did("use_my_location: clicked"):
                    maps_hint = ("HINT: the route from the user's location is now shown on the page. Do NOT click anything else. "
                                 "Read the Page text (travel modes, durations, distances) and reply NOW with action done, "
                                 "giving a detailed answer with the options and times you see.\n\n")
                elif not did("maps:") and not did("FAILED maps"):
                    maps_hint = ('HINT: this is a directions request. Your next action MUST be {"action": "maps", "text": "<destination>"}. '
                                 'If the user gave no starting point, your action after that must be "use_my_location".\n\n')
                elif (did("maps:") and "google.com/maps" in page.url and not re.search(r"\bfrom\b", task, re.I)
                      and not did("use_my_location") and not did("FAILED use_my_location")):
                    maps_hint = 'HINT: no starting point was given, so your next action must be {"action": "use_my_location"}.\n\n'
            if re.search(r"(book|reserv\w*|table)", task, re.I) and not wants_dir and not wants_places:
                if any(re.search(r"reserv|book", e["label"], re.I) and not re.search("popup", e["label"]) for e in obs["elements"]):
                    maps_hint += 'HINT: a Reserve/Book element is on screen - click it (prefer the one with a real link).\n\n'
                elif obs["scroll"] < 100:
                    maps_hint += ("HINT: this is a booking request. Elements lists only what is on screen right now. No Reserve/Book button "
                                  "is visible yet, so scroll down to find it (do not click icon-only or popup buttons).\n\n")
            host_now = urlparse(page.url).netloc.replace("www.", "")
            if host_now not in asked_login and self.is_login_wall(page.url, obs):
                maps_hint += ("HINT: this page wants the user to log in or sign up. NEVER type credentials or create an account. "
                              'Your next action MUST be {"action": "ask_user", "text": "<question>"} asking the user, in the SAME language as the request, '
                              "to sign in themselves in the login window that opens, and to press Continue when done. Do NOT ask whether to click anything." + "\n\n")
            warn = maps_hint + ("WARNING: your last actions repeated without progress. Choose a DIFFERENT action.\n\n"
                    if len(history) >= 3 and len({h.split(". ", 1)[1] for h in history[-3:]}) == 1 else "")
            prompt = (f"Conversation so far:\n{convo}\n\nCURRENT USER REQUEST: {task}\n"
                      f"Preferred search engine: {self.engine_name}\n\nActions taken for this request:\n"
                      + ("\n".join(history[-8:]) or "(none yet)") +
                      f"\n\nURL: {page.url}\nTitle: {await page.title()}\nScrolled: {obs['scroll']}%\n"
                      + (f"DETAILS THE USER ALREADY GAVE YOU (use them, do not ask again): {' | '.join(details)}\n" if details else "")
                      + f"Today is {dt.date.today().strftime('%A %d %B %Y')}.\n"
                      + (f"SITES TO NEVER VISIT AGAIN (blocked or broken): {', '.join(bad_hosts)}\n" if bad_hosts else "") +
                      f"Page text: {obs['text']}\n\nElements:\n{els}\n\n" + warn + "Next action JSON:")
            await self.emit({"type": "status", "text": f"Step {step}: thinking..."})
            try:
                act = await self.ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            except Exception as e:
                await self.emit({"type": "error", "text": f"Model error: {e}. Is Ollama running with '{self.model}'?"})
                return None
            if act.get("action") == "ask_user":
                q = act.get("text") or act.get("thought") or "What would you like me to do next?"
                await self.emit({"type": "step", "n": step, "thought": act.get("thought", ""), "action": "ask_user"})
                await self.emit({"type": "question", "text": q, "login": self.is_login_wall(page.url, obs)})
                await self.emit({"type": "status", "text": "Waiting for your answer..."})
                reply = await s.wait_answer()
                if reply is None:
                    answer = "Stopped by user."
                    break
                history.append(f'{step}. ask_user "{q[:120]}" -> user replied: {reply}')
                details.append(reply)
                asked_login.add(urlparse(s.page.url).netloc.replace("www.", ""))
                continue
            if act.get("action") == "goto" and self.is_bad(act.get("url") or ""):
                desc = f"FAILED goto: {urlparse(act.get('url')).netloc} is blocked/broken - choose a different website"
                history.append(f"{step}. {desc}")
                await self.emit({"type": "step", "n": step, "thought": act.get("thought", ""), "action": desc})
                consecutive_fails += 1
                continue
            if act.get("action") == "press" and re.search(r"f5|reload|refresh", str(act.get("key", "")), re.I):
                refreshes[page.url] = refreshes.get(page.url, 0) + 1
                if refreshes[page.url] > 2:
                    # Refreshing twice didn't fix it: give up on this site and go back to the search results.
                    bad = urlparse(page.url).netloc.replace("www.", "")
                    bad_hosts[bad] = "kept showing an error after refreshing"
                    q = self.last_query or task
                    desc = await self.search_with_fallback(q)
                    desc += (f" (refreshing {bad} did not help, so I left it - pick a DIFFERENT website from these results, "
                             f"not {', '.join(sorted(bad_hosts))})")
                    history.append(f"{step}. {desc}")
                    await self.emit({"type": "step", "n": step, "thought": act.get("thought", ""), "action": desc})
                    consecutive_fails = 0
                    continue
            else:
                desc = None
            desc = await self.execute(act)
            history.append(f"{step}. {desc}")
            last_click = act.get("action") == "click" and not desc.startswith("FAILED")
            await self.emit({"type": "step", "n": step, "thought": act.get("thought", ""), "action": desc})
            if desc.startswith("FAILED") and act.get("action") in ("click", "type"):
                key = f"{page.url}|{act.get('id')}"
                fail_counts[key] = fail_counts.get(key, 0) + 1
                if fail_counts[key] >= 2:
                    hidden.add(key)
                    history.append(f"{step}. (element [{act.get('id')}] failed twice and was removed from the list; choose another way)")
            consecutive_fails = consecutive_fails + 1 if desc.startswith("FAILED") else 0
            if consecutive_fails >= 4:
                answer = (f"I got stuck: the last 4 actions failed (latest: {desc}). The page is still open at {s.page.url}, "
                          "so you can click what is blocking it in the browser window, or tell me what to try next and I'll continue.")
                break
            if act.get("action") == "done":
                answer = act.get("answer") or act.get("thought", "Done.")
                await self.screenshot()
                await self.emit({"type": "answer", "text": answer, "links": self.collect_links(act.get("links"))})
                s.chat.append((task, answer))
                return answer
        if answer is None:
            answer = f"I reached the {self.max_steps}-step limit before finishing. Ask me to continue and I'll pick up from this page."
        await self.emit({"type": "answer", "text": answer, "links": self.collect_links([])})
        s.chat.append((task, answer))
        return answer

    async def execute(self, a):
        page = self.s.page
        act = a.get("action")
        try:
            if act == "search":
                return await self.search_with_fallback(a.get("text", ""))
            if act == "maps_search":
                return await self.maps_search(a.get("text", ""))
            if act == "maps":
                return await self.open_maps(a.get("text", ""))
            if act == "use_my_location":
                return await self.use_my_location()
            if act == "goto":
                url = a.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                return f"goto {url}"
            if act == "click":
                problem = await self.click_element(a["id"])
                if problem:
                    return f"FAILED click {a['id']}: {problem}"
                return f"click [{a['id']}]"
            if act == "type":
                loc = self.locate(a["id"])
                value = a.get("text", "")
                info = await loc.evaluate(
                    "e => ({tag: e.tagName.toLowerCase(), type: e.type || '', ac: e.autocomplete || '', name: e.name || e.id || ''})", timeout=5000)
                if (info["type"] == "password" or re.search(r"username|password|webauthn", info["ac"], re.I)
                        or re.fullmatch(r"(email|e-?mail|user(name)?|login|identifier|session_key|session_password|pass|passwd|loginfmt)",
                                        info["name"], re.I) or (self.is_auth_host(page.url) and info["type"] in ("email", "text", "tel"))):
                    return (f"FAILED type {a['id']}: this is a sign-in field. NEVER type credentials or account details yourself - "
                            "the user signs in themselves. Use ask_user to tell them to sign in in the browser popup, then continue.")
                if info["tag"] == "select":
                    return await self.pick_option(loc, a["id"], value)
                if info["type"] in ("date", "time", "datetime-local", "month"):
                    value = self.format_for_input(info["type"], value)
                    if value is None:
                        return (f"FAILED type {a['id']}: this is a {info['type']} field; use a real value such as "
                                "tomorrow, 12 March or 2026-03-12 (dates) / 7:30pm (times)")
                await loc.fill(value, timeout=5000)
                if not a.get("key") and await loc.evaluate(
                        "e => e.type === 'search' || e.name === 'q' || ['searchbox','combobox'].includes(e.getAttribute('role'))"):
                    a["key"] = "Enter"  # small models forget to submit search boxes
                if a.get("key"):
                    await loc.press(a["key"])
                return f"type [{a['id']}] \"{value}\"" + (f" + {a['key']}" if a.get("key") else "")
            if act == "press":
                await page.keyboard.press(a.get("key", "Enter"))
                return f"press {a.get('key', 'Enter')}"
            if act == "scroll":
                d = 700 if a.get("direction", "down") == "down" else -700
                await page.mouse.wheel(0, d)
                return f"scroll {a.get('direction', 'down')}"
            if act == "back":
                await page.go_back(timeout=10000)
                return "back"
            if act == "wait":
                await asyncio.sleep(2)
                return "wait"
            if act == "done":
                return "done"
            return f"unknown action {act}"
        except Exception as e:
            return f"FAILED {act} {a.get('id', '')}: {str(e).splitlines()[0][:100]}"

    BLOCK_MARKERS = ("captcha", "unusual traffic", "not a bot", "not a robot", "verify you are human", "verify you're human",
                     "are you a robot", "automated queries", "quick check before you continue", "drag the slider",
                     "one last step", "solve the challenge", "challenge below", "bots use duckduckgo", "select all squares",
                     "i'm not a robot", "human verification", "checking your browser", "just a moment", "press & hold",
                     "verify that you are a human", "confirm you are human", "prove you are human", "access denied")
    CAPTCHA_DOM = ("iframe[src*=captcha i],iframe[src*=recaptcha i],iframe[src*=hcaptcha i],iframe[src*=turnstile i],"
                   "iframe[src*=challenge i],.g-recaptcha,.h-captcha,.cf-turnstile,#captcha,[id*=captcha i],[class*=captcha i]")

    def is_blocked(self, url, text, title="", dom=False):
        """True when a page is showing a bot check / block page instead of real content."""
        u, t = url.lower(), (text[:1500] + " " + title).lower()
        u = u.split("?")[0]  # a search query that mentions "captcha" is not a block page
        if dom or "/sorry/" in u or "captcha" in u or "/challenge" in u or "error/418" in u:
            return True
        # Real block pages are sparse; a full results/content page that merely mentions "captcha" is not a block.
        return len(text) < 700 and any(m in t for m in self.BLOCK_MARKERS)

    async def page_blocked(self, page, text=None):
        """Full check of the live page: URL, title, visible text and CAPTCHA widgets in the DOM."""
        try:
            if text is None:
                text = await page.evaluate("document.body ? document.body.innerText.slice(0, 1500) : ''")
            dom = await page.evaluate(  # only a VISIBLE captcha widget counts (login pages often embed invisible ones)
                "(sel) => [...document.querySelectorAll(sel)].some(e => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);"
                " return r.width > 40 && r.height > 40 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; })",
                self.CAPTCHA_DOM)
            return self.is_blocked(page.url, text, await page.title(), dom)
        except Exception:
            return False

    def format_for_input(self, kind, text):
        if kind == "time":
            return parse_time(text)
        d = parse_date(text)
        if d is None:
            return None
        return d.strftime("%Y-%m") if kind == "month" else d.isoformat() + ("T19:00" if kind == "datetime-local" else "")

    async def pick_option(self, loc, el_id, wanted):
        """Choose an option in a native <select> by visible text, value, or - for numbers - the range that contains it."""
        opts = await loc.evaluate("e => [...e.options].map((o, i) => ({i, text: o.text.trim(), value: o.value, off: o.disabled}))")
        opts = [o for o in opts if not o["off"] and o["text"]]
        w = wanted.strip().lower()
        pick = next((o for o in opts if o["text"].lower() == w or o["value"].lower() == w), None)
        if not pick and not w.isdigit():
            pick = next((o for o in opts if w and (w in o["text"].lower() or o["text"].lower() in w)), None)
        if not pick and re.fullmatch(r"\d+", w):
            n = int(w)
            for o in opts:  # "1 - 2 Guests", "3-4", "9+"
                m = re.search(r"(\d+)\s*(?:-|to|–)\s*(\d+)", o["text"]) or re.search(r"(\d+)\s*\+", o["text"])
                if m and ((len(m.groups()) == 2 and int(m.group(1)) <= n <= int(m.group(2))) or
                          (len(m.groups()) == 1 and n >= int(m.group(1)))):
                    pick = o
                    break
        if not pick:
            return f"FAILED select {el_id}: no option matches '{wanted}'. Options: " + "; ".join(o["text"] for o in opts)[:300]
        await loc.select_option(index=pick["i"], timeout=5000)
        return f"select [{el_id}] \"{pick['text']}\""

    LOGIN_PATH = re.compile(r"/(log-?in|sign-?in|signup|sign-up|accounts/login|authwall|uas/login|checkpoint)", re.I)
    LOGIN_TEXT = ("log in to", "sign in to", "log in or sign up", "join linkedin", "sign in or join", "log in to continue",
                  "create an account to", "sign in to see", "login to see", "join now to see", "log in to see", "log in to view",
                  "sign in to view", "join to view", "log in to your account", "this profile is private")

    AUTH_HOSTS = ("accounts.google.com", "login.microsoftonline.com", "login.live.com", "appleid.apple.com", "idmsa.apple.com",
                  "facebook.com/login", "m.facebook.com/login", "www.facebook.com/login", "accounts.youtube.com", "id.atlassian.com",
                  "github.com/login", "login.yahoo.com", "twitter.com/i/flow/login", "x.com/i/flow/login")

    def is_auth_host(self, url):
        u = urlparse(url)
        return any((u.netloc + u.path).lower().startswith(h) or u.netloc.lower().replace("www.", "") == h for h in self.AUTH_HOSTS)

    def is_login_wall(self, url, obs):
        """A page that blocks content behind login/sign-up (Instagram, LinkedIn, ...)."""
        head = obs["text"][:1200].lower()
        has_password = any(e["type"] == "password" for e in obs["elements"])
        return bool(has_password or self.is_auth_host(url) or self.LOGIN_PATH.search(urlparse(url).path)
                    or any(m in head for m in self.LOGIN_TEXT))

    def is_bad(self, url):
        host = urlparse(url).netloc.lower().replace("www.", "")
        return bool(host) and any(host == b or host.endswith("." + b) for b in self.s.bad_hosts)

    def set_engine(self, key):
        self.engine_key = key
        self.engine_name, self.engine_home, self.engine_search = ENGINES[key]

    async def search_with_fallback(self, q):
        """Search with the current engine; if it shows a CAPTCHA/block page, give up on that engine for this session and
        move to the next one (no retrying the blocked engine, no interacting with the check)."""
        s = self.s
        self.last_query = q
        order = [self.engine_key] + [k for k in ENGINES if k != self.engine_key]
        tried = []
        for key in order:
            if key in s.blocked_engines:
                continue
            self.set_engine(key)
            page = s.page
            try:
                await page.goto(self.engine_search.format(q=quote_plus(q)), timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(1000)  # block pages often render a moment after load
                blocked = await self.page_blocked(page)
            except Exception:
                blocked = False
            if not blocked:
                note = f" (switched from {', '.join(tried)} because of a bot check)" if tried else ""
                return f'search "{q}" on {self.engine_name}{note}'
            s.blocked_engines.add(key)
            tried.append(ENGINES[key][0])
            await self.emit({"type": "status", "text": f"{ENGINES[key][0]} showed a bot check - switching search engine"})
        return ("FAILED search: every search engine showed a bot check. Do NOT retry; in your done answer, tell the user "
                "to open a site directly or solve the check themselves in the browser window and say 'continue'.")

    async def maps_search(self, query):
        """Google Maps PLACES search (a list of places, e.g. 'parks near me'), reached via Bing when possible."""
        if not query.strip():
            return "FAILED maps_search: 'text' must be what to find, e.g. parks near me"
        page = await self.reach_maps()
        await page.goto("https://www.google.com/maps/search/" + quote(query), timeout=20000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)  # results list renders after load
        return f'maps_search: "{query}"'

    async def open_maps(self, dest):
        """Reach Google Maps (via Bing when possible), then open directions to `dest`."""
        if not dest.strip():
            return "FAILED maps: 'text' must be the destination, e.g. {\"action\": \"maps\", \"text\": \"Eiffel Tower, Paris\"}"
        page = await self.reach_maps()
        await page.goto("https://www.google.com/maps/dir//" + quote(dest), timeout=20000, wait_until="domcontentloaded")
        return f'maps: directions to "{dest}"'

    async def reach_maps(self):
        """Return a page that is on Google Maps; with Bing as the engine, get there by searching Bing for 'google maps'."""
        page = self.s.page
        if "google.com/maps" in page.url or page.is_closed():
            page = await self.fresh_page()  # a Maps tab is heavy (WebGL); a clean tab per Maps page avoids crashes
        if "google.com/maps" not in page.url:
            if self.engine_name == "Bing":
                await page.goto(self.engine_search.format(q="google maps"), timeout=20000, wait_until="domcontentloaded")
                for e in (await page.evaluate(COLLECT_JS))["elements"]:
                    if real_url(e["href"]).startswith("https://www.google.com/maps"):
                        await page.locator(f'[data-agent-id="{e["id"]}"]').first.click(timeout=5000)
                        await asyncio.sleep(1)  # the result may open in a new tab, which becomes self.s.page
                        page = self.s.page
                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                        break
        return page

    async def fresh_page(self):
        """Replace the current tab with a new one (frees a crashed or heavy Maps tab)."""
        old = self.s.page
        page = await self.s.ctx.new_page()
        self.s.page = page
        try:
            if old and not old.is_closed():
                await old.close()
        except Exception:
            pass
        return page

    async def use_my_location(self):
        """Run the starting-point flow; if the Maps tab crashes, reopen the same URL in a new tab and retry once."""
        url = self.s.page.url
        res = await self._use_my_location()
        if "crashed" in res and "google.com/maps" in url:
            page = await self.fresh_page()
            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            res = await self._use_my_location()
        return res

    async def _use_my_location(self):
        """Google Maps flow: click 'Choose starting point', then click the 'Use precise location' button twice.

        The dropdown button is drawn outside the normal DOM (no text/role locator can see it), so it is clicked by its
        position, which is fixed relative to the starting-point input: just below the field, near its left edge.
        """
        page = self.s.page
        if "google.com/maps" not in page.url:
            return "FAILED use_my_location: not on Google Maps (use the maps action first)"
        if not self.location:
            return ("FAILED use_my_location: the user has not shared their location. Do NOT retry; in your done answer, "
                    "tell the user to click the '📍 My location' button and ask again, or to give a starting point.")
        try:
            start = page.get_by_placeholder("Choose starting point").first
            await start.wait_for(state="visible", timeout=8000)
            await page.wait_for_timeout(2000)  # Maps re-renders right after load and would close the dropdown
            await start.click(timeout=6000)
            await page.wait_for_timeout(1500)  # dropdown with 'Use precise location' opens
            box = await start.bounding_box()
            x, y = box["x"] + 46, box["y"] + box["height"] / 2 + 123
            for _ in range(2):
                await page.mouse.click(x, y)
                await page.wait_for_timeout(1500)
            await page.wait_for_timeout(3000)  # let Maps compute and render the routes
            try:
                value = await page.locator("input").first.input_value(timeout=2000)
            except Exception:
                value = ""
            return f"use_my_location: clicked 'Use precise location' twice (starting point now: {value or 'locating...'})"
        except Exception as e:
            return f"FAILED use_my_location: {str(e).splitlines()[0][:100]}"
