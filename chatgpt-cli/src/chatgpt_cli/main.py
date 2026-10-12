"""
groq-chat-cli: a ChatGPT-style command-line assistant for a free-tier Groq key.

Chat, speech-to-text, text-to-speech and guard models are all usable. Models
are discovered from Groq's live list and only offered after a tiny real
request to them succeeds.

Chat commands:
    /chat new [title]           Start a new chat
    /chat list                  List chats, newest first
    /chat switch [num|title]    Resume a chat (prompts if omitted)
    /chat rename <title>        Rename the current chat (disables auto-title)
    /chat delete [num|title]    Delete a chat (prompts if omitted, asks to confirm)
    /chat clear                 Clear messages in the current chat

Messages:
    /message list               Show numbered turns with previews
    /message edit [num]         Edit a message (user edits truncate + regenerate)
    /message delete [num]       Delete a turn (user message + its reply)
    /regenerate                 Re-run the last reply, then pick the better one

Prompt & examples:
    /system                     Show the current system prompt
    /system set <text>          Set it for this chat
    /system default [text]      Show / set the default for new chats
    /system reset               Reset this chat to the default
    /examples                   Show scripted examples
    /examples add               Add a user -> assistant example pair
    /examples remove [num]      Remove an example

Models:
    /model                      Show the models used by this chat
    /model list                 List verified models (and unavailable ones)
    /model switch [num|name]    Switch model; fills the chat / stt / tts / guard slot
    /model refresh              Re-check the live model list now

Limits:
    /limit                      Tokens used by the last message
    /limit chat                 Requests and tokens used by this chat
    /limit remaining            Remaining requests / tokens (minute and day)

Settings:
    /params [name value|reset]  temperature, max_tokens, reasoning, voice

Audio and guard models:
    /transcribe [path]          Audio file -> text (speech-to-text model)
    /speak [text|last]          Text -> audio file (text-to-speech model)
    /classify [text]            Run text through a guard model

Other:
    /help                       Show the command list
    /exit, /quit                Quit (Ctrl+D also works)

Input rules: anything not starting with "/" is a message. Start a message
with "//" to send a literal leading slash. Alt+Enter inserts a new line.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import shlex
import sys
import tempfile
import time
import traceback
import wave
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, TypeVar

import groq
from dotenv import find_dotenv, load_dotenv
from groq import Groq
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

# ======================================================================
# Constants and paths
# ======================================================================

APP_DIR = Path(os.environ.get("GROQ_CHAT_HOME") or (Path.home() / ".groq_chat"))
CHATS_DIR = APP_DIR / "chats"
AUDIO_DIR = APP_DIR / "audio"
USAGE_FILE = APP_DIR / "usage.json"
MODELS_CACHE_FILE = APP_DIR / "models_cache.json"
CONFIG_FILE = APP_DIR / "config.json"
HISTORY_FILE = APP_DIR / "input_history"

DEFAULT_SYSTEM_CONTENT = "You are a helpful assistant."  # same default as ChatGPT
DEFAULT_TITLE = "New chat"
TITLE_MODEL = "openai/gpt-oss-20b"  # separate model => separate quota bucket
MODEL_CACHE_TTL = 24 * 3600  # seconds a model check stays valid
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # free-tier upload limit (verify in Groq docs)
AUDIO_EXTS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
DAY = 86400

# Groq's own recommended replacements for retired models (used in warnings).
DEPRECATED_REPLACEMENTS = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct": "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "openai/gpt-oss-120b",
    "qwen/qwen3-32b": "openai/gpt-oss-120b",
    "meta-llama/llama-guard-4-12b": "openai/gpt-oss-safeguard-20b",
}

# Voice names are from memory; if a voice is rejected use /params voice <name>.
DEFAULT_VOICES = {
    "canopylabs/orpheus-v1-english": "autumn",
    "canopylabs/orpheus-arabic-saudi": "fahad",
}

T = TypeVar("T")

# ======================================================================
# Console helpers
# ======================================================================

_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def info(msg: str = "") -> None:
    print(msg)


def dim(msg: str) -> None:
    print(_c("2", msg))


def warn(msg: str) -> None:
    print(_c("33", "Warning: ") + msg)


def err(msg: str) -> None:
    print(_c("31", "Error: ") + msg)


def human(n: int | float) -> str:
    """1000 -> '1K', 14400 -> '14.4K'."""
    if n >= 1000:
        return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "K"
    return str(int(n))


def preview(text: str, n: int = 60) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[: n - 1] + "…"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(iso)


# C0/C1 control characters except tab and newline. Dropping ESC (and the 8-bit
# CSI) is enough to stop text from driving the terminal.
_CONTROL = {c: None for c in [*range(0, 32), 127, *range(128, 160)] if c not in (9, 10)}


def strip_control(text: str) -> str:
    """Remove terminal control characters (e.g. ANSI escape sequences)."""
    return text.translate(_CONTROL)


def clean_title(text: str, limit: int = 80) -> str:
    """One printable line, no control characters, bounded length."""
    return " ".join(strip_control(text).split())[:limit]


def clean_input(text: str) -> str:
    """Replace anything that can't be encoded as UTF-8 (e.g. stray surrogates)."""
    return text.encode("utf-8", "replace").decode("utf-8")


def parse_index(text: str) -> int | None:
    """A plain ASCII number as an int, else None. Never raises, however odd the digits."""
    if text.isascii() and text.isdigit():
        return int(text) if len(text) <= 9 else 10**9  # huge = safely out of range
    return None


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON so a crash mid-write can never leave a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # e.g. a lone surrogate: fall back to \u escapes
        text = json.dumps(data, ensure_ascii=True, indent=2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ======================================================================
# Errors: one place that turns exceptions into readable messages
# ======================================================================


class CommandError(Exception):
    """A problem to show the user as a plain message (no traceback)."""


class Stopped(CommandError):
    """The user interrupted a running request."""


class QuitApp(Exception):
    """Raised by /exit and /quit."""


def api_message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return str(getattr(exc, "message", "") or exc)


def explain_error(exc: BaseException) -> str | None:
    """Friendly text for errors we expect, or None for anything else."""
    if isinstance(exc, groq.AuthenticationError):
        return "Groq rejected your API key. Check GROQ_API_KEY in your .env file."
    if isinstance(exc, groq.PermissionDeniedError):
        return (
            f"Access denied: {api_message(exc)}\n"
            "If this model needs its terms accepted, do that in the Groq console."
        )
    if isinstance(exc, groq.NotFoundError):
        return (
            f"Model or endpoint not found: {api_message(exc)}\n"
            "Use /model list to see what is available, then /model switch."
        )
    if isinstance(exc, groq.RateLimitError):
        retry = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            retry = resp.headers.get("retry-after")
        msg = f"Rate limit reached: {api_message(exc)}"
        if retry:
            msg += f"\nTry again in about {retry} seconds."
        return msg + "\nSee /limit remaining for your usage."
    if isinstance(exc, groq.BadRequestError):
        msg = api_message(exc)
        low = msg.lower()
        if "decommission" in low or "does not exist" in low or "not found" in low:
            return f"Model unavailable: {msg}\nUse /model list, then /model switch."
        if "terms" in low:
            return f"This model needs its terms accepted in the Groq console first: {msg}"
        return f"Groq rejected the request: {msg}"
    if isinstance(exc, groq.APITimeoutError):
        return "The request to Groq timed out. Try again."
    if isinstance(exc, groq.APIConnectionError):
        return "Couldn't reach Groq. Check your internet connection."
    if isinstance(exc, groq.APIStatusError):
        return f"Groq returned an error ({exc.status_code}): {api_message(exc)}"
    if isinstance(exc, groq.APIError):
        return f"Groq API error: {api_message(exc)}"
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {exc.filename or exc}"
    if isinstance(exc, PermissionError):
        return f"Permission denied: {exc.filename or exc}"
    if isinstance(exc, OSError):
        return f"File error: {exc}"
    return None


# ======================================================================
# Token estimation (Groq returns real usage; this is only for pre-flight)
# ======================================================================

_encoder: Any = None


def estimate_tokens(text: str) -> int:
    """Rough count. Uses tiktoken if installed and usable, else chars / 4."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("o200k_base")
        except Exception:  # not installed, or vocab can't be downloaded offline
            _encoder = False
    if _encoder:
        try:
            return len(_encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


# ======================================================================
# Models: kinds, limits table, verified registry
# ======================================================================


class Kind(StrEnum):
    CHAT = "chat"
    STT = "stt"
    TTS = "tts"
    GUARD = "guard"


KIND_ORDER = [Kind.CHAT, Kind.STT, Kind.TTS, Kind.GUARD]
SLOT_LABEL = {
    Kind.CHAT: "chat",
    Kind.STT: "speech-to-text",
    Kind.TTS: "text-to-speech",
    Kind.GUARD: "guard",
}
PREFERRED_DEFAULTS = {
    Kind.CHAT: "openai/gpt-oss-120b",
    Kind.STT: "whisper-large-v3-turbo",
    Kind.TTS: "canopylabs/orpheus-v1-english",
    Kind.GUARD: "meta-llama/llama-prompt-guard-2-86m",
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    kind: Kind
    rpm: int | None = None  # requests / minute
    rpd: int | None = None  # requests / day
    tpm: int | None = None  # tokens / minute
    tpd: int | None = None  # tokens / day
    ash: int | None = None  # audio seconds / hour
    asd: int | None = None  # audio seconds / day
    preview: bool = False
    context_window: int | None = None  # filled from the live API


# Free-tier limits as supplied by the user. Limits change: edit this table.
KNOWN_SPECS = [
    ModelSpec("canopylabs/orpheus-arabic-saudi", Kind.TTS, 10, 100, 1200, 3600),
    ModelSpec("canopylabs/orpheus-v1-english", Kind.TTS, 10, 100, 1200, 3600),
    ModelSpec("meta-llama/llama-prompt-guard-2-22m", Kind.GUARD, 30, 14400, 15000, 500000),
    ModelSpec("meta-llama/llama-prompt-guard-2-86m", Kind.GUARD, 30, 14400, 15000, 500000),
    ModelSpec("openai/gpt-oss-120b", Kind.CHAT, 30, 1000, 8000, 200000),
    ModelSpec("openai/gpt-oss-20b", Kind.CHAT, 30, 1000, 8000, 200000),
    ModelSpec("openai/gpt-oss-safeguard-20b", Kind.GUARD, 3, 1000, 2000, 200000),
    # Qwen models were in Groq's Preview tier at the time of writing.
    ModelSpec("qwen/qwen3.8-27b", Kind.CHAT, 30, 1000, 8000, 200000, preview=True),
    ModelSpec("whisper-large-v3", Kind.STT, 20, 2000, ash=7200, asd=28800),
    ModelSpec("whisper-large-v3-turbo", Kind.STT, 20, 2000, ash=7200, asd=28800),
]
KNOWN: dict[str, ModelSpec] = {s.id: s for s in KNOWN_SPECS}


def guess_kind(model_id: str) -> Kind:
    low = model_id.lower()
    if "whisper" in low:
        return Kind.STT
    if "orpheus" in low or "tts" in low:
        return Kind.TTS
    if "guard" in low:  # prompt-guard, llama-guard, gpt-oss-safeguard
        return Kind.GUARD
    return Kind.CHAT


def fmt_limits(spec: ModelSpec) -> str:
    parts = []
    for label, val in (
        ("RPM", spec.rpm),
        ("RPD", spec.rpd),
        ("TPM", spec.tpm),
        ("TPD", spec.tpd),
        ("audio-s/h", spec.ash),
        ("audio-s/day", spec.asd),
    ):
        if val is not None:
            parts.append(f"{human(val)} {label}")
    return " · ".join(parts) or "limits unknown"


def silent_wav(seconds: float = 0.2, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def probe_model(client: Groq, spec: ModelSpec) -> tuple[str, str]:
    """Send one tiny real request. Returns (status, note).

    verified      the request worked
    unchecked     inconclusive (rate limited etc.): retry later, not offered yet
    needs_action  exists but needs something from you (e.g. accept terms)
    unavailable   not usable (retired, wrong type, ...)
    Auth and connection errors propagate to the caller.
    """
    try:
        if spec.kind in (Kind.CHAT, Kind.GUARD):
            client.chat.completions.create(
                model=spec.id,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=16,
            )
        elif spec.kind is Kind.STT:
            client.audio.transcriptions.create(file=("probe.wav", silent_wav()), model=spec.id)
        elif spec.kind is Kind.TTS:
            client.audio.speech.create(
                model=spec.id,
                voice=DEFAULT_VOICES.get(spec.id, "autumn"),
                input="hi",
                response_format="wav",
            )
        return "verified", ""
    except (groq.AuthenticationError, groq.APIConnectionError):
        raise
    except groq.RateLimitError:
        return "unchecked", "rate limited during the check; try /model refresh later"
    except groq.PermissionDeniedError as exc:
        return "needs_action", api_message(exc)
    except groq.NotFoundError as exc:
        return "unavailable", api_message(exc)
    except groq.BadRequestError as exc:
        msg = api_message(exc)
        low = msg.lower()
        if "terms" in low:
            return "needs_action", msg
        if spec.kind is Kind.TTS and "voice" in low:
            return "unchecked", msg
        return "unavailable", msg
    except groq.APIStatusError as exc:
        return "unchecked", f"HTTP {exc.status_code}: {api_message(exc)}"


@dataclass
class ModelEntry:
    spec: ModelSpec
    status: str  # verified | unverified | unchecked | needs_action | unavailable
    note: str = ""
    checked_at: float = 0.0

    @property
    def selectable(self) -> bool:
        return self.status in ("verified", "unverified")


class ModelRegistry:
    """Models that exist on Groq right now, with a verified/unavailable status."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, ModelEntry] = {}
        self.checked_at = 0.0
        self.from_cache = False  # True once entries come from a real live check

    # -- persistence --------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            models = data["models"]
            entries: dict[str, ModelEntry] = {}
            for mid, d in models.items():
                try:
                    base = KNOWN.get(mid) or ModelSpec(mid, Kind(d["kind"]))
                    spec = replace(base, context_window=d.get("context_window"))
                    entries[mid] = ModelEntry(
                        spec, d["status"], d.get("note", ""), float(d.get("checked_at", 0))
                    )
                except (KeyError, ValueError, TypeError):
                    continue
            if not entries:
                raise ValueError("empty cache")
            self.entries = entries
            self.checked_at = float(data.get("checked_at", 0))
            self.from_cache = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self._load_builtin()

    def _load_builtin(self) -> None:
        self.entries = {s.id: ModelEntry(s, "unverified") for s in KNOWN_SPECS}
        self.checked_at = 0.0
        self.from_cache = False

    def save(self) -> None:
        data = {
            "checked_at": self.checked_at,
            "models": {
                mid: {
                    "kind": e.spec.kind.value,
                    "status": e.status,
                    "note": e.note,
                    "context_window": e.spec.context_window,
                    "checked_at": e.checked_at,
                }
                for mid, e in self.entries.items()
            },
        }
        try:
            write_json_atomic(self.path, data)
        except OSError:
            pass

    def is_stale(self) -> bool:
        return (not self.from_cache) or time.time() - self.checked_at > MODEL_CACHE_TTL

    # -- queries ------------------------------------------------------
    def spec_for(self, model_id: str) -> ModelSpec:
        entry = self.entries.get(model_id)
        if entry:
            return entry.spec
        return KNOWN.get(model_id) or ModelSpec(model_id, guess_kind(model_id))

    def selectable(self, kind: Kind | None = None) -> list[ModelEntry]:
        items = [
            e
            for e in self.entries.values()
            if e.selectable and (kind is None or e.spec.kind is kind)
        ]
        return sorted(items, key=lambda e: (KIND_ORDER.index(e.spec.kind), e.spec.id))

    def not_selectable(self) -> list[ModelEntry]:
        items = [e for e in self.entries.values() if not e.selectable]
        return sorted(items, key=lambda e: (KIND_ORDER.index(e.spec.kind), e.spec.id))

    def default_for(self, kind: Kind) -> str | None:
        options = self.selectable(kind)
        preferred = PREFERRED_DEFAULTS[kind]
        if any(e.spec.id == preferred for e in options):
            return preferred
        return options[0].spec.id if options else None

    # -- live refresh -------------------------------------------------
    def refresh(
        self,
        client: Groq,
        tracker: "UsageTracker",
        *,
        force: bool = False,
        say: Callable[[str], None] = print,
    ) -> None:
        """Fetch the live list and probe every model that needs checking.

        Raises groq.AuthenticationError / groq.APIConnectionError unchanged;
        in that case the existing entries are left as they were.
        """
        live: dict[str, int | None] = {}
        for m in client.models.list().data:
            if getattr(m, "active", True) is False:
                continue
            live[m.id] = getattr(m, "context_window", None)

        now = time.time()
        fresh: dict[str, ModelEntry] = {}
        for i, (mid, ctx) in enumerate(sorted(live.items()), 1):
            base = KNOWN.get(mid) or ModelSpec(mid, guess_kind(mid))
            spec = replace(base, context_window=ctx)
            old = self.entries.get(mid)
            reuse = (
                old is not None
                and not force
                and self.from_cache
                and old.status in ("verified", "needs_action", "unavailable")
                and now - old.checked_at < MODEL_CACHE_TTL
            )
            if reuse:
                fresh[mid] = replace(old, spec=spec)
                continue
            say(f"  checking {mid} ({spec.kind.value}) [{i}/{len(live)}]")
            status, note = probe_model(client, spec)
            tracker.record(mid, tokens=8, audio_seconds=0.2 if spec.kind is Kind.STT else 0.0)
            fresh[mid] = ModelEntry(spec, status, note, now)

        self.entries = fresh
        self.checked_at = now
        self.from_cache = True
        self.save()


# ======================================================================
# Usage tracking (global per model, because quotas are per model)
# ======================================================================

LIMIT_LABELS = {
    "rpm": "requests / minute",
    "rpd": "requests / day",
    "tpm": "tokens / minute",
    "tpd": "tokens / day",
    "ash": "audio seconds / hour",
    "asd": "audio seconds / day",
}
HEADER_LABELS = {
    "x-ratelimit-remaining-requests": "remaining requests (server)",
    "x-ratelimit-remaining-tokens": "remaining tokens (server)",
    "x-ratelimit-reset-requests": "requests reset in",
    "x-ratelimit-reset-tokens": "tokens reset in",
}


class UsageTracker:
    """Rolling 24h log of requests, tokens and audio seconds per model."""

    def __init__(self, path: Path):
        self.path = path
        self.events: dict[str, list[list[float]]] = {}  # model -> [[ts, tokens, audio_s]]
        self.headers: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        events = data.get("events")
        if isinstance(events, dict):
            for model, evs in events.items():
                if not isinstance(evs, list):
                    continue
                clean = []
                for e in evs:
                    try:
                        clean.append([float(e[0]), float(e[1]), float(e[2])])
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
                self.events[str(model)] = clean
        headers = data.get("headers")
        if isinstance(headers, dict):
            for model, snap in headers.items():
                if (
                    isinstance(snap, dict)
                    and isinstance(snap.get("ts"), (int, float))
                    and isinstance(snap.get("values"), dict)
                ):
                    self.headers[str(model)] = snap
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - DAY
        for model in list(self.events):
            self.events[model] = [e for e in self.events[model] if e[0] >= cutoff]
            if not self.events[model]:
                del self.events[model]

    def _save(self) -> None:
        try:
            write_json_atomic(self.path, {"events": self.events, "headers": self.headers})
        except OSError:
            pass

    def record(self, model: str, tokens: int = 0, audio_seconds: float = 0.0) -> None:
        self.events.setdefault(model, []).append([time.time(), float(tokens), float(audio_seconds)])
        self._prune()
        self._save()

    def note_headers(self, model: str, headers: dict[str, str]) -> None:
        vals = {k.lower(): v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}
        if vals:
            self.headers[model] = {"ts": time.time(), "values": vals}
            self._save()

    def used(self, model: str, seconds: int) -> tuple[int, int, float]:
        cutoff = time.time() - seconds
        reqs = tokens = 0
        audio = 0.0
        for ts, tok, aud in self.events.get(model, []):
            if ts >= cutoff:
                reqs += 1
                tokens += int(tok)
                audio += aud
        return reqs, tokens, audio

    def remaining(self, spec: ModelSpec) -> dict[str, tuple[int, int, int]]:
        """key -> (limit, used, left) for every limit this model has."""
        r60, t60, _ = self.used(spec.id, 60)
        rday, tday, aday = self.used(spec.id, DAY)
        _, _, ahour = self.used(spec.id, 3600)
        rows = {}
        for key, limit, used in (
            ("rpm", spec.rpm, r60),
            ("rpd", spec.rpd, rday),
            ("tpm", spec.tpm, t60),
            ("tpd", spec.tpd, tday),
            ("ash", spec.ash, int(ahour)),
            ("asd", spec.asd, int(aday)),
        ):
            if limit is not None:
                rows[key] = (limit, used, max(limit - used, 0))
        return rows

    def active_models(self) -> list[str]:
        return list(self.events)


# ======================================================================
# Chat data model and storage
# ======================================================================


def _known_kwargs(cls: type, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class Candidate:
    text: str
    model: str = ""
    completion_tokens: int = 0


@dataclass
class Turn:
    user: str
    candidates: list[Candidate] = field(default_factory=list)
    selected: int = 0

    @property
    def reply(self) -> Candidate | None:
        if not self.candidates:
            return None
        return self.candidates[min(self.selected, len(self.candidates) - 1)]


@dataclass
class ChatUsage:  # display-only totals; the real quota lives in UsageTracker
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_prompt: int = 0
    last_completion: int = 0
    last_estimated: bool = False


@dataclass
class Params:
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    voice: str | None = None


@dataclass
class ChatSession:
    id: str = field(default_factory=lambda: os.urandom(4).hex())
    title: str = DEFAULT_TITLE
    title_is_auto: bool = True
    models: dict[str, str] = field(default_factory=dict)  # kind value -> model id
    system_prompt: str = DEFAULT_SYSTEM_CONTENT
    examples: list[list[str]] = field(default_factory=list)  # [[user, assistant], ...]
    turns: list[Turn] = field(default_factory=list)
    usage: ChatUsage = field(default_factory=ChatUsage)
    params: Params = field(default_factory=Params)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    schema_version: int = 1

    def api_messages(
        self, *, drop_last_reply: bool = False, budget: int | None = None
    ) -> tuple[list[dict], int]:
        """Messages to send to the API plus how many old turns were left out."""
        head: list[dict] = []
        if self.system_prompt:
            head.append({"role": "system", "content": self.system_prompt})
        for user, assistant in self.examples:
            head.append({"role": "user", "content": user})
            head.append({"role": "assistant", "content": assistant})

        groups: list[list[dict]] = []
        for i, t in enumerate(self.turns):
            group = [{"role": "user", "content": t.user}]
            is_last = i == len(self.turns) - 1
            if t.reply and not (is_last and drop_last_reply):
                group.append({"role": "assistant", "content": t.reply.text})
            groups.append(group)

        trimmed = 0
        if budget is not None and groups:
            used = estimate_messages_tokens(head)
            keep: list[list[dict]] = []
            for group in reversed(groups):
                cost = estimate_messages_tokens(group)
                if keep and used + cost > budget:
                    break
                keep.append(group)
                used += cost
            keep.reverse()
            trimmed = len(groups) - len(keep)
            groups = keep
        return head + [m for g in groups for m in g], trimmed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChatSession":
        turns = []
        for t in d.get("turns", []):
            cands = [
                Candidate(
                    text=str(c.get("text", "")),
                    model=str(c.get("model", "")),
                    completion_tokens=int(c.get("completion_tokens", 0)),
                )
                for c in t.get("candidates", [])
            ]
            turns.append(
                Turn(user=str(t.get("user", "")), candidates=cands, selected=int(t.get("selected", 0)))
            )
        base = _known_kwargs(
            cls,
            {
                k: v
                for k, v in d.items()
                if k not in ("turns", "usage", "params", "models", "examples")
            },
        )
        obj = cls(
            **base,
            turns=turns,
            usage=ChatUsage(**_known_kwargs(ChatUsage, d.get("usage") or {})),
            params=Params(**_known_kwargs(Params, d.get("params") or {})),
            models={str(k): str(v) for k, v in (d.get("models") or {}).items()},
            examples=[[str(a), str(b)] for a, b in (d.get("examples") or [])],
        )
        obj.id = str(obj.id)
        obj.title = clean_title(str(obj.title)) or DEFAULT_TITLE
        obj.created_at, obj.updated_at = str(obj.created_at), str(obj.updated_at)
        return obj


class ChatStore:
    """One JSON file per chat, named by id (titles change; ids never do)."""

    def __init__(self, directory: Path, warn_fn: Callable[[str], None]):
        self.dir = directory
        self.warn = warn_fn

    def path(self, chat_id: str) -> Path:
        return self.dir / f"{chat_id}.json"

    def exists(self, chat_id: str) -> bool:
        return self.path(chat_id).exists()

    def save(self, session: ChatSession) -> None:
        session.updated_at = now_iso()
        write_json_atomic(self.path(session.id), session.to_dict())

    def load_file(self, p: Path) -> ChatSession:
        session = ChatSession.from_dict(json.loads(p.read_text(encoding="utf-8")))
        session.id = p.stem  # the filename is the id: a tampered file can't aim elsewhere
        return session

    def list(self) -> list[ChatSession]:
        chats = []
        for p in self.dir.glob("*.json"):
            try:
                chats.append(self.load_file(p))
            except (OSError, ValueError, TypeError, AttributeError):
                self.warn(f"Couldn't read {p.name}; moved aside as .bad")
                try:
                    p.rename(p.with_suffix(".bad"))
                except OSError:
                    pass
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        return chats

    def find_prefix(self, prefix: str) -> ChatSession | None:
        for chat in self.list():
            if chat.id.startswith(prefix):
                return chat
        return None

    def delete(self, chat_id: str) -> None:
        self.path(chat_id).unlink(missing_ok=True)


@dataclass
class Config:
    default_system: str = DEFAULT_SYSTEM_CONTENT

    @classmethod
    def load(cls, path: Path) -> "Config":
        try:
            cfg = cls(**_known_kwargs(cls, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()
        if not isinstance(cfg.default_system, str) or not cfg.default_system.strip():
            cfg.default_system = DEFAULT_SYSTEM_CONTENT
        return cfg

    def save(self, path: Path) -> None:
        write_json_atomic(path, asdict(self))


# ======================================================================
# Command registry, input parsing, completion
# ======================================================================


@dataclass
class Command:
    path: tuple[str, ...]
    handler: Callable[["App", str], None]
    usage: str
    summary: str
    group: str


COMMANDS: dict[tuple[str, ...], Command] = {}


def command(*path: str, usage: str = "", summary: str = "", group: str = "Other"):
    def register(fn: Callable[["App", str], None]):
        COMMANDS[path] = Command(path, fn, usage, summary, group)
        return fn

    return register


@dataclass
class Parsed:
    kind: str  # "empty" | "message" | "command"
    text: str = ""  # the message, or the raw argument string of a command
    path: tuple[str, ...] = ()


def parse(line: str) -> Parsed:
    """Decide whether a line is a message or a command."""
    line = clean_input(line).strip()
    if not line:
        return Parsed("empty")
    if line.startswith("//"):  # escape: "//etc is a folder" -> message "/etc is a folder"
        return Parsed("message", line[1:])
    if not line.startswith("/"):
        return Parsed("message", line)
    words = line[1:].split()
    for n in range(min(len(words), 3), 0, -1):  # longest match wins
        key = tuple(w.lower() for w in words[:n])
        if key in COMMANDS:
            rest = line[1:].split(None, n)[n] if len(words) > n else ""
            return Parsed("command", rest, key)
    return Parsed("command", "", tuple(w.lower() for w in words[:1]))


def report_unknown(line: str) -> None:
    words = line.strip()[1:].split()
    if not words:
        raise CommandError("Type /help to see the available commands.")
    top = words[0].lower()
    subs = sorted(p[1] for p in COMMANDS if p[0] == top and len(p) > 1)
    if subs:
        listing = ", ".join(f"/{top} {s}" for s in subs)
        if len(words) > 1:
            raise CommandError(f"Unknown subcommand '{words[1]}' for /{top}. Try: {listing}")
        raise CommandError(f"/{top} needs a subcommand. Try: {listing}")
    close = difflib.get_close_matches(top, sorted({p[0] for p in COMMANDS}), n=1)
    hint = f" Did you mean /{close[0]}?" if close else ""
    raise CommandError(
        f"Unknown command '/{top}'.{hint} Type /help for the list. "
        "(To send a message that starts with '/', begin it with '//'.)"
    )


def no_args(arg: str, name: str) -> None:
    """Bare commands such as /model must not silently ignore extra words."""
    if arg.strip():
        subs = sorted(p[1] for p in COMMANDS if p[0] == name and len(p) > 1)
        extra = f" Try: {', '.join(subs)}" if subs else ""
        raise CommandError(f"Unknown subcommand '{arg.split()[0]}' for /{name}.{extra}")


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or text.startswith("//"):
            return
        words = text[1:].split(" ")
        done = [w.lower() for w in words[:-1]]
        current = words[-1].lower()
        options = {
            p[len(done)] for p in COMMANDS if len(p) > len(done) and list(p[: len(done)]) == done
        }
        for opt in sorted(options):
            if opt.startswith(current):
                yield Completion(opt, start_position=-len(current))


# ======================================================================
# Application state
# ======================================================================


class App:
    def __init__(
        self,
        client: Groq,
        config: Config,
        registry: ModelRegistry,
        tracker: UsageTracker,
        store: ChatStore,
        debug: bool = False,
    ):
        self.client = client
        self.config = config
        self.registry = registry
        self.tracker = tracker
        self.store = store
        self.debug = debug
        self.prefill = ""  # puts an unsent message back in the prompt
        self.session = self.new_session()

        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        self._prompt = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=SlashCompleter(),
            key_bindings=kb,
            complete_while_typing=True,
        )
        self._ask = PromptSession()

    # -- sessions -----------------------------------------------------
    def new_session(self, title: str | None = None) -> ChatSession:
        models = {
            kind.value: self.registry.default_for(kind) or PREFERRED_DEFAULTS[kind]
            for kind in KIND_ORDER
        }
        return ChatSession(
            title=title or DEFAULT_TITLE,
            title_is_auto=title is None,
            models=models,
            system_prompt=self.config.default_system,
        )

    def save(self) -> None:
        s = self.session
        if s.turns or self.store.exists(s.id):
            try:
                self.store.save(s)
            except OSError as exc:
                warn(f"Couldn't save this chat: {exc}")

    def check_models(self, session: ChatSession) -> bool:
        """Warn about, and replace, models a loaded chat uses that are gone."""
        changed = False
        reg = self.registry
        for kind in KIND_ORDER:
            mid = session.models.get(kind.value)
            if mid is None:
                session.models[kind.value] = reg.default_for(kind) or PREFERRED_DEFAULTS[kind]
                changed = True
                continue
            if not reg.from_cache:  # nothing live-verified, so we can't judge
                continue
            entry = reg.entries.get(mid)
            if entry is not None and (entry.selectable or entry.status == "unchecked"):
                continue
            new = reg.default_for(kind)
            if new is None:
                warn(f"The {SLOT_LABEL[kind]} model '{mid}' is unavailable and no replacement was found.")
                continue
            why = "is no longer available" if entry is None else f"isn't usable ({entry.note or entry.status})"
            hint = DEPRECATED_REPLACEMENTS.get(mid)
            extra = f" Groq recommends {hint}." if hint else ""
            warn(f"The {SLOT_LABEL[kind]} model '{mid}' {why}. Switched this chat to '{new}'.{extra}")
            session.models[kind.value] = new
            changed = True
        return changed

    # -- prompts ------------------------------------------------------
    def read_line(self) -> str:
        default, self.prefill = self.prefill, ""
        return clean_input(self._prompt.prompt("\n> ", default=default))

    def ask(self, text: str, default: str = "") -> str:
        try:
            return clean_input(self._ask.prompt(text, default=default))
        except EOFError:
            raise KeyboardInterrupt from None

    def confirm(self, question: str) -> bool:
        return self.ask(f"{question} [y/N] ").strip().lower() in ("y", "yes")


# ======================================================================
# API layer
# ======================================================================


def call_with_headers(endpoint: Any, **kwargs: Any) -> tuple[Any, dict[str, str]]:
    """Call endpoint.create(), also returning response headers when the SDK allows."""
    raw_api = getattr(endpoint, "with_raw_response", None)
    if raw_api is not None:
        raw = raw_api.create(**kwargs)
        return raw.parse(), {k.lower(): v for k, v in raw.headers.items()}
    return endpoint.create(**kwargs), {}


def chat_kwargs(params: Params, temperature: float | None = None) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    temp = temperature if temperature is not None else params.temperature
    if temp is not None:
        kw["temperature"] = temp
    if params.max_tokens:
        kw["max_completion_tokens"] = params.max_tokens
    if params.reasoning_effort:
        kw["reasoning_effort"] = params.reasoning_effort
    return kw


def context_budget(spec: ModelSpec, params: Params) -> int | None:
    """How many input tokens to send, leaving room for the reply (TPM-limited models)."""
    if not spec.tpm:
        return None
    reserve = (params.max_tokens or 2000) + 300
    return max(spec.tpm - reserve, 1000)


def run_chat(app: App, messages: list[dict], *, temperature: float | None = None) -> Candidate:
    """Stream one reply to the terminal and return it."""
    s = app.session
    model = s.models["chat"]
    spec = app.registry.spec_for(model)

    est = estimate_messages_tokens(messages)
    if spec.tpm and est > spec.tpm:
        raise CommandError(
            f"This conversation is about {est:,} tokens, over {model}'s limit of "
            f"{spec.tpm:,} tokens per minute. Try /chat clear, /message delete, "
            "or a model with a higher limit."
        )
    left = app.tracker.remaining(spec).get("tpm")
    if left and est > left[2]:
        warn("You've used a lot of this minute's tokens; the request may hit the limit.")

    pieces: list[str] = []
    usage = None
    stream = None
    try:
        stream, headers = call_with_headers(
            app.client.chat.completions,
            model=model,
            messages=messages,
            stream=True,
            **chat_kwargs(s.params, temperature),
        )
        app.tracker.note_headers(model, headers)
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if choices:
                piece = strip_control(getattr(choices[0].delta, "content", None) or "")
                if piece:
                    pieces.append(piece)
                    print(piece, end="", flush=True)
            u = getattr(getattr(chunk, "x_groq", None), "usage", None) or getattr(chunk, "usage", None)
            if u:
                usage = u
    except KeyboardInterrupt:
        print()
        raise Stopped("Stopped.") from None
    finally:
        close = getattr(stream, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass
    print()

    text = "".join(pieces)
    if not text.strip():
        raise CommandError(
            "The model returned an empty reply (reasoning models can use their whole "
            "output budget thinking). Try /params reasoning low or /params max_tokens 2000."
        )

    estimated = usage is None
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    else:
        prompt_tokens, completion_tokens = est, estimate_tokens(text)

    app.tracker.record(model, tokens=prompt_tokens + completion_tokens)
    u = s.usage
    u.requests += 1
    u.prompt_tokens += prompt_tokens
    u.completion_tokens += completion_tokens
    u.last_prompt, u.last_completion, u.last_estimated = prompt_tokens, completion_tokens, estimated
    mark = "~" if estimated else ""
    dim(f"[{model} · {mark}{prompt_tokens:,} in / {mark}{completion_tokens:,} out tokens]")
    return Candidate(text=text, model=model, completion_tokens=completion_tokens)


def send_message(app: App, text: str) -> None:
    s = app.session
    turn = Turn(user=text)
    s.turns.append(turn)
    ok = False
    try:
        budget = context_budget(app.registry.spec_for(s.models["chat"]), s.params)
        messages, trimmed = s.api_messages(budget=budget)
        if trimmed:
            dim(f"(left out the {trimmed} oldest turn(s) to stay under the token limit)")
        turn.candidates = [run_chat(app, messages)]
        ok = True
    finally:
        if not ok:  # nothing was added; put the message back in the prompt
            s.turns.pop()
            app.prefill = text
    app.save()
    auto_title(app)


def auto_title(app: App) -> None:
    s = app.session
    if not (s.title_is_auto and s.title == DEFAULT_TITLE and s.turns and s.turns[0].reply):
        return
    if len(s.turns) > 3:
        return
    selectable = {e.spec.id for e in app.registry.selectable(Kind.CHAT)}
    model = TITLE_MODEL if TITLE_MODEL in selectable else s.models["chat"]
    first = s.turns[0]
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Write a title of 3 to 6 words for this conversation. "
                "Reply with only the title, no quotes.",
            },
            {
                "role": "user",
                "content": f"User: {first.user[:500]}\n\nAssistant: {first.reply.text[:500]}",
            },
        ],
        max_completion_tokens=256,
    )
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"
    try:
        resp = app.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        app.tracker.record(model, tokens=int(getattr(usage, "total_tokens", 0) or 0))
        content = (resp.choices[0].message.content or "").strip()
    except Exception:  # titles are cosmetic: never let them interrupt the chat
        return
    if not content:
        return
    title = clean_title(content.splitlines()[0].strip(" \"'`*#."), 60)
    if title:
        s.title = title
        app.save()


# ======================================================================
# Helpers for commands
# ======================================================================


def choose(
    app: App,
    items: list[T],
    label: Callable[[T], str],
    what: str,
    arg: str,
    display: Callable[[T], str] | None = None,
) -> T | None:
    """Pick an item by number, by title (quotes force title), or from a prompt."""
    if not items:
        raise CommandError(f"There are no {what}s to choose from.")
    show = display or label
    arg = arg.strip()
    if not arg:
        for i, item in enumerate(items, 1):
            print(f"  {i:>2}. {show(item)}")
        arg = app.ask(f"Which {what}? (number or title, blank to cancel) ").strip()
        if not arg:
            info("Cancelled.")
            return None
    force_title = arg[:1] in ('"', "'")
    try:
        text = " ".join(shlex.split(arg)).strip()
    except ValueError:
        text = arg
    n = None if force_title else parse_index(text)
    if n is not None:
        if 1 <= n <= len(items):
            return items[n - 1]
        raise CommandError(f"There is no {what} number {text[:20]} (choose 1-{len(items)}).")
    folded = text.casefold()
    matches = [it for it in items if label(it).casefold() == folded] or [
        it for it in items if folded in label(it).casefold()
    ]
    if not matches:
        raise CommandError(f'No {what} matches "{text}".')
    if len(matches) == 1:
        return matches[0]
    info(f'Several {what}s match "{text}":')
    return choose(app, matches, label, what, "", display)


def pick_turn(app: App, arg: str, *, blank_is_last: bool) -> int | None:
    turns = app.session.turns
    if not turns:
        raise CommandError("This chat has no messages yet.")
    arg = arg.strip()
    if not arg:
        message_list(app, "")
        hint = "Enter = last" if blank_is_last else "blank to cancel"
        arg = app.ask(f"Turn number ({hint}): ").strip()
        if not arg:
            return len(turns) - 1 if blank_is_last else None
    n = parse_index(arg)
    if n is None or not 1 <= n <= len(turns):
        raise CommandError(f"Turn number must be between 1 and {len(turns)}.")
    return n - 1


def clean_path(raw: str) -> Path:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        raw = raw[1:-1]
    if os.name != "nt":
        raw = raw.replace("\\ ", " ")  # drag-and-drop escapes spaces
    try:
        return Path(raw).expanduser()
    except RuntimeError:  # "~someone" that doesn't exist
        return Path(raw)


def validate_audio(path: Path) -> None:
    try:
        if not path.exists():
            raise CommandError(f"File not found: {preview(str(path), 100)}")
        if not path.is_file():
            raise CommandError(f"Not a file: {preview(str(path), 100)}")
        if path.suffix.lower() not in AUDIO_EXTS:
            raise CommandError(
                f"Unsupported audio type '{path.suffix}'. Use one of: {', '.join(sorted(AUDIO_EXTS))}"
            )
        size = path.stat().st_size
    except OSError as exc:  # e.g. "File name too long", permission problems
        raise CommandError(f"Can't open that file: {exc.strerror or exc}") from None
    if size == 0:
        raise CommandError("That audio file is empty.")
    if size > MAX_AUDIO_BYTES:
        raise CommandError(
            f"File is {size / 1048576:.1f} MB; the limit is {MAX_AUDIO_BYTES // 1048576} MB."
        )


def save_binary(response: Any, path: Path) -> None:
    writer = getattr(response, "write_to_file", None)
    if writer is not None:
        writer(path)
    else:
        path.write_bytes(response.read())


def refresh_registry(app: App, *, force: bool) -> None:
    info("Checking available models (this uses a few requests)...")
    try:
        app.registry.refresh(app.client, app.tracker, force=force, say=dim)
    except groq.APIConnectionError:
        warn("Couldn't reach Groq to check models; using the saved list.")
        return
    counts: dict[str, int] = {}
    for e in app.registry.entries.values():
        counts[e.status] = counts.get(e.status, 0) + 1
    info(
        f"Models: {counts.get('verified', 0)} verified, "
        f"{len(app.registry.entries) - counts.get('verified', 0)} not offered. "
        "See /model list."
    )


# ======================================================================
# Commands: chat
# ======================================================================


@command("chat", "new", usage="[title]", summary="Start a new chat", group="Chat")
def chat_new(app: App, arg: str) -> None:
    app.session = app.new_session(clean_title(arg) or None)
    info(f"Started a new chat: {app.session.title}")


@command("chat", "list", summary="List chats, newest first", group="Chat")
def chat_list(app: App, arg: str) -> None:
    chats = app.store.list()
    cur = app.session
    if not chats and not cur.turns:
        info("No saved chats yet.")
        return
    for i, c in enumerate(chats, 1):
        mark = "*" if c.id == cur.id else " "
        print(f"{mark} {i:>2}. {c.title}  ({len(c.turns)} turns, {fmt_time(c.updated_at)})")
    if not any(c.id == cur.id for c in chats):
        print(f"*  -  {cur.title}  (current, not saved yet)")


@command("chat", "switch", usage="[num|title]", summary="Resume a chat", group="Chat")
def chat_switch(app: App, arg: str) -> None:
    chats = app.store.list()
    target = choose(app, chats, lambda c: c.title, "chat", arg)
    if target is None:
        return
    if target.id == app.session.id:
        info("You're already in that chat.")
        return
    changed = app.check_models(target)
    app.session = target
    if changed:
        app.save()
    info(f"Switched to: {target.title}")
    if target.turns:
        last = target.turns[-1]
        dim(f"Last message: {preview(last.user)}")


@command("chat", "rename", usage="<title>", summary="Rename this chat", group="Chat")
def chat_rename(app: App, arg: str) -> None:
    title = clean_title(arg)
    if not title:
        raise CommandError("Usage: /chat rename <title>")
    app.session.title = title
    app.session.title_is_auto = False
    app.save()
    info(f"Renamed to: {app.session.title}")


@command("chat", "delete", usage="[num|title]", summary="Delete a chat (asks to confirm)", group="Chat")
def chat_delete(app: App, arg: str) -> None:
    chats = app.store.list()
    target = choose(app, chats, lambda c: c.title, "chat", arg)
    if target is None:
        return
    if not app.confirm(f'Delete "{target.title}"?'):
        info("Cancelled.")
        return
    app.store.delete(target.id)
    info("Deleted.")
    if target.id == app.session.id:
        app.session = app.new_session()
        info("Started a new chat.")


@command("chat", "clear", summary="Clear messages in this chat", group="Chat")
def chat_clear(app: App, arg: str) -> None:
    if not app.session.turns:
        info("Nothing to clear.")
        return
    if not app.confirm("Clear all messages in this chat?"):
        info("Cancelled.")
        return
    app.session.turns.clear()
    app.save()
    info("Cleared.")


# ======================================================================
# Commands: messages
# ======================================================================


@command("message", "list", summary="Show numbered turns", group="Messages")
def message_list(app: App, arg: str) -> None:
    turns = app.session.turns
    if not turns:
        info("This chat has no messages yet.")
        return
    for i, t in enumerate(turns, 1):
        reply = t.reply.text if t.reply else "(no reply)"
        print(f"{i:>3}. you: {preview(t.user)}")
        print(f"     ai:  {preview(reply)}")


@command(
    "message",
    "edit",
    usage="[num]",
    summary="Edit a message (user edits discard later turns and regenerate)",
    group="Messages",
)
def message_edit(app: App, arg: str) -> None:
    s = app.session
    idx = pick_turn(app, arg, blank_is_last=True)
    turn = s.turns[idx]
    which = "u"
    if turn.reply:
        which = app.ask("Edit the [u]ser message or the [a]ssistant reply? (Enter = u): ").strip().lower() or "u"
        if which not in ("u", "a"):
            raise CommandError("Please answer u or a.")

    if which == "a":
        new = app.ask("edit> ", default=turn.reply.text).strip()
        if not new or new == turn.reply.text:
            info("No changes.")
            return
        turn.reply.text = new
        app.save()
        info("Reply updated.")
        return

    new = app.ask("edit> ", default=turn.user).strip()
    if not new or new == turn.user:
        info("No changes.")
        return
    later = len(s.turns) - idx - 1
    if later and not app.confirm(
        f"This discards the {later} turn(s) after it and regenerates the reply. Continue?"
    ):
        info("Cancelled.")
        return

    backup_turns, old_user = list(s.turns), turn.user
    old_cands, old_sel = turn.candidates, turn.selected
    s.turns = s.turns[: idx + 1]
    turn.user, turn.candidates, turn.selected = new, [], 0
    ok = False
    try:
        budget = context_budget(app.registry.spec_for(s.models["chat"]), s.params)
        messages, _ = s.api_messages(budget=budget)
        turn.candidates = [run_chat(app, messages)]
        ok = True
    finally:
        if not ok:  # put everything back exactly as it was
            s.turns = backup_turns
            turn.user, turn.candidates, turn.selected = old_user, old_cands, old_sel
    app.save()


@command("message", "delete", usage="[num]", summary="Delete a turn (message + reply)", group="Messages")
def message_delete(app: App, arg: str) -> None:
    idx = pick_turn(app, arg, blank_is_last=False)
    if idx is None:
        info("Cancelled.")
        return
    turn = app.session.turns[idx]
    if not app.confirm(f'Delete turn {idx + 1} ("{preview(turn.user, 40)}")?'):
        info("Cancelled.")
        return
    del app.session.turns[idx]
    app.save()
    info("Deleted.")


@command("regenerate", summary="Re-run the last reply, then pick the better one", group="Messages")
def regenerate(app: App, arg: str) -> None:
    s = app.session
    if not s.turns or not s.turns[-1].reply:
        raise CommandError("There's no reply to regenerate yet.")
    turn = s.turns[-1]
    previous = min(turn.selected, len(turn.candidates) - 1)

    # At a low temperature a regenerate can come back identical; nudge it up.
    temp = s.params.temperature
    if temp is not None and temp < 0.7:
        temp = 0.8

    keep: int | None = None
    try:
        while keep is None:
            budget = context_budget(app.registry.spec_for(s.models["chat"]), s.params)
            messages, _ = s.api_messages(drop_last_reply=True, budget=budget)
            turn.candidates.append(run_chat(app, messages, temperature=temp))
            while keep is None:
                n = len(turn.candidates)
                for i, c in enumerate(turn.candidates, 1):
                    tag = "previous" if i - 1 == previous else ("newest" if i == n else "")
                    dim(f"  [{i}] {tag} · {c.model} · {preview(c.text, 50)}")
                try:
                    answer = app.ask(
                        f"Keep which? [1-{n}] number, r = regenerate again, v<N> = view (Enter = newest): "
                    ).strip().lower()
                except KeyboardInterrupt:
                    keep = previous
                    info("Kept the previous reply.")
                    break
                pick = parse_index(answer)
                view = parse_index(answer[1:]) if answer.startswith("v") else None
                if answer == "":
                    keep = n - 1
                elif pick is not None and 1 <= pick <= n:
                    keep = pick - 1
                elif view is not None and 1 <= view <= n:
                    print(turn.candidates[view - 1].text)
                elif answer == "r":
                    break  # generate another candidate
                else:
                    warn("Please enter a number, r, or v<N>.")
    finally:
        if keep is None:  # an error or interrupt: fall back to the original reply
            keep = previous
        turn.candidates = [turn.candidates[keep]]
        turn.selected = 0
        app.save()


# ======================================================================
# Commands: system prompt and examples
# ======================================================================


@command("system", summary="Show the current system prompt", group="Prompt & examples")
def system_show(app: App, arg: str) -> None:
    no_args(arg, "system")
    info(app.session.system_prompt or "(empty)")


@command("system", "set", usage="<text>", summary="Set it for this chat", group="Prompt & examples")
def system_set(app: App, arg: str) -> None:
    text = arg.strip() or app.ask("system> ", default=app.session.system_prompt).strip()
    if not text:
        raise CommandError("The system prompt can't be empty. /system reset restores the default.")
    app.session.system_prompt = text
    app.save()
    info("System prompt updated.")


@command("system", "default", usage="[text]", summary="Show / set the default for new chats", group="Prompt & examples")
def system_default(app: App, arg: str) -> None:
    text = arg.strip()
    if not text:
        info(f"Default system prompt for new chats:\n{app.config.default_system}")
        return
    app.config.default_system = text
    app.config.save(CONFIG_FILE)
    info("Default updated (applies to new chats).")


@command("system", "reset", summary="Reset this chat to the default", group="Prompt & examples")
def system_reset(app: App, arg: str) -> None:
    app.session.system_prompt = app.config.default_system
    app.save()
    info("System prompt reset to the default.")


@command("examples", summary="Show scripted examples", group="Prompt & examples")
def examples_show(app: App, arg: str) -> None:
    no_args(arg, "examples")
    if not app.session.examples:
        info("No examples yet. Add one with /examples add.")
        return
    for i, (u, a) in enumerate(app.session.examples, 1):
        print(f"{i:>2}. user: {preview(u)}\n    assistant: {preview(a)}")


@command("examples", "add", summary="Add a user -> assistant example pair", group="Prompt & examples")
def examples_add(app: App, arg: str) -> None:
    user = app.ask("Example user message: ").strip()
    if not user:
        info("Cancelled.")
        return
    assistant = app.ask("Example assistant reply: ").strip()
    if not assistant:
        info("Cancelled.")
        return
    app.session.examples.append([user, assistant])
    app.save()
    info(f"Added example {len(app.session.examples)}.")


@command("examples", "remove", usage="[num]", summary="Remove an example", group="Prompt & examples")
def examples_remove(app: App, arg: str) -> None:
    ex = app.session.examples
    if not ex:
        raise CommandError("There are no examples to remove.")
    labels = [f"user: {preview(u, 40)}  ->  assistant: {preview(a, 40)}" for u, a in ex]
    idx_items = list(range(len(ex)))
    pick = choose(app, idx_items, lambda i: labels[i], "example", arg)
    if pick is None:
        return
    del ex[pick]
    app.save()
    info("Removed.")


# ======================================================================
# Commands: models
# ======================================================================


@command("model", summary="Show the models used by this chat", group="Models")
def model_show(app: App, arg: str) -> None:
    no_args(arg, "model")
    for kind in KIND_ORDER:
        mid = app.session.models.get(kind.value, "?")
        spec = app.registry.spec_for(mid)
        tag = " (preview)" if spec.preview else ""
        print(f"  {SLOT_LABEL[kind]:<15} {mid}{tag}   {_c('2', fmt_limits(spec))}")


@command("model", "list", summary="List verified models", group="Models")
def model_list(app: App, arg: str) -> None:
    selectable = app.registry.selectable()
    if not app.registry.from_cache:
        warn("These models haven't been verified against Groq yet. Try /model refresh.")
    n = 0
    current_kind = None
    for e in selectable:
        n += 1
        if e.spec.kind is not current_kind:
            current_kind = e.spec.kind
            print(f"\n{SLOT_LABEL[current_kind].upper()}")
        active = "*" if app.session.models.get(e.spec.kind.value) == e.spec.id else " "
        tag = " (preview)" if e.spec.preview else ""
        ctx = f" · {human(e.spec.context_window)} context" if e.spec.context_window else ""
        print(f" {active} {n:>2}. {e.spec.id}{tag}  {_c('2', fmt_limits(e.spec) + ctx)}")
    others = app.registry.not_selectable()
    if others:
        print("\nNOT AVAILABLE RIGHT NOW")
        for e in others:
            print(f"      {e.spec.id}  {_c('2', f'{e.status}: ' + preview(e.note, 70))}")


@command("model", "switch", usage="[num|name]", summary="Switch model (fills the matching slot)", group="Models")
def model_switch(app: App, arg: str) -> None:
    entries = app.registry.selectable()
    if not entries:
        raise CommandError("No verified models are available. Try /model refresh.")
    entry = choose(
        app,
        entries,
        lambda e: e.spec.id,
        "model",
        arg,
        display=lambda e: f"{e.spec.id}  [{e.spec.kind.value}]",
    )
    if entry is None:
        return
    app.session.models[entry.spec.kind.value] = entry.spec.id
    app.save()
    info(f"{SLOT_LABEL[entry.spec.kind].capitalize()} model set to {entry.spec.id}.")
    if entry.spec.preview:
        warn("This is a preview model and may be discontinued without notice.")


@command("model", "refresh", summary="Re-check the live model list now", group="Models")
def model_refresh(app: App, arg: str) -> None:
    refresh_registry(app, force=True)
    if app.check_models(app.session):
        app.save()


# ======================================================================
# Commands: limits and parameters
# ======================================================================


@command("limit", summary="Tokens used by the last message", group="Limits")
def limit_last(app: App, arg: str) -> None:
    no_args(arg, "limit")
    u = app.session.usage
    if not u.requests:
        info("No messages sent in this chat yet.")
        return
    est = " (estimated)" if u.last_estimated else ""
    total = u.last_prompt + u.last_completion
    info(f"Last message: {u.last_prompt:,} prompt + {u.last_completion:,} completion = {total:,} tokens{est}")


@command("limit", "chat", summary="Requests and tokens used by this chat", group="Limits")
def limit_chat(app: App, arg: str) -> None:
    u = app.session.usage
    total = u.prompt_tokens + u.completion_tokens
    info(
        f"This chat: {u.requests:,} request(s), {u.prompt_tokens:,} prompt + "
        f"{u.completion_tokens:,} completion = {total:,} tokens"
    )


@command("limit", "remaining", summary="Remaining requests/tokens (minute and day)", group="Limits")
def limit_remaining(app: App, arg: str) -> None:
    current = [app.session.models[k.value] for k in KIND_ORDER if k.value in app.session.models]
    models = [current[0]] + [m for m in app.tracker.active_models() if m != current[0]]
    for mid in models:
        spec = app.registry.spec_for(mid)
        rows = app.tracker.remaining(spec)
        print(f"\n{mid}")
        if not rows:
            dim("  no known limits for this model")
        for key, (limit, used, left) in rows.items():
            print(f"  {LIMIT_LABELS[key]:<22} {used:>8,} / {limit:<9,} {left:>9,} left")
        snap = app.tracker.headers.get(mid)
        if snap:
            age = int(time.time() - snap["ts"])
            dim(f"  server-reported ({age}s ago):")
            for k, label in HEADER_LABELS.items():
                if k in snap["values"]:
                    dim(f"    {label}: {snap['values'][k]}")
    dim(
        "\nLocal counts cover this app only and use a rolling 24h window; "
        "other programs sharing your key aren't counted."
    )


@command(
    "params",
    usage="[temperature|max_tokens|reasoning|voice <value|reset>]",
    summary="Show / set generation settings for this chat",
    group="Settings",
)
def params_cmd(app: App, arg: str) -> None:
    p = app.session.params
    parts = arg.split()
    if not parts:
        print(f"  temperature  {p.temperature if p.temperature is not None else '(model default)'}")
        print(f"  max_tokens   {p.max_tokens or '(model default)'}")
        print(f"  reasoning    {p.reasoning_effort or '(model default)'}")
        print(f"  voice        {p.voice or '(model default)'}")
        return
    if len(parts) < 2:
        raise CommandError("Usage: /params <temperature|max_tokens|reasoning|voice> <value|reset>")
    key, val = parts[0].lower(), parts[1]
    reset = val.lower() == "reset"
    if key == "temperature":
        if reset:
            p.temperature = None
        else:
            try:
                t = float(val)
            except ValueError:
                raise CommandError("temperature must be a number between 0 and 2.") from None
            if not 0 <= t <= 2:
                raise CommandError("temperature must be between 0 and 2.")
            p.temperature = t
    elif key == "max_tokens":
        if reset:
            p.max_tokens = None
        else:
            n = parse_index(val)
            if n is None or not 1 <= n <= 1_000_000:
                raise CommandError("max_tokens must be a whole number between 1 and 1,000,000.")
            p.max_tokens = n
    elif key == "reasoning":
        if reset:
            p.reasoning_effort = None
        else:
            if val.lower() not in ("low", "medium", "high"):
                raise CommandError("reasoning must be low, medium or high.")
            p.reasoning_effort = val.lower()
    elif key == "voice":
        if not reset and (len(val) > 64 or not val.isprintable()):
            raise CommandError("That doesn't look like a voice name.")
        p.voice = None if reset else val
    else:
        raise CommandError(f"Unknown setting '{key}'. Choose temperature, max_tokens, reasoning or voice.")
    app.save()
    info("Updated.")


# ======================================================================
# Commands: audio and guard models
# ======================================================================


@command("transcribe", usage="[path]", summary="Audio file -> text", group="Audio and guard models")
def transcribe_cmd(app: App, arg: str) -> None:
    raw = arg.strip() or app.ask("Path to audio file: ").strip()
    if not raw:
        info("Cancelled.")
        return
    path = clean_path(raw)
    validate_audio(path)
    model = app.session.models["stt"]
    info(f"Transcribing {path.name} with {model}...")
    result, headers = call_with_headers(
        app.client.audio.transcriptions,
        file=(path.name, path.read_bytes()),
        model=model,
        response_format="verbose_json",
    )
    app.tracker.note_headers(model, headers)
    text = result if isinstance(result, str) else (getattr(result, "text", "") or "")
    try:
        seconds = float(getattr(result, "duration", 0) or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    app.tracker.record(model, tokens=0, audio_seconds=seconds)
    if not text.strip():
        info("(no speech detected)")
        return
    print(text.strip())
    if seconds:
        dim(f"[{model} · {seconds:.1f}s of audio]")
    if app.confirm("Send this transcript to the chat as a message?"):
        send_message(app, text.strip())


@command("speak", usage="[text|last]", summary="Text -> audio file", group="Audio and guard models")
def speak_cmd(app: App, arg: str) -> None:
    s = app.session
    text = arg.strip()
    if text.lower() == "last":
        reply = s.turns[-1].reply if s.turns else None
        if not reply:
            raise CommandError("There's no assistant reply to speak yet.")
        text = reply.text
    elif not text:
        text = app.ask("Text to speak: ").strip()
        if not text:
            info("Cancelled.")
            return
    model = s.models["tts"]
    voice = s.params.voice or DEFAULT_VOICES.get(model, "autumn")
    info(f"Generating speech with {model} (voice: {voice})...")
    response = app.client.audio.speech.create(
        model=model, voice=voice, input=text, response_format="wav"
    )
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_DIR / f"speech-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    save_binary(response, out)
    app.tracker.record(model, tokens=estimate_tokens(text))
    info(f"Saved: {out}")


@command("classify", usage="[text]", summary="Run text through a guard model", group="Audio and guard models")
def classify_cmd(app: App, arg: str) -> None:
    s = app.session
    text = arg.strip() or app.ask("Text to classify: ").strip()
    if not text:
        info("Cancelled.")
        return
    model = s.models["guard"]
    messages: list[dict] = []
    if "safeguard" in model and s.system_prompt:  # safeguard models read a policy
        messages.append({"role": "system", "content": s.system_prompt})
    messages.append({"role": "user", "content": text})
    resp = app.client.chat.completions.create(model=model, messages=messages)
    usage = getattr(resp, "usage", None)
    app.tracker.record(model, tokens=int(getattr(usage, "total_tokens", 0) or 0))
    info(f"[{model}] {resp.choices[0].message.content}")


# ======================================================================
# Commands: help and exit
# ======================================================================


@command("help", summary="Show this help", group="Other")
def help_cmd(app: App, arg: str) -> None:
    groups: dict[str, list[Command]] = {}
    for cmd in COMMANDS.values():
        groups.setdefault(cmd.group, []).append(cmd)
    for group, cmds in groups.items():
        print(f"\n{group}")
        for c in cmds:
            name = ("/" + " ".join(c.path) + (" " + c.usage if c.usage else "")).strip()
            print(f"  {name:<34} {c.summary}")
    print(
        "\nAnything not starting with '/' is sent as a message. Start with '//' to send a "
        "literal '/'.\nAlt+Enter inserts a new line. Ctrl+C stops a reply. Ctrl+D quits."
    )


@command("exit", summary="Quit", group="Other")
def exit_cmd(app: App, arg: str) -> None:
    raise QuitApp


@command("quit", summary="Quit", group="Other")
def quit_cmd(app: App, arg: str) -> None:
    raise QuitApp


# ======================================================================
# Main loop
# ======================================================================


def handle_line(app: App, line: str) -> None:
    parsed = parse(line)
    if parsed.kind == "empty":
        return
    if parsed.kind == "message":
        send_message(app, parsed.text)
        return
    cmd = COMMANDS.get(parsed.path)
    if cmd is None:
        report_unknown(line)
        return
    cmd.handler(app, parsed.text)


def get_version() -> str:
    try:
        return metadata.version("groq-chat-cli")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="groqchat", description="Chat with Groq models in your terminal.")
    parser.add_argument("--chat", metavar="ID", help="resume the chat whose id starts with ID")
    parser.add_argument("-c", "--continue", dest="cont", action="store_true", help="resume the most recent chat")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    parser.add_argument("--version", action="version", version=f"groq-chat-cli {get_version()}")
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv())
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "Error: GROQ_API_KEY not found.\n"
            "Create a key at https://console.groq.com/keys and add this line to your .env file:\n"
            "    GROQ_API_KEY=your_key_here"
        )

    for d in (APP_DIR, CHATS_DIR, AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # max_retries=0: show rate-limit errors immediately instead of silently waiting.
    client = Groq(api_key=api_key, max_retries=0, timeout=60.0)
    config = Config.load(CONFIG_FILE)
    tracker = UsageTracker(USAGE_FILE)
    registry = ModelRegistry(MODELS_CACHE_FILE)
    registry.load()
    store = ChatStore(CHATS_DIR, warn)
    app = App(client, config, registry, tracker, store, debug=args.debug)

    try:
        if registry.is_stale():
            refresh_registry(app, force=False)
    except groq.AuthenticationError as exc:
        sys.exit("Error: " + (explain_error(exc) or "Invalid API key."))
    except groq.APIError as exc:
        warn(explain_error(exc) or str(exc))

    # Choose the starting chat.
    start: ChatSession | None = None
    if args.chat:
        start = store.find_prefix(args.chat)
        if start is None:
            warn(f"No chat with an id starting with '{args.chat}'; starting a new one.")
    elif args.cont:
        chats = store.list()
        start = chats[0] if chats else None
    if start is not None:
        if app.check_models(start):
            store.save(start)
        app.session = start
    else:
        app.session = app.new_session()
        app.check_models(app.session)

    info(f"groq-chat-cli {get_version()} · chat model: {app.session.models['chat']}")
    dim("Type a message, or /help for commands. Alt+Enter = new line, Ctrl+D = quit.")
    if start is not None:
        info(f"Resumed: {start.title}")

    while True:
        try:
            line = app.read_line()
        except EOFError:
            break
        except KeyboardInterrupt:
            continue  # Ctrl+C at the prompt just clears the line
        try:
            handle_line(app, line)
        except QuitApp:
            break
        except Stopped as exc:
            dim(str(exc))
        except CommandError as exc:
            err(str(exc))
        except KeyboardInterrupt:
            dim("Cancelled.")
        except Exception as exc:  # last resort so one bug never ends the session
            friendly = explain_error(exc)
            if friendly:
                err(friendly)
            else:
                err(f"Unexpected error: {exc!r}. Run with --debug for details.")
            if app.debug:
                traceback.print_exc()
    info("Goodbye.")


if __name__ == "__main__":
    main()
