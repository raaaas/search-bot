# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Clients for the chat and embedding endpoints (both OpenAI-compatible)."""
import json
import numpy as np
import requests
from . import config

_embed_dim = None


def auth() -> dict:
    """Bearer header for a keyed provider; empty dict for a local server."""
    return {"Authorization": "Bearer " + config.API_KEY} if config.API_KEY else {}


def embed(texts, is_query: bool = False, batch_size: int = 8):
    """Return list of np vectors (normalized). Queries get a retrieval-instruction prefix."""
    global _embed_dim
    out = []
    for i in range(0, len(texts), batch_size):
        raw = texts[i:i + batch_size]
        batch = [config.QUERY_INSTRUCT + t if is_query else t for t in raw]
        payload = {"input": batch}
        if config.EMBED_MODEL:
            payload["model"] = config.EMBED_MODEL
        # vectors are L2-normalized here, so no server-side normalize option is used
        r = requests.post(config.EMBED_URL + "/embeddings", json=payload,
                          headers=auth(), timeout=180)
        if r.status_code == 500 and len(batch) > 1:
            # one oversized text poisons the batch — go one-by-one. Retry the
            # *original* texts, never the prefixed ones: handing back a batch
            # element would either re-apply the instruction or drop it.
            for one in raw:
                out.extend(embed([one], is_query=is_query, batch_size=1))
            continue
        if r.status_code == 500 and len(batch) == 1 and "too large" in r.text.lower():
            # llama.cpp rejects any single sequence longer than its physical
            # batch (--batch-size, 512 by default) and other servers cap by
            # tokens too, so an oversized chunk must not kill the index. Split
            # it, mean-pool the parts and renormalize: the same pooling the
            # server would have done, just with a smaller window.
            parts = [raw[0][j:j + config.EMBED_MAX_CHARS]
                     for j in range(0, len(raw[0]), config.EMBED_MAX_CHARS)]
            if len(parts) < 2:
                r.raise_for_status()
            v = np.mean([embed([p], is_query=is_query, batch_size=1)[0] for p in parts],
                        axis=0)
            out.append(v / np.linalg.norm(v))
            continue
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        for d in data:
            v = np.asarray(d["embedding"], dtype=np.float32)
            n = np.linalg.norm(v)
            if n == 0:
                raise RuntimeError("embedder returned zero vector — check --pooling")
            v = v / n
            if _embed_dim is None:
                _embed_dim = int(v.shape[0])
            out.append(v)
    return out


def embed_dim() -> int:
    global _embed_dim
    if _embed_dim is None:
        embed(["warmup"])
    return _embed_dim or 768


_chat_model = None


def chat_model():
    """Resolve the chat model id: config override or auto-detect from /models."""
    global _chat_model
    if config.CHAT_MODEL:
        return config.CHAT_MODEL
    if _chat_model is None:
        try:
            data = requests.get(config.CHAT_URL + "/models",
                                headers=auth(), timeout=10).json()["data"]
            _chat_model = data[0]["id"]
        except Exception:
            _chat_model = "default"
    return _chat_model


import re

_CONTROL_BLOCK = re.compile(
    r"<\|[^|>]{0,40}_start\|>.*?<\|[^|>]{0,40}_end\|>", re.S)
_CONTROL_TOKEN = re.compile(r"<\|[^|>]{0,40}\|>")
_PSEUDO_CALL = re.compile(r"\b(?:google|search|tool_call|functions?\.[\w.]+)\s*\([^)]*\)")


def sanitize(text: str) -> str:
    """Strip hallucinated tool-call syntax a local GGUF may emit into plain
    content: whole <..._start|>...<..._end|> blocks, stray control tokens,
    and pseudo-call fragments left over."""
    if not text:
        return ""
    text = _CONTROL_BLOCK.sub(" ", text)
    text = _CONTROL_TOKEN.sub(" ", text)
    text = _PSEUDO_CALL.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _one_shot(payload):
    r = requests.post(config.CHAT_URL + "/chat/completions", json=payload,
                      headers=auth(), timeout=config.QUERY_TIMEOUT_S)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return (choice["message"].get("content") or "").strip(), \
        choice.get("finish_reason")


def _stream_once(payload, on_delta):
    """Stream one completion; feed deltas through a control-token scrubber
    so '<|tool_call_start|>' never reaches the UI. Returns (clean_text, finish)."""
    buf = ""            # unflushed tail (may be a partial control token)
    emitted = ""        # what the UI has seen
    pending = ""        # raw text seen, in case we must fall back whole
    finish = None
    r = requests.post(config.CHAT_URL + "/chat/completions",
                      json={**payload, "stream": True},
                      headers=auth(), timeout=config.QUERY_TIMEOUT_S, stream=True)
    r.raise_for_status()
    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            ch = json.loads(data)["choices"][0]
        except Exception:
            continue
        delta = (ch.get("delta") or {}).get("content") or ""
        finish = ch.get("finish_reason") or finish
        if not delta:
            continue
        pending += delta
        buf += delta
        # flush everything except a tail that could still become a control token
        flush_upto = len(buf)
        i = buf.find("<|")
        if i != -1:
            close = buf.find("|>", i)
            if close == -1 and len(buf) - i <= 50:
                flush_upto = i                      # maybe a token forming at end
            elif close == -1:
                i2 = buf.rfind("<|")                # too long to be a token -> safe
                flush_upto = i2 if i2 > i else len(buf)
        out = buf[:flush_upto]
        if out:
            buf = buf[flush_upto:]
            out = _CONTROL_BLOCK.sub(" ", out)
            out = _CONTROL_TOKEN.sub(" ", out)
            emitted += out
            on_delta(out)
    # tail: flush only if it is not part of a tool block
    tail = _CONTROL_BLOCK.sub(" ", buf)
    tail = _CONTROL_TOKEN.sub(" ", tail)
    if tail:
        emitted += tail
        on_delta(tail)
    if "<|" in pending and "<|" not in emitted.replace(" ", "") and len(emitted) <= 80:
        # output was almost pure control-token garbage: treat as empty so caller retries
        return "", finish
    return emitted.strip(), finish


def chat_llm(messages, temperature=0.3, max_tokens=700, on_delta=None, on_retry=None):
    """Completion from the chat endpoint. Returns clean prose text.

    Two failure modes of reasoning/tool-trained chat models are handled:
      1. thinking eats the token budget -> empty content (finish=length)
      2. tool-call syntax bleeds into content as raw control tokens
         ('<|tool_call_start|>[google(...)]<|tool_call_end|>')
    Strategy: up to 3 attempts with growing budget and temperature 0 on
    retries; a candidate is accepted only if it is non-empty, not truncated
    short, and contains no '<|' control tokens. If on_delta is given the
    text is streamed to it live (scrubbed)."""
    last = ""
    for attempt, (budget, temp) in enumerate(
            [(max_tokens, temperature),
             (int(max_tokens * 2.5), 0.0),
             (max_tokens * 5, 0.0)]):
        if attempt and on_retry:
            on_retry(budget)
        payload = {"model": chat_model(), "messages": messages,
                   "temperature": temp, "max_tokens": budget}
        if on_delta:
            try:
                text, fin = _stream_once(payload, on_delta)
            except Exception:
                text, fin = _one_shot(payload)      # fall back to non-stream
                text = sanitize(text)
                if text:
                    on_delta(text)
        else:
            text, fin = _one_shot(payload)
        if text and _CONTROL_TOKEN.search(text):
            cleaned = sanitize(text)
            if len(cleaned) > 80:
                text = cleaned              # salvage the prose, drop the bleed
            else:
                text = ""                   # mostly garbage -> retry
        if text:
            last = text
            if not (fin == "length" and len(text) < 120):
                return text
        # else: empty or short-truncated -> grow budget / temp 0 and retry
    return sanitize(last)
