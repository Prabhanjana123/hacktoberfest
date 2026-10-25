"""Collect recent headlines from Yahoo Finance (via yfinance) and Google News RSS."""
from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timedelta, timezone

import feedparser
import yfinance as yf


def _iso(value) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def yahoo_headlines(ticker: str, limit: int = 20) -> list[dict]:
    t = yf.Ticker(ticker)
    try:
        raw = t.get_news(count=limit)
    except Exception:
        try:
            raw = t.news
        except Exception:
            raw = []
    items = []
    for r in raw or []:
        c = r.get("content", r) if isinstance(r, dict) else {}
        title = (c.get("title") or "").strip()
        if not title:
            continue
        provider = c.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else r.get("publisher")
        url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
               or r.get("link"))
        items.append({
            "title": title,
            "summary": (c.get("summary") or c.get("description") or "").strip(),
            "publisher": publisher or "Yahoo Finance",
            "published": _iso(c.get("pubDate") or c.get("displayTime") or r.get("providerPublishTime")),
            "url": url, "source": "yahoo",
        })
    return items


def google_headlines(query: str, limit: int = 20) -> list[dict]:
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
           + "+when:14d&hl=en-US&gl=US&ceid=US:en")
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:limit]:
        title = e.get("title", "")
        publisher = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(publisher) - 3]
        published = None
        if e.get("published_parsed"):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        items.append({"title": title.strip(), "summary": "", "publisher": publisher or "Google News",
                      "published": published, "url": e.get("link"), "source": "google_news"})
    return items


def _key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:60]


def collect_headlines(ticker: str, name: str, limit: int = 15, max_age_days: int = 14) -> list[dict]:
    pool: list[dict] = []
    for fetch in (lambda: yahoo_headlines(ticker),
                  lambda: google_headlines(f'"{name}" stock' if name and name != ticker else f"{ticker} stock")):
        try:
            pool.extend(fetch())
        except Exception:
            continue

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    seen, out = set(), []
    for h in pool:
        k = _key(h["title"])
        if not k or k in seen:
            continue
        if h["published"] and datetime.fromisoformat(h["published"]) < cutoff:
            continue
        seen.add(k)
        out.append(h)
    out.sort(key=lambda h: h["published"] or "", reverse=True)
    return out[:limit]


def headline_evidence(headlines: list[dict]) -> list[dict]:
    ev = []
    for i, h in enumerate(headlines, 1):
        date = (h.get("published") or "")[:10] or "date unknown"
        text = f'Headline ({h["publisher"]}, {date}): "{h["title"]}".'
        if h.get("summary"):
            text += f" Summary: {h['summary'][:320]}"
        ev.append({"id": f"H{i}", "kind": "headline", "text": text, "data": h})
    return ev
