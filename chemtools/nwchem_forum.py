"""Search the archived NWChem community forums.

The forums are hosted as static HTML at:
  https://nwchemgit.github.io/Special_AWCforum/

This module fetches and parses forum pages at runtime to find threads
matching a search query. Useful when the agent encounters unusual errors
or edge-case NWChem behavior that may have been discussed by the community.
"""
from __future__ import annotations

import html
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any


_FORUM_BASE = "https://nwchemgit.github.io/Special_AWCforum"

# Subforums worth searching (id, slug, label)
_SUBFORUMS = [
    (5, "Running_NWChem", "Running NWChem"),
    (6, "NWChem_functionality", "NWChem functionality"),
    (3, "General_Topics", "General Topics"),
    (4, "Compiling_NWChem", "Compiling NWChem"),
    (9, "QMMM", "QM/MM"),
]

_TIMEOUT = 15  # seconds per HTTP request


@dataclass
class _ForumThread:
    thread_id: int
    title: str
    url: str
    subforum: str
    score: float = 0.0


@dataclass
class _ForumPost:
    author: str
    timestamp: str
    content: str


def _fetch(url: str) -> str:
    """Fetch a URL and return the decoded text."""
    req = urllib.request.Request(url, headers={"User-Agent": "chemtools-nwchem/0.1"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_threads_from_listing(page_html: str, subforum_label: str) -> list[_ForumThread]:
    """Parse thread links from a subforum listing page."""
    threads: list[_ForumThread] = []
    # Thread links look like: <a href="../../../st/id3610/CCSD_Lambda_...html">title</a>
    pattern = re.compile(
        r'href="[^"]*?/st/id(\d+)/([^"]+\.html)"[^>]*>\s*(.*?)\s*</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(page_html):
        thread_id = int(match.group(1))
        slug = match.group(2)
        title = html.unescape(re.sub(r"<[^>]+>", "", match.group(3))).strip()
        if not title:
            continue
        url = f"{_FORUM_BASE}/st/id{thread_id}/{slug}"
        threads.append(_ForumThread(
            thread_id=thread_id,
            title=title,
            url=url,
            subforum=subforum_label,
        ))
    return threads


def _score_thread(thread: _ForumThread, tokens: list[str], phrase: str) -> float:
    """Score a thread title against search tokens."""
    title_lc = thread.title.casefold()
    score = 0.0
    if phrase in title_lc:
        score += 10.0
    token_hits = sum(1 for t in tokens if t in title_lc)
    if token_hits > 0:
        score += token_hits * 2.0
    return score


def _parse_thread_content(page_html: str) -> list[_ForumPost]:
    """Extract posts from a thread page. Best-effort HTML parsing."""
    posts: list[_ForumPost] = []

    # Strip scripts and style tags
    clean = re.sub(r"<script[^>]*>.*?</script>", "", page_html, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)

    # The forum uses various structures — try to find post content blocks
    # Look for post-like divs or content between clear separators
    # Common pattern: username + timestamp + content block

    # Strategy: split on common post separator patterns and extract text
    # The static archive has relatively simple HTML
    text = re.sub(r"<br\s*/?>", "\n", clean, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?tr[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?td[^>]*>", " ", text, flags=re.IGNORECASE)
    # Preserve code/pre blocks with markers
    text = re.sub(r"<pre[^>]*>", "\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</pre>", "\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<code[^>]*>", "`", text, flags=re.IGNORECASE)
    text = re.sub(r"</code>", "`", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    # Collapse excessive whitespace but preserve newlines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    # Return the whole page as one "post" — parsing individual posts
    # from the static archive structure is brittle and not worth it
    content = "\n".join(lines)

    # Trim navigation/boilerplate from start and end
    # The forum content typically starts after breadcrumb nav
    for marker in ("Forum >>", "NWChem's corner >>"):
        idx = content.find(marker)
        if idx >= 0:
            newline = content.find("\n", idx)
            if newline >= 0:
                content = content[newline + 1:]
            break

    # Trim footer
    for marker in ("Forum >>", "Moderator(s)", "Go to another board"):
        idx = content.rfind(marker)
        if idx > len(content) // 2:
            content = content[:idx].rstrip()

    if content.strip():
        posts.append(_ForumPost(author="", timestamp="", content=content.strip()))

    return posts


def search_forum(
    query: str,
    *,
    max_results: int = 5,
    max_pages_per_subforum: int = 3,
    fetch_content: bool = True,
    subforums: list[str] | None = None,
) -> dict[str, Any]:
    """Search NWChem community forum threads by keyword.

    Args:
        query: Search terms (e.g. "CCSD convergence", "DFT grid error").
        max_results: Maximum threads to return.
        max_pages_per_subforum: How many listing pages to scan per subforum (25 threads/page).
        fetch_content: If True, fetch and include the content of matching threads.
        subforums: Optional list of subforum labels to restrict search.
            Default: all subforums. Options: "Running NWChem", "NWChem functionality",
            "General Topics", "Compiling NWChem", "QM/MM".

    Returns:
        Dict with query, result_count, and results list.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty")

    tokens = [t for t in re.split(r"[^a-zA-Z0-9_+-]+", query.casefold()) if len(t) >= 2]
    phrase = query.casefold()

    # Filter subforums if requested
    forums_to_search = _SUBFORUMS
    if subforums:
        sub_lc = {s.casefold() for s in subforums}
        forums_to_search = [sf for sf in _SUBFORUMS if sf[2].casefold() in sub_lc]
        if not forums_to_search:
            forums_to_search = _SUBFORUMS

    all_threads: list[_ForumThread] = []
    errors: list[str] = []

    for sf_id, sf_slug, sf_label in forums_to_search:
        for page_idx in range(max_pages_per_subforum):
            if page_idx == 0:
                url = f"{_FORUM_BASE}/sf/id{sf_id}/{sf_slug}.html"
            else:
                offset = (page_idx * 25) - 1  # 24, 49, 74, ...
                url = f"{_FORUM_BASE}/sf/id{sf_id}/limit_{offset}%2c25/index.html"
            try:
                page_html = _fetch(url)
                threads = _extract_threads_from_listing(page_html, sf_label)
                if not threads:
                    break  # No more pages
                all_threads.extend(threads)
            except (urllib.error.URLError, OSError) as exc:
                errors.append(f"{sf_label} page {page_idx + 1}: {exc}")
                break

    # Deduplicate by thread_id
    seen: set[int] = set()
    unique: list[_ForumThread] = []
    for t in all_threads:
        if t.thread_id not in seen:
            seen.add(t.thread_id)
            unique.append(t)

    # Score and rank
    for t in unique:
        t.score = _score_thread(t, tokens, phrase)
    ranked = [t for t in unique if t.score > 0]
    ranked.sort(key=lambda t: -t.score)
    top = ranked[:max_results]

    results: list[dict[str, Any]] = []
    for thread in top:
        entry: dict[str, Any] = {
            "title": thread.title,
            "url": thread.url,
            "subforum": thread.subforum,
            "score": thread.score,
        }
        if fetch_content:
            try:
                page_html = _fetch(thread.url)
                posts = _parse_thread_content(page_html)
                if posts:
                    # Truncate very long threads
                    content = posts[0].content
                    if len(content) > 4000:
                        content = content[:4000] + "\n\n[... truncated — visit URL for full thread]"
                    entry["content"] = content
            except (urllib.error.URLError, OSError) as exc:
                entry["content_error"] = str(exc)
        results.append(entry)

    return {
        "query": query,
        "threads_scanned": len(unique),
        "result_count": len(results),
        "results": results,
        **({"errors": errors} if errors else {}),
    }
