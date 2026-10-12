import random
import time
from typing import Optional

import html2text
import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError as exc:
    DDGS = None
    DDGS_IMPORT_ERROR = exc
else:
    DDGS_IMPORT_ERROR = None


def ddg_search_engine_query(
    keywords: str,
    max_results: Optional[int] = 10,
    region: Optional[str] = "wt-wt",
    show_snippet: bool = True,
) -> list | dict:
    if DDGS is None:
        return {
            "error": (
                f"ddgs library not installed ({DDGS_IMPORT_ERROR}). "
                "Install with: pip install ddgs"
            )
        }

    try:
        max_results = int(max_results) if max_results is not None else 10
    except (TypeError, ValueError):
        max_results = 10

    backoff = 2
    while True:
        try:
            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(
                        keywords,
                        region=region or "wt-wt",
                        max_results=max_results,
                    )
                )
            break
        except Exception as exc:
            error = str(exc)
            if "429" in error or "ratelimit" in error.lower() or "rate" in error.lower():
                wait_time = backoff + random.uniform(0, backoff)
                print(
                    "*" * 100
                    + f"\n[WebSearchAPI/DDG] Rate-limited by DuckDuckGo. "
                    f"Retrying in {wait_time:.1f}s.\n"
                    + "*" * 100
                )
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue

            print(
                "*" * 100
                + f"\n[WebSearchAPI/DDG] Error from DuckDuckGo: {error}. "
                "This is not a rate-limit error, so it will not be retried.\n"
                + "*" * 100
            )
            return {"error": error}

    results = []
    for result in raw_results[:max_results]:
        entry = {
            "title": result.get("title", ""),
            "href": result.get("href", ""),
        }
        if show_snippet:
            entry["body"] = result.get("body", "")
        results.append(entry)
    return results


FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/112.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.google.com/",
}


def ddg_fetch_url_content(url: str, mode: str = "raw") -> dict:
    if not url.startswith(("http://", "https://")):
        return {"error": f"Invalid URL: {url}"}

    try:
        response = requests.get(
            url, headers=FETCH_HEADERS, timeout=20, allow_redirects=True
        )
        response.raise_for_status()
    except Exception as exc:
        return {"error": f"An error occurred while fetching {url}: {exc}"}

    try:
        if mode == "raw":
            return {"content": response.text}

        if mode == "markdown":
            converter = html2text.HTML2Text()
            return {"content": converter.handle(response.text)}

        if mode == "truncate":
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.extract()
            return {"content": soup.get_text(separator="\n", strip=True)}

        return {"error": f"Unsupported mode: {mode}"}
    except Exception as exc:
        return {"error": f"Failed to process {url} in mode={mode}: {exc}"}
