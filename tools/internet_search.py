import re
import urllib.parse
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from langchain_core.tools import tool
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

load_dotenv()

BRIGHTDATA_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
BRIGHTDATA_ZONE = os.getenv("BRIGHTDATA_ZONE")

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL")

_SCRAPE_TOP_URLS = 5
_SUMMARIZE_MODEL = "google/gemma-3-4b-it"
_MAX_PAGE_CHARS = 12000
_MAX_DOWNLOAD_BYTES = 1 * 1024 * 1024  # 1 MB cap untuk fetch halaman
_BINARY_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".gz", ".mp4", ".mp3", ".png", ".jpg", ".jpeg"}
_BINARY_CONTENT_TYPES = ("application/pdf", "application/octet-stream", "application/zip", "application/msword", "image/", "audio/", "video/")

# Tool yang digunakan oleh agent untuk melakukan pencarian di internet.
@tool
def internet_search(query: str):
    """Search the internet for real-time business and market information.
    Use this for market trends, competitor landscape, regulatory info, pricing data,
    and any other facts needed for business planning.

    Args:
        query: The search query string

    Returns:
        Formatted search results with titles, URLs, snippets, and relevant page content
    """
    t0 = time.time()
    try:
        logging.debug(f"[internet_search] START query='{query}'")

        t_search = time.time()
        response = requests.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "zone": BRIGHTDATA_ZONE,
                "url": f"https://www.google.com/search?q={urllib.parse.quote(query)}&num=5",
                "format": "json",
            },
            timeout=30,
        )
        logging.debug(f"[internet_search] BrightData search took {time.time() - t_search:.2f}s, status={response.status_code}")

        if response.status_code != 200:
            return f"[Search] HTTP {response.status_code} for query: '{query}'"

        try:
            search_results = _parse_json_results(response.json())
        except Exception:
            return _strip_html(query, response.text)

        logging.debug(f"[internet_search] Parsed {len(search_results)} results")

        parts = [f"Search results for '{query}':\n"]

        for i, item in enumerate(search_results, 1):
            parts.append(f"\n{i}. {item['title']}")
            if item["link"]:
                parts.append(f"   {item['link']}")
            if item["snippet"]:
                parts.append(f"   {item['snippet']}")

        urls_to_scrape = [r["link"] for r in search_results[:_SCRAPE_TOP_URLS] if r["link"]]
        if urls_to_scrape:
            t_summarize = time.time()
            summaries: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=len(urls_to_scrape)) as executor:
                futures = {executor.submit(_fetch_and_summarize, url, query): url for url in urls_to_scrape}
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        summaries[url] = future.result()
                    except Exception:
                        summaries[url] = ""
            logging.debug(f"[internet_search] Parallel fetch+summarize took {time.time() - t_summarize:.2f}s")

            parts.append("\n\n-- Page Summaries --")
            for url in urls_to_scrape:
                summary = summaries.get(url, "")
                if summary:
                    parts.append(f"\n[{url}]\n{summary}")

        logging.debug(f"[internet_search] TOTAL took {time.time() - t0:.2f}s")
        return "\n".join(parts)

    except Exception as e:
        logging.debug(f"[internet_search] ERROR after {time.time() - t0:.2f}s: {e}")
        return f"[Search] Error for '{query}': {e}"


# Fungsi Helpers


def _parse_json_results(data: dict) -> list[dict]:
    json_body = json.loads(data["body"])
    results = []
    for item in json_body.get("organic", [])[:5]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        if title or snippet:
            results.append({"title": title, "link": link, "snippet": snippet})
    return results


def _fetch_and_summarize(url: str, query: str) -> str:
    try:
        # Skip binary file URLs berdasarkan ekstensi sebelum membuat HTTP request
        path = urllib.parse.urlparse(url).path.lower()
        if any(path.endswith(ext) for ext in _BINARY_EXTENSIONS):
            logging.debug(f"[_fetch_page_text] Skipping binary URL: {url}")
            return ""
        text = _fetch_page_text(url)
        if not text:
            return ""
        return _summarize_page(text, query)
    except Exception:
        return ""


def _fetch_page_text(url: str) -> str:
    t = time.time()
    resp = requests.get(
        url,
        timeout=(10, 15),  # (connect timeout, read timeout per chunk)
        headers={"User-Agent": "Mozilla/5.0 (compatible; ClarioAI/1.0)"},
        stream=True,  # ambil header dulu sebelum download body
    )
    logging.debug(f"[_fetch_page_text] HTTP GET took {time.time() - t:.2f}s, status={resp.status_code}, url={url}")
    resp.raise_for_status()

    # Skip berdasarkan Content-Type header
    content_type = resp.headers.get("Content-Type", "").lower()
    if any(ct in content_type for ct in _BINARY_CONTENT_TYPES):
        resp.close()
        logging.debug(f"[_fetch_page_text] Skipping binary content-type '{content_type}': {url}")
        return ""

    # Download dengan batas ukuran 1 MB
    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        total += len(chunk)
        chunks.append(chunk)
        if total >= _MAX_DOWNLOAD_BYTES:
            logging.debug(f"[_fetch_page_text] Size cap hit ({total} bytes), stopping download: {url}")
            break
    resp.close()

    # Decode eksplisit agar tidak bergantung chardet (penyebab "Unable to determine charset")
    raw = b"".join(chunks)
    text = raw.decode("utf-8", errors="ignore") or raw.decode("latin-1", errors="ignore")

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _summarize_page(text: str, query: str) -> str:
    t = time.time()
    truncated = text[:_MAX_PAGE_CHARS]
    prompt = (
        f'Summarize the following webpage content in relation to the search query: "{query}"\n\n'
        f"Focus on the most relevant facts, data, and insights. Be concise.\n\n"
        f"Webpage content:\n{truncated}\n\nSummary:"
    )
    resp = requests.post(
        f"{DEEPINFRA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _SUMMARIZE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.2,
        },
        timeout=60,
    )
    logging.debug(f"[_summarize_page] Summarization took {time.time() - t:.2f}s, status={resp.status_code}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _strip_html(query: str, html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return f"Search results for '{query}':\n{text[:3000]}"
