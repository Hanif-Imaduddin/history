import re
import urllib.parse
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

_CHUNK_WORDS = 150
_CHUNK_OVERLAP = 20
_MAX_CHUNKS = 50
_SCRAPE_TOP_URLS = 5
_SCRAPE_TOP_K = 3

# Tool yang digunakan oleh agent untuk melakukan pencarian di internet. Tool ini dipanggil di banyak agent (nggak cuman di market scout doang)
# Di tool ini sudah embed helper untuk scraping tiap website yang didapat dari searching, jadi LLM cukup panggil tool ini sekali aja untuk pencarian.
@tool
def internet_search(query: str):

    # Deskripsi tool yang baka dibaca oleh agent.
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

        # Menggunakan BrightData untuk melakukan pencarian Google dengan query yang diberikan. Hasilnya akan dalam format JSON.
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

        # Kalau status code bukan 200, berarti ada error saat request ke BrightData, jadi kita return pesan error dengan status code dan query yang dicari.
        if response.status_code != 200:
            return f"[Search] HTTP {response.status_code} for query: '{query}'"

        try:
            # Parsing hasil pencarian dari response JSON, yang diambil cuman bagian 'organic' (hasil pencarian organik) nya saja
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
            # Step 1: Fetch + chunk semua URL secara paralel
            t_fetch = time.time()
            url_chunks: dict[str, list[str]] = {}
            with ThreadPoolExecutor(max_workers=len(urls_to_scrape)) as executor:
                fetch_futures = {executor.submit(_fetch_and_chunk, url): url for url in urls_to_scrape}
                for future in as_completed(fetch_futures):
                    url = fetch_futures[future]
                    try:
                        url_chunks[url] = future.result()
                    except Exception:
                        url_chunks[url] = []
            logging.debug(f"[internet_search] Parallel fetch+chunk took {time.time() - t_fetch:.2f}s")

            # Step 2: Kumpulkan semua chunks dari semua URL, lalu panggil _get_embeddings sekali saja
            urls_with_chunks = [(url, url_chunks[url]) for url in urls_to_scrape if url_chunks.get(url)]
            if urls_with_chunks:
                all_chunk_texts: list[str] = []
                url_chunk_ranges: dict[str, tuple[int, int]] = {}
                for url, chunks in urls_with_chunks:
                    start = len(all_chunk_texts)
                    all_chunk_texts.extend(chunks)
                    url_chunk_ranges[url] = (start, len(all_chunk_texts))

                t_embed = time.time()
                all_embeddings = _get_embeddings([query] + all_chunk_texts)
                logging.debug(f"[internet_search] Batch embed ({len(all_chunk_texts)} chunks) took {time.time() - t_embed:.2f}s")

                query_emb = np.array(all_embeddings[0])
                all_chunk_embs = np.array(all_embeddings[1:])

                # Step 3: Ranking per URL menggunakan embedding yang sudah di-batch
                parts.append("\n\n-- Page Content --")
                for url in urls_to_scrape:
                    if url not in url_chunk_ranges:
                        continue
                    start, end = url_chunk_ranges[url]
                    chunks = url_chunks[url]
                    chunk_embs = all_chunk_embs[start:end]
                    scraped = _rank_and_format(url, chunks, query_emb, chunk_embs, _SCRAPE_TOP_K)
                    if scraped:
                        parts.append(f"\n{scraped}")

        logging.debug(f"[internet_search] TOTAL took {time.time() - t0:.2f}s")
        return "\n".join(parts)

    except Exception as e:
        logging.debug(f"[internet_search] ERROR after {time.time() - t0:.2f}s: {e}")
        return f"[Search] Error for '{query}': {e}"


# Fungsi Helpers


# Fungsi untuk mengambil hasil 'organic' dari response JSON
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


def _fetch_and_chunk(url: str) -> list[str]:
    try:
        text = _fetch_page_text(url)
        if not text:
            return []
        chunks = _chunk_text(text)
        return chunks[:_MAX_CHUNKS] if chunks else []
    except Exception:
        return []


def _rank_and_format(url: str, chunks: list[str], query_emb: np.ndarray, chunk_embs: np.ndarray, top_k: int) -> str:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    c = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
    scores = c @ q

    top_indices = np.argsort(scores)[-top_k:]

    expanded = set()
    for idx in top_indices:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < len(chunks):
                expanded.add(int(neighbor))

    selected_chunks = [chunks[i] for i in sorted(expanded)]
    body = "\n---\n".join(selected_chunks)
    return f"[{url}]\n{body}"


def _fetch_page_text(url: str) -> str:
    t = time.time()
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ClarioAI/1.0)"},
    )
    logging.debug(f"[_fetch_page_text] HTTP GET took {time.time() - t:.2f}s, status={resp.status_code}, url={url}")
    resp.raise_for_status()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", resp.text)
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, step = [], _CHUNK_WORDS - _CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + _CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
    return chunks


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    t = time.time()
    logging.debug(f"[_get_embeddings] Requesting embeddings for {len(texts)} texts")
    resp = requests.post(
        f"{DEEPINFRA_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60,
    )
    logging.debug(f"[_get_embeddings] took {time.time() - t:.2f}s, status={resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def _strip_html(query: str, html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return f"Search results for '{query}':\n{text[:3000]}"
