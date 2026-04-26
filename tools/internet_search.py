import re
import urllib.parse
import json

import numpy as np
import requests
from langchain_core.tools import tool
import os

from dotenv import load_dotenv

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
    try:
        # Menggunakan BrightData untuk melakukan pencarian Google dengan query yang diberikan. Hasilnya akan dalam format JSON.
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

        # Kalau status code bukan 200, berarti ada error saat request ke BrightData, jadi kita return pesan error dengan status code dan query yang dicari.
        if response.status_code != 200:
            return f"[Search] HTTP {response.status_code} for query: '{query}'"

        try:
            # Parsing hasil pencarian dari response JSON, yang diambil cuman bagian 'organic' (hasil pencarian organik) nya saja
            search_results = _parse_json_results(response.json())
        except Exception:
            return _strip_html(query, response.text)

        parts = [f"Search results for '{query}':\n"]

        for i, item in enumerate(search_results, 1):
            parts.append(f"\n{i}. {item['title']}")
            if item["link"]:
                parts.append(f"   {item['link']}")
            if item["snippet"]:
                parts.append(f"   {item['snippet']}")

        # Ini bagian untuk scraping konten dari URL yang didapat dari hasil pencarian. Kita ambil maksimal 5 URL teratas.
        # Tiap URL diambil 3 chunk yang paling relevan (Berdasarkan cosine similarity embedding dengan query)
        # Selain diambil 3 chunk, diambil juga chunk sebelum dan sesudah dari masing-masing chunk yang relevan, jadi totalnya bisa sampai 9 chunk per URL.
        urls_to_scrape = [r["link"] for r in search_results[:_SCRAPE_TOP_URLS] if r["link"]]
        if urls_to_scrape:
            parts.append("\n\n-- Page Content --")
            for url in urls_to_scrape:
                scraped = _scrape_relevant_content(url, query, top_k=_SCRAPE_TOP_K)
                if scraped:
                    parts.append(f"\n{scraped}")
        return "\n".join(parts)

    except Exception as e:
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


def _scrape_relevant_content(url: str, query: str, top_k: int) -> str:
    try:
        text = _fetch_page_text(url)
        if not text:
            return ""

        chunks = _chunk_text(text)
        if not chunks:
            return ""

        chunks = chunks[:_MAX_CHUNKS]

        all_embeddings = _get_embeddings([query] + chunks)
        query_emb = np.array(all_embeddings[0])
        chunk_embs = np.array(all_embeddings[1:])

        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        c = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
        scores = c @ q

        top_indices = np.argsort(scores)[-top_k:]

        expanded = set()
        for idx in top_indices:
            for neighbor in (idx - 1, idx, idx + 1):
                if 0 <= neighbor < len(chunks):
                    expanded.add(neighbor)

        selected_chunks = [chunks[i] for i in sorted(expanded)]
        body = "\n---\n".join(selected_chunks)
        return f"[{url}]\n{body}"

    except Exception:
        return ""

def _fetch_page_text(url: str) -> str:
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ClarioAI/1.0)"},
    )
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
    resp = requests.post(
        f"{DEEPINFRA_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

def _strip_html(query: str, html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return f"Search results for '{query}':\n{text[:3000]}"
