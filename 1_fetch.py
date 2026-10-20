"""Step 1: Fetch papers.

Searches via Asta MCP or Semantic Scholar, saves abstracts.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

import httpx

import _resolvers  # noqa: F401 – registers OmegaConf resolvers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV2MD_API = "https://arxiv2md.org/api/markdown"


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    if len(text) > max_len:
        text = text[:max_len].rsplit("-", 1)[0]
    return text


def extract_arxiv_id(paper: dict) -> str | None:
    """Extract arXiv ID from paper's externalIds or URL."""
    ext_ids = paper.get("externalIds") or {}
    if "ArXiv" in ext_ids:
        return ext_ids["ArXiv"]
    url = paper.get("url") or ""
    match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", url)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Asta MCP backend
# ---------------------------------------------------------------------------

async def asta_initialize(
    client: httpx.AsyncClient, url: str, api_key: str,
) -> dict | None:
    """Perform MCP initialize handshake."""
    init_payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "survey4-fetch", "version": "1.0"},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key,
    }

    resp = await client.post(url, json=init_payload, headers=headers, timeout=30)
    resp.raise_for_status()
    result = _parse_jsonrpc_response(resp)

    notif_payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    await client.post(url, json=notif_payload, headers=headers, timeout=10)
    return result


def _parse_jsonrpc_response(resp: httpx.Response) -> dict | None:
    """Parse a JSON-RPC response, handling plain JSON and SSE/JSONL."""
    content_type = resp.headers.get("content-type", "")

    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str:
                    try:
                        return json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
        return None

    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        pass

    for line in resp.text.splitlines():
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _extract_papers_from_mcp_result(result: dict) -> list[dict]:
    """Extract paper list from MCP tool call result."""
    if isinstance(result, list):
        return result

    # Prefer structuredContent (typed output) over content (text fallback)
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        for key in ("result", "data", "papers", "results"):
            if key in structured and isinstance(structured[key], list):
                return structured[key]
    if isinstance(structured, list):
        return structured

    content_list = result.get("content", [])
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    for key in ("data", "papers", "results", "result"):
                        if key in parsed and isinstance(parsed[key], list):
                            return parsed[key]
                    return [parsed]
            except json.JSONDecodeError:
                continue

    for key in ("data", "papers", "results", "result"):
        if key in result and isinstance(result[key], list):
            return result[key]

    log.warning("Could not extract papers from Asta result: %s", str(result)[:200])
    return []


async def asta_search(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    query: str,
    fields: str,
    limit: int,
) -> list[dict]:
    """Search via Asta MCP (JSON-RPC 2.0)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_papers_by_relevance",
            "arguments": {
                "keyword": query,
                "fields": fields,
                "limit": limit,
            },
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key,
    }

    for attempt in range(3):
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            rpc_resp = _parse_jsonrpc_response(resp)
            if rpc_resp is None:
                log.error("Empty response from Asta")
                return []
            if "error" in rpc_resp:
                log.error("Asta RPC error: %s", rpc_resp["error"])
                return []
            result = rpc_resp.get("result", {})
            log.info("DEBUG raw rpc_resp keys: %s", list(rpc_resp.keys()))
            log.info("DEBUG result type: %s, keys: %s", type(result).__name__, list(result.keys()) if isinstance(result, dict) else "N/A")
            log.info("DEBUG result (first 2000 chars): %s", str(result)[:2000])
            return _extract_papers_from_mcp_result(result)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            delay = 2 ** (attempt + 1)
            log.warning("Asta error, retry %d/3 (waiting %ds): %s", attempt + 1, delay, e)
            await asyncio.sleep(delay)

    log.error("Failed to search Asta after 3 attempts")
    return []


# ---------------------------------------------------------------------------
# Semantic Scholar public API backend (no key required)
# ---------------------------------------------------------------------------

async def s2_search(
    client: httpx.AsyncClient,
    query: str,
    fields: str,
    limit: int,
) -> list[dict]:
    """Search via Semantic Scholar public REST API (no API key needed)."""
    all_papers: list[dict] = []
    offset = 0
    page_size = 100
    max_offset = limit * 3  # safety cap
    headers = {"User-Agent": "survey4/1.0 (research-paper-classification)"}

    while offset < max_offset:
        params = {
            "query": query,
            "fields": fields,
            "limit": page_size,
            "offset": offset,
        }

        for attempt in range(5):
            try:
                resp = await client.get(
                    S2_API_URL, params=params, headers=headers, timeout=30,
                )
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = int(retry_after) if retry_after else 10 * (attempt + 1)
                    log.warning("S2 rate limited, retry %d/5 (waiting %ds)", attempt + 1, delay)
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("data", [])
                all_papers.extend(batch)
                if len(batch) < page_size:
                    return all_papers
                break
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                delay = 5 * (attempt + 1)
                log.warning("S2 error, retry %d/5 (waiting %ds): %s", attempt + 1, delay, e)
                await asyncio.sleep(delay)
        else:
            log.error("Failed to fetch S2 page at offset %d after 5 attempts", offset)
            break

        arxiv_count = sum(1 for p in all_papers if extract_arxiv_id(p))
        if arxiv_count >= limit:
            break

        offset += page_size

    return all_papers


# ---------------------------------------------------------------------------
# Full paper retrieval (arxiv2md.org API -> arXiv HTML fallback)
# ---------------------------------------------------------------------------

async def _fetch_paper_arxiv2md(
    client: httpx.AsyncClient, arxiv_id: str,
) -> str | None:
    """Fetch full paper markdown via arxiv2md.org API."""
    try:
        resp = await client.get(
            ARXIV2MD_API,
            params={"url": f"https://arxiv.org/abs/{arxiv_id}"},
            timeout=60,
        )
        if resp.status_code == 200 and len(resp.text) > 100:
            return resp.text
        log.warning("arxiv2md returned %d for %s", resp.status_code, arxiv_id)
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        log.warning("arxiv2md failed for %s: %s", arxiv_id, e)
    return None


async def _fetch_paper_arxiv_html(
    client: httpx.AsyncClient, arxiv_id: str,
) -> str | None:
    """Fallback: fetch arXiv HTML and convert with markdownify."""
    try:
        from markdownify import markdownify as md
    except ImportError:
        log.warning("markdownify not installed, cannot use HTML fallback")
        return None

    try:
        resp = await client.get(
            f"https://arxiv.org/html/{arxiv_id}v1",
            follow_redirects=True,
            timeout=60,
        )
        if resp.status_code != 200:
            return None
        match = re.search(r"<article[^>]*>(.*?)</article>", resp.text, re.DOTALL)
        if match:
            return md(match.group(1), strip=["img", "figure"])
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        log.warning("arXiv HTML failed for %s: %s", arxiv_id, e)
    return None


async def fetch_full_paper(
    client: httpx.AsyncClient,
    arxiv_id: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Fetch full paper markdown. Returns True if saved."""
    async with semaphore:
        markdown = await _fetch_paper_arxiv2md(client, arxiv_id)
        if not markdown:
            markdown = await _fetch_paper_arxiv_html(client, arxiv_id)
        if not markdown:
            log.warning("Could not fetch full paper for %s", arxiv_id)
            return False
        output_path.write_text(markdown, encoding="utf-8")
        return True


async def fetch_full_papers(
    arxiv_ids: list[str],
    papers_dir: Path,
    delay: float = 2.0,
    max_concurrent: int = 3,
) -> tuple[int, int, int]:
    """Fetch full papers for a list of arXiv IDs. Returns (new, skipped, failed)."""
    papers_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in papers_dir.glob("*.md")}

    to_fetch = [aid for aid in arxiv_ids if aid not in existing]
    skipped = len(arxiv_ids) - len(to_fetch)

    if not to_fetch:
        return 0, skipped, 0

    semaphore = asyncio.Semaphore(max_concurrent)
    new_count = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, arxiv_id in enumerate(to_fetch):
            output_path = papers_dir / f"{arxiv_id}.md"
            ok = await fetch_full_paper(client, arxiv_id, output_path, semaphore)
            if ok:
                new_count += 1
                log.info("OK %s (%d/%d)", arxiv_id, i + 1, len(to_fetch))
            else:
                failed += 1
                log.warning("FAIL %s (%d/%d)", arxiv_id, i + 1, len(to_fetch))
            if i < len(to_fetch) - 1:
                await asyncio.sleep(delay)

    return new_count, skipped, failed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _save_query_results(
    papers: list[dict],
    query_dir: Path,
    abstracts_dir: Path,
) -> tuple[int, int, int]:
    """Save search results and extract abstracts. Returns (new, skipped, no_arxiv)."""
    results_path = query_dir / "search_results.json"
    results_path.write_text(
        json.dumps(papers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    existing = {p.stem for p in abstracts_dir.glob("*.md")}
    seen_ids: set[str] = set()
    new_count = 0
    skipped = 0
    no_arxiv = 0

    for paper in papers:
        arxiv_id = extract_arxiv_id(paper)
        if not arxiv_id:
            no_arxiv += 1
            log.debug("Skipping non-arXiv paper: %s", paper.get("title", "unknown"))
            continue

        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        if arxiv_id in existing:
            skipped += 1
            continue

        abstract = paper.get("abstract", "")
        if not abstract:
            log.warning("Empty abstract for %s", arxiv_id)

        abstract_path = abstracts_dir / f"{arxiv_id}.md"
        abstract_path.write_text(f"\n{abstract.strip()}\n", encoding="utf-8")
        new_count += 1

    return new_count, skipped, no_arxiv


async def run_fetch(cfg: DictConfig) -> None:
    """Search for papers via Asta or Semantic Scholar."""
    asta = cfg.asta
    base_output = Path(to_absolute_path(asta.output_dir))
    query = asta.search_query
    use_asta = bool(asta.api_key)

    if use_asta:
        log.info("ASTA_API_KEY is set — using Asta MCP backend")
    else:
        log.info("ASTA_API_KEY not set — falling back to Semantic Scholar public API (no key needed)")

    async with httpx.AsyncClient() as client:
        if use_asta:
            log.info("Initializing MCP session with %s", asta.url)
            try:
                server_info = await asta_initialize(client, asta.url, asta.api_key)
                if server_info:
                    log.info("MCP session initialized")
                else:
                    log.warning("MCP init returned empty response, proceeding anyway")
            except Exception as e:
                log.warning("Asta MCP init failed: %s — falling back to public API", e)
                use_asta = False

        slug = slugify(query)
        query_dir = base_output / slug
        abstracts_dir = query_dir / "abstracts"
        abstracts_dir.mkdir(parents=True, exist_ok=True)

        log.info("Query: %s", query)
        log.info("Output dir: %s", query_dir)
        backend = "Asta" if use_asta else "Semantic Scholar"
        log.info("Backend: %s", backend)

        if use_asta:
            papers = await asta_search(
                client, asta.url, asta.api_key,
                query, asta.fields, asta.limit * 2,
            )
        else:
            papers = await s2_search(
                client, query, asta.fields, asta.limit,
            )

        log.info("%s returned %d papers (pre-filter)", backend, len(papers))

        papers = [p for p in papers if extract_arxiv_id(p)][:asta.limit]
        log.info("After arXiv filter: %d papers", len(papers))

        if not papers:
            print(f"\n[{slug}] No papers found")
            return

        new_count, skipped, no_arxiv = _save_query_results(
            papers, query_dir, abstracts_dir,
        )

        arxiv_count = sum(1 for p in papers if extract_arxiv_id(p))
        print(f"\n[{slug}] Found {len(papers)} papers, "
              f"{arxiv_count} with arXiv IDs, "
              f"{new_count} new, {skipped} skipped")

        # Fetch full papers if configured
        if asta.get("fetch_full_papers", False):
            arxiv_ids = [
                extract_arxiv_id(p) for p in papers if extract_arxiv_id(p)
            ]
            markdown_dir = query_dir / "papers_markdown"
            print(f"  Fetching full papers ({len(arxiv_ids)} papers)...")
            fp_new, fp_skip, fp_fail = await fetch_full_papers(
                arxiv_ids, markdown_dir,
                delay=asta.get("full_paper_delay", 2.0),
                max_concurrent=asta.get("full_paper_concurrent", 3),
            )
            print(f"  Full papers: {fp_new} new, {fp_skip} skipped, {fp_fail} failed")

    print(f"\nTotal: {new_count} new papers saved, "
          f"{skipped} skipped (already exist), "
          f"{no_arxiv} skipped (no arXiv ID)")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(run_fetch(cfg))


if __name__ == "__main__":
    main()
