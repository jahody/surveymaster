"""Scrape AlphaXiv blog posts for papers in a search_results.json file.

Uses the AlphaXiv MCP server (get_paper_content) when ALPHAXIV_API_KEY is set,
falling back to Playwright scraping otherwise.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

ALPHAXIV_MCP_URL = "https://api.alphaxiv.org/mcp/v1"


# ---------------------------------------------------------------------------
# AlphaXiv MCP backend
# ---------------------------------------------------------------------------

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


async def _alphaxiv_initialize(
    client: httpx.AsyncClient, url: str, api_key: str,
) -> None:
    """Perform MCP initialize handshake with AlphaXiv."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "survey-scrape", "version": "1.0"},
        },
    }
    resp = await client.post(url, json=init_payload, headers=headers, timeout=30)
    resp.raise_for_status()
    _parse_jsonrpc_response(resp)

    notif_payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    await client.post(url, json=notif_payload, headers=headers, timeout=10)


async def _alphaxiv_get_content(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    arxiv_id: str,
) -> str | None:
    """Call get_paper_content and return the report text, or None on failure."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "get_paper_content",
            "arguments": {
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "fullText": False,
            },
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        rpc_resp = _parse_jsonrpc_response(resp)
        if not rpc_resp:
            return None
        if "error" in rpc_resp:
            print(f"AlphaXiv MCP error: {rpc_resp['error']}", flush=True)
            return None
        result = rpc_resp.get("result", {})
        texts = [
            item["text"].strip()
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text", "").strip()
        ]
        if texts:
            return "\n\n".join(texts)
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            text = structured.get("text") or structured.get("content") or structured.get("report")
            if text:
                return str(text)
        return None
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"AlphaXiv MCP request failed: {e}", flush=True)
        return None


async def _fetch_blog_post_mcp(
    arxiv_id: str,
    api_key: str,
    alphaxiv_url: str = ALPHAXIV_MCP_URL,
) -> str | None:
    """Fetch blog-style paper content via AlphaXiv MCP."""
    async with httpx.AsyncClient() as client:
        try:
            await _alphaxiv_initialize(client, alphaxiv_url, api_key)
        except Exception as e:
            print(f"AlphaXiv MCP init failed: {e}", flush=True)
            return None
        return await _alphaxiv_get_content(client, alphaxiv_url, api_key, arxiv_id)


# ---------------------------------------------------------------------------
# Playwright fallback backend
# ---------------------------------------------------------------------------

async def _scrape_blog_post(paper_id: str, output_path: Path, generate_timeout: int = 300, debug: bool = False) -> Path:
    """Scrape blog content from AlphaXiv for a given paper."""
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

    output_path = Path(output_path)
    url = f"https://www.alphaxiv.org/overview/{paper_id}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(5000)

            if response is None or response.status == 404:
                raise RuntimeError(f"Page not found: {url}")
            if response.status >= 400:
                raise RuntimeError(f"Failed to load page: HTTP {response.status}")

            title_element = await page.query_selector("h1")
            title = await title_element.inner_text() if title_element else f"Paper {paper_id}"

            machine_btn = await page.query_selector(
                'button:has-text("Machine"), [role="tab"]:has-text("Machine")'
            )
            if machine_btn:
                await machine_btn.click()
                await page.wait_for_timeout(3000)

            if debug:
                debug_dir = output_path.parent / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(debug_dir / f"{paper_id}.png"), full_page=True)

            # Check if blog needs to be generated
            await _maybe_trigger_generation(page, paper_id, timeout=generate_timeout)

            markdown_content = await page.evaluate("""() => {
                const candidates = document.querySelectorAll('div, pre, code, section');
                let best = null;
                let bestLen = 0;

                function looksLikeBlog(text) {
                    if (text.length < 200) return false;
                    if (text.match(/^#+\\s/m)) return true;
                    if (text.match(/^\\d+\\.\\s+\\w/m) && text.length > 500) return true;
                    if (text.match(/Research\\s+(Paper\\s+)?Report/i)) return true;
                    return false;
                }

                for (const el of candidates) {
                    const text = el.innerText || '';
                    if (!looksLikeBlog(text) || text.length <= bestLen) continue;

                    let childMatches = false;
                    for (const child of el.children) {
                        const ct = child.innerText || '';
                        if (looksLikeBlog(ct)) {
                            childMatches = true;
                            break;
                        }
                    }
                    if (!childMatches) {
                        best = el;
                        bestLen = text.length;
                    }
                }
                return best ? best.innerText : null;
            }""")

            if debug:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"{paper_id}.txt").write_text(
                    markdown_content or "(empty)", encoding="utf-8"
                )

            if not markdown_content or not markdown_content.strip():
                raise RuntimeError(f"No blog content found for paper {paper_id}")

            final_markdown = f"# {title}\n\n> Source: {url}\n\n{markdown_content}"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_markdown, encoding="utf-8")

            return output_path

        except PlaywrightTimeout as e:
            raise RuntimeError(f"Timeout loading page {url}: {e}") from e

        finally:
            await browser.close()


async def _maybe_trigger_generation(page, paper_id: str, timeout: int = 120) -> None:
    """If blog hasn't been generated yet, click the Generate button and wait."""
    from playwright.async_api import TimeoutError as PlaywrightTimeout

    generate_btn = None
    for selector in [
        'button:has-text("Generate")',
        'button:has-text("Create")',
        'button:has-text("Try again")',
        'button:has-text("Retry")',
        'button:has-text("generate")',
        'button:has-text("create")',
        'a:has-text("Generate")',
        'a:has-text("Create")',
        'a:has-text("Try again")',
    ]:
        generate_btn = await page.query_selector(selector)
        if generate_btn:
            break

    if generate_btn:
        if not await generate_btn.is_visible():
            generate_btn = None
        else:
            print(f"triggering generation for {paper_id}...", end=" ", flush=True)
            await generate_btn.click(timeout=10000)
            await page.wait_for_timeout(3000)

    is_loading = await page.evaluate("""() => {
        const spinners = document.querySelectorAll(
            '[class*="spinner"], [class*="loading"], [class*="animate-spin"], svg.animate-spin'
        );
        for (const s of spinners) {
            if (s.offsetParent !== null) return true;
        }
        const body = document.body.innerText;
        if (body.includes('Transform this paper into a blog')) return true;
        return false;
    }""")

    if not generate_btn and not is_loading:
        return

    if is_loading:
        print(f"waiting for generation ({paper_id})...", end=" ", flush=True)

    poll_interval = 5
    elapsed = 0
    while elapsed < timeout:
        await page.wait_for_timeout(poll_interval * 1000)
        elapsed += poll_interval

        has_content = await page.evaluate("""() => {
            const els = document.querySelectorAll('div, pre, section');
            for (const el of els) {
                const t = el.innerText || '';
                if (t.length > 200 && (t.match(/^#+\\s/m) || t.match(/Research\\s+(Paper\\s+)?Report/i)))
                    return true;
            }
            return false;
        }""")
        if has_content:
            print(f"generated ({elapsed}s)", end=" ", flush=True)
            machine_btn = await page.query_selector(
                'button:has-text("Machine"), [role="tab"]:has-text("Machine")'
            )
            if machine_btn and await machine_btn.is_visible():
                try:
                    await machine_btn.click(timeout=10000)
                    await page.wait_for_timeout(3000)
                except PlaywrightTimeout:
                    pass
            return

        failed = await page.query_selector('button:has-text("Try again")')
        if failed and generate_btn and await failed.is_visible():
            print("generation failed, retrying...", end=" ", flush=True)
            try:
                await failed.click(timeout=10000)
            except PlaywrightTimeout:
                pass
            await page.wait_for_timeout(3000)

    print(f"generation timed out ({timeout}s)", end=" ", flush=True)


# ---------------------------------------------------------------------------
# Main batch scraper
# ---------------------------------------------------------------------------

async def scrape_blogs(search_results_path: Path, output_dir: Path, delay: int = 3, debug: bool = False) -> None:
    """Batch-scrape AlphaXiv blog posts for papers in a search_results.json file."""
    api_key = os.getenv("ALPHAXIV_API_KEY", "")
    use_mcp = bool(api_key)

    if use_mcp:
        print("ALPHAXIV_API_KEY set — using AlphaXiv MCP backend")
    else:
        print("ALPHAXIV_API_KEY not set — using Playwright scraper")

    output_dir.mkdir(parents=True, exist_ok=True)
    papers = json.loads(search_results_path.read_text(encoding="utf-8"))
    existing = {p.stem for p in output_dir.glob("*.md")}

    to_scrape: list[str] = []
    no_arxiv = 0
    for paper in papers:
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        if not arxiv_id:
            no_arxiv += 1
            continue
        if arxiv_id not in existing and arxiv_id not in to_scrape:
            to_scrape.append(arxiv_id)

    print(f"Found {len(papers)} papers, {len(to_scrape)} to scrape, "
          f"{len(existing)} already exist, {no_arxiv} without arXiv ID")

    if not to_scrape:
        return

    scraped = 0
    failed = 0
    no_blog = 0

    for i, paper_id in enumerate(to_scrape):
        output_path = output_dir / f"{paper_id}.md"
        print(f"  [{i + 1}/{len(to_scrape)}] {paper_id} ...", end=" ", flush=True)

        ok = False

        if use_mcp:
            content = await _fetch_blog_post_mcp(paper_id, api_key)
            if content:
                source_url = f"https://www.alphaxiv.org/overview/{paper_id}"
                output_path.write_text(
                    f"# {paper_id}\n\n> Source: {source_url}\n\n{content}",
                    encoding="utf-8",
                )
                print("OK (MCP)")
                scraped += 1
                ok = True
            else:
                print("MCP returned no content, falling back to Playwright...", end=" ", flush=True)

        if not ok:
            for attempt in range(3):
                try:
                    await _scrape_blog_post(paper_id, output_path, debug=debug)
                    print("OK")
                    scraped += 1
                    ok = True
                    break
                except RuntimeError as e:
                    if "No blog content" in str(e):
                        print("no blog")
                        no_blog += 1
                        break
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        print(f"retry ({e})", flush=True)
                        await asyncio.sleep(wait)
                    else:
                        print(f"FAILED ({e})")
                        failed += 1
                except Exception as e:
                    print(f"ERROR ({e})")
                    failed += 1
                    break

        if i < len(to_scrape) - 1:
            await asyncio.sleep(delay)

    print(f"\nDone: {scraped} scraped, {no_blog} no blog, {failed} failed")


def main():
    # Default to the latest topic directory
    base = Path("papers")
    dirs = sorted(base.iterdir()) if base.exists() else []
    topic_dirs = [d for d in dirs if d.is_dir() and (d / "search_results.json").exists()]

    if not topic_dirs:
        print("No topic directories with search_results.json found under papers/")
        sys.exit(1)

    # Pick topic dir: first CLI arg, or list choices
    if len(sys.argv) > 1:
        chosen = base / sys.argv[1]
    elif len(topic_dirs) == 1:
        chosen = topic_dirs[0]
    else:
        print("Available topics:")
        for i, d in enumerate(topic_dirs):
            print(f"  [{i}] {d.name}")
        idx = input("Pick a number: ").strip()
        chosen = topic_dirs[int(idx)]

    results_file = chosen / "search_results.json"
    blog_dir = chosen / "blogPosts"

    print(f"Topic dir: {chosen}")
    print(f"Output:    {blog_dir}")
    print()

    asyncio.run(scrape_blogs(results_file, blog_dir))


if __name__ == "__main__":
    main()
