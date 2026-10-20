import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from notebooklm import NotebookLMClient, RPCError
from notebooklm import SourceType

log = logging.getLogger(__name__)

async def get_or_create_notebook(client: NotebookLMClient, topic_name: str) -> str:
    """Find a notebook by title, or create one if it doesn't exist."""
    notebooks = await client.notebooks.list()
    for nb in notebooks:
        if nb.title.lower() == topic_name.lower():
            log.info(f"Found existing notebook: {nb.id} for topic: {topic_name}")
            return nb.id
    
    log.info(f"Creating new notebook for topic: {topic_name}")
    nb = await client.notebooks.create(topic_name)
    return nb.id

async def sync_sources(client: NotebookLMClient, notebook_id: str, papers_dir: Path) -> dict[str, str]:
    """
    Ensure all markdown/txt files in papers_dir are sources in the notebook.
    Returns a mapping of paper stem (e.g. arxiv ID) to source_id.
    """
    existing_sources = {}
    sources = await client.sources.list(notebook_id)
    for src in sources:
        if src.title:
            existing_sources[src.title] = src.id

    paper_files = list(papers_dir.glob("*.md")) + list(papers_dir.glob("*.txt"))
    added_count = 0
    
    log.info(f"Syncing {len(paper_files)} papers to notebook {notebook_id}...")
    
    for p_file in paper_files:
        title = p_file.stem
        if title in existing_sources:
            continue
            
        content = p_file.read_text(encoding="utf-8")
        if not content.strip():
            log.warning(f"File {p_file.name} is empty, skipping.")
            continue
            
        try:
            log.info(f"Adding source: {title}")
            new_src = await client.sources.add_text(notebook_id, title=title, content=content)
            existing_sources[title] = new_src.id
            added_count += 1
            await asyncio.sleep(2) # Prevent hammering the API
        except RPCError as e:
            log.error(f"Failed to add source {title}: {e}")
            
    if added_count > 0:
        log.info(f"Added {added_count} new sources to the notebook.")
    else:
        log.info("Sources already up-to-date in notebook.")
        
    return existing_sources

def parse_json_response(text: str) -> Any:
    """Extract and parse JSON from a markdown code block or raw string."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)

async def call_notebooklm(
    client: NotebookLMClient,
    notebook_id: str,
    system_prompt: str,
    user_prompt: str,
    source_ids: Optional[list[str]] = None,
    parse_json: bool = True
) -> Any:
    """
    Calls NotebookLM's Chat API. Attempts to configure persona if supported or concatenates
    the system prompt, then asks the question and parses JSON.
    """
    # NotebookLM's configure persona function relies on ChatGoal etc. which we might not have exposed exactly.
    # The safest way to handle a powerful system prompt is to prepend it to the user prompt if complex.
    full_prompt = f"System Instruction / Persona:\n{system_prompt}\n\nUser Request:\n{user_prompt}\n\n(Return ONLY valid JSON)"
    
    for attempt in range(4):
        try:
            result = await client.chat.ask(
                notebook_id,
                full_prompt,
                source_ids=source_ids
            )
            val = result.answer
            if not val or not val.strip():
                if attempt < 3:
                    log.warning(f"Empty response from NotebookLM, retrying ({attempt+1}/4)...")
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError("NotebookLM returned empty response after 4 attempts.")
            if parse_json:
                return parse_json_response(val)
            return val
            
        except RPCError as e:
            delay = 2 ** (attempt + 1)
            log.warning(f"RPC Error, retrying {attempt+1}/4 in {delay}s: {e}")
            await asyncio.sleep(delay)
        except json.JSONDecodeError as e:
            if attempt < 3:
                log.warning(f"JSON decode error, retrying: {e}")
                await asyncio.sleep(2)
            else:
                raise
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            raise
    
    raise RuntimeError("Failed to get a valid response from NotebookLM.")
