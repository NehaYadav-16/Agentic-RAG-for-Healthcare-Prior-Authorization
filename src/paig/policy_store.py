"""Loads payer policy documents and chunks them for retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PolicyChunk:
    """A retrievable unit of a policy document (one section)."""

    chunk_id: str
    doc_id: str
    payer: str
    procedure_category: str
    procedure_codes: str
    title: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id} § {self.heading}"


def _parse_front_matter(raw: str) -> tuple[Dict[str, str], str]:
    """Parses the `---` delimited metadata block at the top of a policy file."""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta_block, body = parts[1], parts[2]
    meta: Dict[str, str] = {}
    for line in meta_block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body


def _split_sections(body: str) -> List[tuple[str, str]]:
    """Splits a markdown body into (heading, text) pairs on `##` headers."""
    sections: List[tuple[str, str]] = []
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))
    return sections


def load_policies(policies_dir: str | Path) -> List[PolicyChunk]:
    """Loads every `.md` policy file in a directory into a flat list of chunks."""
    policies_dir = Path(policies_dir)
    chunks: List[PolicyChunk] = []
    for path in sorted(policies_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        doc_id = meta.get("doc_id", path.stem)
        payer = meta.get("payer", "Unknown Payer")
        category = meta.get("procedure_category", path.stem)
        codes = meta.get("procedure_codes", "")
        title = meta.get("title", path.stem)
        for heading, text in _split_sections(body):
            chunk_id = f"{doc_id}::{heading}".replace(" ", "_")
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    payer=payer,
                    procedure_category=category,
                    procedure_codes=codes,
                    title=title,
                    heading=heading,
                    text=text,
                )
            )
    return chunks


def chunks_by_category(chunks: List[PolicyChunk]) -> Dict[str, List[PolicyChunk]]:
    grouped: Dict[str, List[PolicyChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.procedure_category, []).append(chunk)
    return grouped
