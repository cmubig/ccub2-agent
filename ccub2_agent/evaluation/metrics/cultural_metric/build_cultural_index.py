"""Utility to convert multi-source cultural data into a retrieval index.

Supported sources:
    - PDF files (*.pdf) — original Wikipedia country PDFs
    - Text files (*.txt) — Wikipedia / Wikivoyage plain-text downloads
    - JSONL files (*.jsonl) — UNESCO ICH structured data

Steps:
    1. Parse each file in the data directory into section-aware text chunks.
    2. Clean text and attach metadata (country, section, source path, source type).
    3. Embed chunks with a SentenceTransformer model.
    4. Persist a FAISS index + metadata for later RAG queries.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

# Source authority weights (C3)
SOURCE_AUTHORITY_WEIGHTS: dict[str, float] = {
    "unesco_ich": 1.0,
    "wikipedia": 0.7,
    "wikivoyage": 0.5,
}
DEFAULT_SOURCE_AUTHORITY: float = 0.5


SECTION_TITLES = [
    "Contents hide",
    "Etymology",
    "History",
    "Geography",
    "Government and politics",
    "Economy",
    "Demographics",
    "Culture",
    "See also",
    "Notes",
    "References",
    "Sources and further reading",
    "External links",
]

SECTION_PATTERN = re.compile(r"^([A-Z][A-Za-z\s\-\(\)]+)$")

# Markdown-style header pattern for .txt files (## Title or ### Title)
MD_HEADER_PATTERN = re.compile(r"^#{2,4}\s+(.+)$", re.MULTILINE)


@dataclass
class SectionChunk:
    country: str
    section: str
    text: str
    source: Path
    source_type: str = "wikipedia"  # "wikipedia", "unesco_ich", "wikivoyage"


# ---------------------------------------------------------------------------
# PDF parsing (original behaviour, unchanged)
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def split_into_sections(raw_text: str, country: str, source: Path) -> List[SectionChunk]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    chunks: List[SectionChunk] = []
    current_title = "intro"
    buffer: List[str] = []

    def flush():
        if buffer:
            joined = " ".join(buffer)
            cleaned = re.sub(r"\s+", " ", joined).strip()
            if cleaned:
                chunks.append(SectionChunk(country=country, section=current_title, text=cleaned, source=source))
            buffer.clear()

    for line in lines:
        title_candidate = line.replace("▶", "").strip()
        if title_candidate in SECTION_TITLES or SECTION_PATTERN.match(title_candidate):
            flush()
            current_title = title_candidate.lower().replace(" ", "_")
            continue
        buffer.append(line)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# TXT parsing (Wikipedia / Wikivoyage downloads)
# ---------------------------------------------------------------------------

def _detect_source_type_from_path(path: Path) -> str:
    """Infer the source type from the parent directory name."""
    parent = path.parent.name.lower()
    if "unesco" in parent:
        return "unesco_ich"
    if "wikivoyage" in parent:
        return "wikivoyage"
    if "wikipedia" in parent:
        return "wikipedia"
    return "wikipedia"


def split_txt_into_sections(raw_text: str, country: str, source: Path) -> List[SectionChunk]:
    """Split a markdown-like .txt file into section chunks.

    Handles headers formatted as:
        ## Title
        ### Page — Section
        ---  (separators)
    """
    source_type = _detect_source_type_from_path(source)
    chunks: List[SectionChunk] = []
    current_title = "intro"
    buffer: List[str] = []

    def flush():
        if buffer:
            joined = " ".join(buffer)
            cleaned = re.sub(r"\s+", " ", joined).strip()
            if cleaned and len(cleaned) > 30:
                chunks.append(SectionChunk(
                    country=country, section=current_title,
                    text=cleaned, source=source, source_type=source_type,
                ))
            buffer.clear()

    for line in raw_text.splitlines():
        stripped = line.strip()
        # Skip separators
        if stripped == "---":
            flush()
            continue
        # Markdown header
        header_match = re.match(r"^#{2,4}\s+(.+)$", stripped)
        if header_match:
            flush()
            raw_title = header_match.group(1).strip()
            # Normalise "Page — Section" → keep section part
            if " — " in raw_title:
                raw_title = raw_title.split(" — ", 1)[1]
            current_title = raw_title.lower().replace(" ", "_")
            continue
        if stripped:
            buffer.append(stripped)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# JSONL parsing (UNESCO ICH data)
# ---------------------------------------------------------------------------

def parse_jsonl_file(jsonl_path: Path) -> List[SectionChunk]:
    """Parse a JSONL file where each line is a JSON object with at least
    'title', 'description', and optionally 'country'.
    """
    country = jsonl_path.stem.lower()
    chunks: List[SectionChunk] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at {jsonl_path}:{line_no}")
                continue

            title = record.get("title", "unknown")
            description = record.get("description", "")
            if not description or len(description) < 30:
                continue

            section = title.lower().replace(" ", "_")
            rec_country = record.get("country", country)

            chunks.append(SectionChunk(
                country=rec_country,
                section=section,
                text=description,
                source=jsonl_path,
                source_type="unesco_ich",
            ))

    return chunks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    tokens = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end == len(tokens):
            break
        start = max(end - overlap, 0)
    return chunks


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------

def iter_country_chunks(pdf_dir: Path) -> Iterable[SectionChunk]:
    """Yield chunks from PDF files in *pdf_dir* (original behaviour)."""
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        country = pdf_path.stem.lower()
        raw_text = extract_text(pdf_path)
        sections = split_into_sections(raw_text, country, pdf_path)
        for section_chunk in sections:
            for text_chunk in chunk_text(section_chunk.text):
                yield SectionChunk(
                    country=section_chunk.country,
                    section=section_chunk.section,
                    text=text_chunk,
                    source=section_chunk.source,
                )


def iter_txt_chunks(txt_dir: Path) -> Iterable[SectionChunk]:
    """Yield chunks from .txt files (Wikipedia / Wikivoyage)."""
    for txt_path in sorted(txt_dir.glob("*.txt")):
        country = txt_path.stem.lower()
        raw_text = txt_path.read_text(encoding="utf-8")
        sections = split_txt_into_sections(raw_text, country, txt_path)
        for section_chunk in sections:
            for text_chunk in chunk_text(section_chunk.text):
                yield SectionChunk(
                    country=section_chunk.country,
                    section=section_chunk.section,
                    text=text_chunk,
                    source=section_chunk.source,
                    source_type=section_chunk.source_type,
                )


def iter_jsonl_chunks(jsonl_dir: Path) -> Iterable[SectionChunk]:
    """Yield chunks from .jsonl files (UNESCO ICH)."""
    for jsonl_path in sorted(jsonl_dir.glob("*.jsonl")):
        section_chunks = parse_jsonl_file(jsonl_path)
        for section_chunk in section_chunks:
            for text_chunk in chunk_text(section_chunk.text):
                yield SectionChunk(
                    country=section_chunk.country,
                    section=section_chunk.section,
                    text=text_chunk,
                    source=section_chunk.source,
                    source_type=section_chunk.source_type,
                )


def iter_all_chunks(data_dir: Path) -> Iterable[SectionChunk]:
    """Yield chunks from all supported sources under *data_dir*.

    Expected layout:
        data_dir/
        ├── *.pdf              (legacy PDF files at top level)
        ├── wikipedia/*.txt
        ├── unesco/*.jsonl
        └── wikivoyage/*.txt
    """
    # 1. Legacy PDFs at top level
    pdf_count = 0
    for chunk in iter_country_chunks(data_dir):
        pdf_count += 1
        yield chunk
    if pdf_count:
        logger.info(f"PDF source: {pdf_count} chunks")

    # 2. Wikipedia .txt
    wiki_dir = data_dir / "wikipedia"
    if wiki_dir.is_dir():
        txt_count = 0
        for chunk in iter_txt_chunks(wiki_dir):
            txt_count += 1
            yield chunk
        logger.info(f"Wikipedia TXT source: {txt_count} chunks")

    # 3. UNESCO JSONL
    unesco_dir = data_dir / "unesco"
    if unesco_dir.is_dir():
        jsonl_count = 0
        for chunk in iter_jsonl_chunks(unesco_dir):
            jsonl_count += 1
            yield chunk
        logger.info(f"UNESCO ICH source: {jsonl_count} chunks")

    # 4. Wikivoyage .txt
    voyage_dir = data_dir / "wikivoyage"
    if voyage_dir.is_dir():
        voy_count = 0
        for chunk in iter_txt_chunks(voyage_dir):
            voy_count += 1
            yield chunk
        logger.info(f"Wikivoyage TXT source: {voy_count} chunks")


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_index(data_dir: Path, out_dir: Path, model_name: str) -> None:
    ensure_directory(out_dir)
    chunks = list(iter_all_chunks(data_dir))
    if not chunks:
        raise RuntimeError(f"No chunks found in {data_dir}")

    logger.info(f"Total chunks: {len(chunks)}")

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode([chunk.text for chunk in chunks], batch_size=32, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    metadata = [
        {
            "country": chunk.country,
            "section": chunk.section,
            "source": str(chunk.source),
            "source_type": chunk.source_type,
            "source_authority": SOURCE_AUTHORITY_WEIGHTS.get(
                chunk.source_type, DEFAULT_SOURCE_AUTHORITY
            ),
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    (out_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(meta, ensure_ascii=False) for meta in metadata),
        encoding="utf-8",
    )

    # Collect per-source statistics
    source_counts: dict[str, int] = {}
    country_counts: dict[str, int] = {}
    for m in metadata:
        st = m.get("source_type", "wikipedia")
        source_counts[st] = source_counts.get(st, 0) + 1
        country_counts[m["country"]] = country_counts.get(m["country"], 0) + 1

    (out_dir / "index_config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "embedding_dim": dimension,
                "chunk_count": len(metadata),
                "source_counts": source_counts,
                "country_counts": country_counts,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    logger.info(f"Index saved to {out_dir}: {len(metadata)} chunks, dim={dimension}")
    logger.info(f"  Sources: {source_counts}")
    logger.info(f"  Countries: {len(country_counts)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "external_data"
    default_out = Path(__file__).resolve().parent / "vector_store"

    parser = argparse.ArgumentParser(description="Build FAISS index for cultural knowledge base.")
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir,
        help="Root data directory (contains *.pdf, wikipedia/, unesco/, wikivoyage/)",
    )
    # Keep --pdf-dir as alias for backward compatibility
    parser.add_argument("--pdf-dir", type=Path, default=None, help="(deprecated) Alias for --data-dir")
    parser.add_argument("--out-dir", type=Path, default=default_out, help="Output directory for the FAISS index")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()
    data_dir = args.pdf_dir if args.pdf_dir is not None else args.data_dir
    build_index(data_dir, args.out_dir, args.model_name)


if __name__ == "__main__":
    main()
