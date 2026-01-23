"""Utility to convert country-level Wikipedia PDFs into a retrieval index.

Steps:
    1. Parse each PDF in the source directory into section-aware text chunks.
    2. Clean text and attach metadata (country, section, source path).
    3. Embed chunks with a SentenceTransformer model.
    4. Persist a FAISS index + metadata for later RAG queries.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


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


@dataclass
class SectionChunk:
    country: str
    section: str
    text: str
    source: Path


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


def iter_country_chunks(pdf_dir: Path) -> Iterable[SectionChunk]:
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


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_index(pdf_dir: Path, out_dir: Path, model_name: str) -> None:
    ensure_directory(out_dir)
    chunks = list(iter_country_chunks(pdf_dir))
    if not chunks:
        raise RuntimeError(f"No PDF chunks found in {pdf_dir}")

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
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    (out_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(meta, ensure_ascii=False) for meta in metadata),
        encoding="utf-8",
    )

    (out_dir / "index_config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "embedding_dim": dimension,
                "chunk_count": len(metadata),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_pdf_dir = repo_root / "external_data"
    default_out = Path(__file__).resolve().parent / "vector_store"

    parser = argparse.ArgumentParser(description="Build FAISS index for cultural knowledge base.")
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir, help="Directory with country PDFs")
    parser.add_argument("--out-dir", type=Path, default=default_out, help="Output directory for the FAISS index")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args.pdf_dir, args.out_dir, args.model_name)


if __name__ == "__main__":
    main()
