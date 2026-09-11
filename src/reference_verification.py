"""Reference-verification mode via pgvector (Phase 7 — stretch).

Given a suspect clip and a known-authentic reference clip, compares their
ArcFace identity embeddings with a vector-similarity search to surface
identity-impersonation risk.  Dependencies (``psycopg2-binary``, a running
Postgres with the pgvector extension) are optional and only needed when this
mode is exercised.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import load_config

logger = logging.getLogger("reference_verification")


def embed_clip_identity(clip_path: str | Path, output_dir: str | Path, ctx_id: int = 0) -> np.ndarray:
    """Mean-pool ArcFace embeddings of a clip into a single identity vector.

    Args:
        clip_path: Path to the clip to embed.
        output_dir: Temporary directory for the embedding cache.
        ctx_id: Detector device index.

    Returns:
        Unit-normalised ``(512,)`` identity embedding.

    Raises:
        RuntimeError: If the clip produced no face embeddings.
    """
    from src.extraction.extract_video import process_video

    clip_path = Path(clip_path)
    out_file = process_video(clip_path, clip_path.stem, Path(output_dir), ctx_id=ctx_id)
    payload = torch.load(out_file, map_location="cpu")
    embeddings = payload["embeddings"].numpy()
    if embeddings.shape[0] == 0:
        raise RuntimeError(f"No faces found in {clip_path}; cannot build identity vector.")
    vector = embeddings.mean(axis=0)
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-12)


class ReferenceStore:
    """Thin pgvector wrapper for identity embeddings.

    Args:
        url: Postgres connection URL (defaults to config). Must point to a
            database with the pgvector extension and a ``reference_embeddings``
            table (created lazily if missing).

    Raises:
        ImportError: If ``psycopg2`` is not installed.
    """

    def __init__(self, url: str | None = None) -> None:
        try:
            import psycopg2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for reference-verification mode. "
                "Install with: pip install psycopg2-binary"
            ) from exc
        import pgvector.psycopg2  # noqa: F401

        config = load_config()
        self.url = url or config["reference_verification"]["pgvector_url"]
        self._connection: Any = None

    def connect(self) -> Any:
        """Return and memoise a psycopg2 connection."""
        if self._connection is None or self._connection.closed:
            import psycopg2

            self._connection = psycopg2.connect(self.url)
            self._ensure_schema()
        return self._connection

    def _ensure_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_embeddings (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR(512) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            self._connection.commit()

    def upsert(self, entity_id: str, embedding: np.ndarray) -> None:
        """Insert or replace an identity embedding for ``entity_id``.

        Args:
            entity_id: Stable identity string (e.g. person or reference clip name).
            embedding: ``(512,)`` unit-normalised embedding.
        """
        connection = self.connect()
        vector_literal = "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reference_embeddings (id, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
                """,
                (entity_id, vector_literal),
            )
            connection.commit()

    def search(self, embedding: np.ndarray, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the closest reference identities by cosine distance.

        Args:
            embedding: ``(512,)`` suspect identity vector.
            top_k: Number of neighbours to return (defaults to config).

        Returns:
            List of dicts with ``id``, ``distance`` and ``similarity``.
        """
        config = load_config()
        k = int(top_k or config["reference_verification"]["top_k"])
        connection = self.connect()
        vector_literal = "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, embedding <=> %s::vector AS distance
                FROM reference_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector_literal, vector_literal, k),
            )
            rows = cursor.fetchall()
        results = []
        for entity_id, distance in rows:
            results.append(
                {
                    "id": entity_id,
                    "distance": float(distance),
                    "similarity": 1.0 - float(distance),
                }
            )
        return results

    def impersonation_likelihood(
        self, suspect_embedding: np.ndarray, threshold: float | None = None
    ) -> dict[str, Any]:
        """Judge impersonation risk for a suspect identity vector.

        Args:
            suspect_embedding: ``(512,)`` unit-normalised identity vector.
            threshold: Similarity threshold above which a match counts as
                potential impersonation (defaults to config).

        Returns:
            Dict with ``matches``, ``threshold``, ``max_similarity`` and
            ``impersonation_risk`` (``low``/``medium``/``high``).
        """
        config = load_config()
        threshold = float(
            threshold if threshold is not None else config["evaluation"]["similarity_threshold"]
        )
        matches = self.search(suspect_embedding)
        max_similarity = matches[0]["similarity"] if matches else 0.0
        if matches and max_similarity >= threshold:
            risk = "high" if max_similarity >= threshold + 0.15 else "medium"
        else:
            risk = "low"
        return {
            "matches": matches,
            "threshold": threshold,
            "max_similarity": max_similarity,
            "impersonation_risk": risk,
        }