"""Build and query the SQLite store used by the graph channel.

The store is built once from the released graph tables (parquet):
entities, categories, papers, edges_entity_paper and mentions.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

STOPWORDS = {"a", "an", "and", "are", "articles", "for", "have", "in", "of", "on", "or", "reported",
             "the", "to", "what", "which", "with"}
EVIDENCE_LIMIT = 50000

SCHEMA = """
CREATE TABLE papers (paper_id TEXT PRIMARY KEY, doi TEXT, year INTEGER, journal TEXT);
CREATE TABLE categories (category_id TEXT PRIMARY KEY, level TEXT, entity_type TEXT,
                         name_en TEXT, name_zh TEXT, parent_id TEXT);
CREATE TABLE entities (entity_id TEXT PRIMARY KEY, name TEXT, entity_type TEXT,
                       l3_category_id TEXT, l2_category_id TEXT, paper_count INTEGER);
CREATE TABLE entity_papers (entity_id TEXT, paper_id TEXT);
CREATE VIRTUAL TABLE mention_fts USING fts5(surface_text, paper_id UNINDEXED);
"""
INDEXES = """
CREATE INDEX idx_ep_entity ON entity_papers(entity_id);
CREATE INDEX idx_ent_l2 ON entities(l2_category_id);
CREATE INDEX idx_ent_l3 ON entities(l3_category_id);
"""


def build_store(graph_dir: str | Path, out: str | Path, batch: int = 200_000) -> None:
    """Create ``out`` (SQLite) from the parquet tables in ``graph_dir``/parquet."""
    import pandas as pd
    import pyarrow.parquet as pq

    pdir = Path(graph_dir) / "parquet"
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)

    papers = pd.read_parquet(pdir / "papers.parquet", columns=["paper_id", "doi", "year", "journal"])
    papers["year"] = papers["year"].astype("object").where(papers["year"].notna(), None)
    con.executemany("INSERT INTO papers VALUES (?,?,?,?)", papers.itertuples(index=False, name=None))
    cats = pd.read_parquet(pdir / "categories.parquet")
    con.executemany("INSERT INTO categories VALUES (?,?,?,?,?,?)",
                    cats[["category_id", "level", "entity_type", "name_en", "name_zh", "parent_id"]]
                    .itertuples(index=False, name=None))
    ents = pd.read_parquet(pdir / "entities.parquet", columns=[
        "entity_id", "name", "entity_type", "l3_category_id", "l2_category_id", "paper_count"])
    con.executemany("INSERT INTO entities VALUES (?,?,?,?,?,?)", ents.itertuples(index=False, name=None))

    for rb in pq.ParquetFile(pdir / "edges_entity_paper.parquet").iter_batches(
            batch_size=batch, columns=["entity_id", "paper_id"]):
        con.executemany("INSERT INTO entity_papers VALUES (?,?)", zip(*[c.to_pylist() for c in rb.columns]))

    seen = set()
    for rb in pq.ParquetFile(pdir / "mentions.parquet").iter_batches(
            batch_size=batch, columns=["surface_text", "paper_id"]):
        rows = []
        for text, pid in zip(*[c.to_pylist() for c in rb.columns]):
            key = (text, pid)
            if text and key not in seen:
                seen.add(key)
                rows.append(key)
        con.executemany("INSERT INTO mention_fts VALUES (?,?)", rows)
    con.executescript(INDEXES)
    con.commit()
    con.close()


def fts_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class GraphStore:
    def __init__(self, path: str | Path):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        rows = self.con.execute(
            "SELECT category_id, level, entity_type, name_en, parent_id FROM categories").fetchall()
        self.categories = {r[0]: r for r in rows}
        self._by_name = {(r[2], r[1], r[3].lower()): r[0] for r in rows}

    # ---- taxonomy shown to LLM2
    def taxonomy_text(self, entity_type: str) -> str:
        rows = self.con.execute(
            "SELECT l3.name_en, l2.name_en, COUNT(e.entity_id) FROM categories l2 "
            "JOIN categories l3 ON l3.category_id = l2.parent_id "
            "LEFT JOIN entities e ON e.l2_category_id = l2.category_id "
            "WHERE l2.level = 'L2' AND l2.entity_type = ? GROUP BY l2.category_id ORDER BY l2.category_id",
            (entity_type,)).fetchall()
        return "\n".join(f"{a} / {b} / {n} entities" for a, b, n in rows) or "(no categories for this type)"

    def resolve_category(self, entity_type: str, l3: str, l2: str) -> str | None:
        """Map LLM2's (l3, l2) names onto a category id; an empty l2 means the whole L3."""
        if l2:
            cid = self._by_name.get((entity_type, "L2", l2.strip().lower()))
            if cid:
                return cid
        return self._by_name.get((entity_type, "L3", (l3 or "").strip().lower()))

    # ---- the two channels
    def category_papers(self, category_id: str) -> set[str]:
        col = "l2_category_id" if self.categories[category_id][1] == "L2" else "l3_category_id"
        rows = self.con.execute(
            f"SELECT DISTINCT ep.paper_id FROM entity_papers ep JOIN entities e ON e.entity_id = ep.entity_id "
            f"WHERE e.{col} = ?", (category_id,)).fetchall()
        return {r[0] for r in rows}

    def evidence_papers(self, text: str) -> set[str]:
        toks = fts_tokens(text)
        if not toks:
            return set()
        q = " OR ".join(dict.fromkeys(toks[:10]))
        try:
            rows = self.con.execute(
                "SELECT DISTINCT paper_id FROM mention_fts WHERE mention_fts MATCH ? LIMIT ?",
                (q, EVIDENCE_LIMIT)).fetchall()
        except sqlite3.OperationalError:
            return set()
        return {r[0] for r in rows}

    def paper_meta(self, paper_ids: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(paper_ids), 900):
            chunk = paper_ids[i:i + 900]
            q = "SELECT paper_id, doi, year, journal FROM papers WHERE paper_id IN (%s)" % ",".join("?" * len(chunk))
            for pid, doi, year, journal in self.con.execute(q, chunk):
                out[pid] = {"doi": doi, "year": year, "journal": journal}
        return out
