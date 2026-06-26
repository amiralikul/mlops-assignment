"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"
TABLE_METADATA = DB_DIR / "dev_20240627" / "dev_tables.json"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


@lru_cache(maxsize=32)
def _column_descriptions(db_id: str) -> dict[tuple[str, str], str]:
    """Load BIRD's semantic column names keyed by original table/column names."""
    if not TABLE_METADATA.exists():
        return {}

    for db in json.loads(TABLE_METADATA.read_text()):
        if db.get("db_id") != db_id:
            continue

        table_names = db.get("table_names_original", [])
        original_columns = db.get("column_names_original", [])
        semantic_columns = db.get("column_names", [])
        descriptions: dict[tuple[str, str], str] = {}

        for original, semantic in zip(original_columns, semantic_columns, strict=False):
            table_index, column_name = original
            if table_index == -1:
                continue
            description = semantic[1]
            if isinstance(description, str) and description != column_name:
                descriptions[(table_names[table_index], column_name)] = description
        return descriptions

    return {}


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    descriptions = _column_descriptions(db_id)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                description = descriptions.get((t, name))
                if description:
                    line += f" -- {description}"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                from_col = fk[3]
                ref_table = fk[2]
                to_col = fk[4]
                if not from_col or not ref_table:
                    continue
                reference = f"  FOREIGN KEY ({_q(from_col)}) REFERENCES {_q(ref_table)}"
                if to_col:
                    reference += f"({_q(to_col)})"
                col_lines.append(reference)
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
