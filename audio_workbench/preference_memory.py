from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any

def _db(root: str):
    p=Path(root); p.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(p/"preference_memory.sqlite3")
    con.row_factory=sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS decisions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, context_json TEXT,
      intervention_json TEXT, result TEXT, reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    return con

def remember(root: str, scope: str, context: dict[str,Any], intervention: dict[str,Any],
             result: str, reason: str) -> dict[str,Any]:
    if result not in ("accepted","rejected","tie","uncertain"):
        raise ValueError("invalid preference result")
    con=_db(root)
    cur=con.execute("INSERT INTO decisions(scope,context_json,intervention_json,result,reason) VALUES(?,?,?,?,?)",
                    (scope,json.dumps(context,ensure_ascii=False),
                     json.dumps(intervention,ensure_ascii=False),result,reason))
    con.commit(); return {"id":cur.lastrowid,"scope":scope,"result":result}

def retrieve(root: str, scope: str, limit: int=20) -> list[dict[str,Any]]:
    con=_db(root)
    rows=con.execute("SELECT * FROM decisions WHERE scope=? ORDER BY id DESC LIMIT ?",(scope,int(limit))).fetchall()
    return [dict(r) for r in rows]
