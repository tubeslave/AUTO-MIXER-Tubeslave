from __future__ import annotations
import numpy as np

def active_spread_db(frame_db:np.ndarray,active:np.ndarray)->float:
    q=frame_db[active]
    if len(q)<4:return 0.
    return float(np.percentile(q,90)-np.percentile(q,10))

def evaluate(before:dict,after:dict)->dict:
    rows={};fail=[]
    for name,b in before.items():
        a=after[name];improvement=b["spread_db"]-a["spread_db"]
        rows[name]={"before_spread_db":b["spread_db"],"after_spread_db":a["spread_db"],
                    "spread_reduction_db":improvement,"p95_gr_db":a.get("p95_gr_db",0)}
        if a.get("p95_gr_db",0)>5.1:fail.append(f"{name}:excess_gr")
        if improvement>7:fail.append(f"{name}:overflattened")
    return {"accept":not fail,"failures":fail,"tracks":rows}
