#!/usr/bin/env python3
"""Pre-freeze: series parents of the 30 DEV entries (14 §3 S2), parsed with the frozen series_parent().
DEV entry pages only. Writes heldout/frame/dev_series_parents.json (frozen with the bundle).
"""
import importlib.util
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sr", HO / "screen_and_retrieve.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def main():
    parents, status = {}, {}
    rules = sr.robots_rules()
    for slug in sr.DEV_SLUGS:
        path = f"/memes/{slug}"
        assert sr.robots_allowed(path, rules)
        r = urllib.request.urlopen(urllib.request.Request("https://knowyourmeme.com" + path,
                                                          headers={"User-Agent": sr.UA}), timeout=40)
        final = r.geturl()
        html = r.read().decode("utf-8", "ignore")
        parents[slug] = sr.series_parent(html)
        status[slug] = dict(http=r.status, final_url=final)
        print(slug, parents[slug], flush=True)
        time.sleep(sr.PAGE_DELAY + 1)
    out = dict(fetched_at=datetime.now().astimezone().isoformat(timespec="seconds"), parents=parents, status=status,
               dev_parent_set=sorted({p for p in parents.values() if p != "none"}))
    (HO / "frame" / "dev_series_parents.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
