#!/usr/bin/env python3
"""Independent verification of the held-out freeze (14 section 16). Read-only.

Checks: FREEZE_RECORD.json exists; protocol SHA-256 matches; bundle file SHA-256 matches; every bundle item
re-hashes to its recorded value; the combined hash recomputes from the item map and matches the record.
Run any time (before screening, after screening, before submission). Exit 0 = verified.
"""
import hashlib
import json
import sys
from pathlib import Path

HO = Path(__file__).resolve().parent
AN, RR, PROJ = HO.parent, HO.parents[1], HO.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rec = json.loads((HO / "FREEZE_RECORD.json").read_text())
    ok = True
    p = sha(RR / rec["protocol_file"])
    print(f"protocol   {'OK ' if p == rec['protocol_sha256'] else 'BAD'} {p}")
    ok &= p == rec["protocol_sha256"]
    bf = HO / rec["bundle_file"]
    bh = sha(bf)
    print(f"bundle file {'OK ' if bh == rec['bundle_file_sha256'] else 'BAD'} {bh}")
    ok &= bh == rec["bundle_file_sha256"]
    b = json.loads(bf.read_text())
    bad = []
    for rel, h in b["items"].items():
        path = RR / rel if rel == rec["protocol_file"] else (AN / rel if (AN / rel).exists() else PROJ / rel)
        if not path.exists() or sha(path) != h:
            bad.append(rel)
    print(f"items      {len(b['items']) - len(bad)}/{len(b['items'])} verified" + (f"; FAILED: {bad}" if bad else ""))
    ok &= not bad
    comb = hashlib.sha256(json.dumps(b["items"], sort_keys=True).encode()).hexdigest()
    cmb_ok = comb == b["bundle_sha256"] == rec["bundle_sha256"]
    print(f"combined   {'OK ' if cmb_ok else 'BAD'} {comb}")
    ok &= cmb_ok
    print("FREEZE VERIFIED" if ok else "FREEZE VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
