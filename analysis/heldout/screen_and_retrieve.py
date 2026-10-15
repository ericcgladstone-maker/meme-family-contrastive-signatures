#!/usr/bin/env python3
"""Held-out-family screening and retrieval (14 §3-§6, §12). Implements the frozen rules literally.

HARD BOUNDARY: refuses to contact any candidate unless heldout/FREEZE_RECORD.json exists and its
recorded protocol hash equals the SHA-256 of 14_Heldout_Family_Protocol.md (i.e., after freeze).
`--selftest` runs only offline unit checks (queue, tokens, robots matcher, parsers on saved DEV HTML).

Frozen parameters (14 D7, D8): MIN_VARIANTS=10 distinct analyzable variants after all image-level exclusions and
D1-D4; IMAGE_SOURCES="gallery" (first gallery page only); EXCLUDE_GATED_IMAGES=True (never decode withheld URLs).
Outputs (after freeze): heldout/out/dispositions.csv, heldout/out/images/<slug>/..., heldout/out/requests.log,
heldout/out/heldout_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RR = HERE.parents[1]
FRAME = HERE / "frame"
OUT = HERE / "out"
PROTOCOL = RR / "14_Heldout_Family_Protocol.md"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
PAGE_DELAY, CDN_DELAY = 5.0, 1.0

# ---------------------------------------------------------------- frozen constants (14)
QUEUE_SEED = 20260929
H_TARGET, C_CAP, NEW_TARGET, BUDGET, ANCHOR_CAP = 60, 20, 70, 1500, 5
DEV_SLUGS = ["american-chopper-argument", "coffin-dance-dancing-pallbearers", "disaster-girl", "distracted-boyfriend",
             "doge", "drakeposting", "galaxy-brain", "hide-the-pain-harold", "is-this-a-pigeon", "lolcats", "loss",
             "mocking-spongebob", "nobody", "real-life-npc", "ok-boomer", "pepe-the-frog", "pov-memes", "roll-safe",
             "git-gud", "stonks", "surprised-pikachu", "they-did-surgery-on-a-grape", "this-is-fine", "trade-offer",
             "daily-struggle-two-buttons", "wojak", "woman-yelling-at-a-cat", "all-the-things", "y-all-got-anymore-of",
             "xzibit-yo-dawg"]
EXCLUDED_SLUGS = {"snowclone"}
EXACT_EXCLUSIONS = {"drake-and-josh-crash"}
ANCHORS = [  # (anchor DEV family, tokens) - table order = precedence
    ("wojak_soyjak", ["wojak", "soyjak", "doomer"]),
    ("pepe_the_frog", ["pepe", "apu-apustaja"]),
    ("mocking_spongebob", ["spongebob", "squidward"]),
    ("drake_hotline_bling", ["drake"]),
    ("surprised_pikachu", ["pikachu"]),
    ("doge", ["doge", "cheems"]),
    ("npc_i_support_the_current_thing", ["npc"]),
    ("stonks_meme_man", ["stonks", "meme-man"]),
]
FREEZE_PARAMS = dict(MIN_VARIANTS=10, IMAGE_SOURCES="gallery", EXCLUDE_GATED_IMAGES=True)  # 14 D7, D8 (locked 2026-09-29)


# ---------------------------------------------------------------- pure rules (unit-tested offline)
def candidate_queue(slugs):
    s = sorted(slugs)
    perm = np.random.default_rng(QUEUE_SEED).permutation(len(s))
    return [s[i] for i in perm]


def token_match(slug, token):
    sw, tw = slug.lower().split("-"), token.lower().split("-")
    return any(sw[i:i + len(tw)] == tw for i in range(len(sw) - len(tw) + 1))


def anchor_of(slug):
    if slug in EXACT_EXCLUSIONS:
        return None
    for anchor, toks in ANCHORS:
        if any(token_match(slug, t) for t in toks):
            return anchor
    return None


def robots_rules(path=FRAME / "kym_robots_2026-09-29.txt"):
    rules, active = [], False
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = [x.strip() for x in line.split(":", 1)]
        if k.lower() == "user-agent":
            active = (v == "*")
        elif active and k.lower() in ("allow", "disallow") and v:
            rules.append((k.lower(), v))
    return rules


def _pat(rule):
    rx = re.escape(rule).replace(r"\*", ".*")
    if rx.endswith(r"\$"):
        rx = rx[:-2] + "$"
    return re.compile("^" + rx)


def robots_allowed(path_and_query, rules):
    best = None  # (length, allow)
    for kind, rule in rules:
        if _pat(rule).match(path_and_query):
            L = len(rule)
            if best is None or L > best[0] or (L == best[0] and kind == "allow"):
                best = (L, kind == "allow")
    return True if best is None else best[1]


def series_parent(html):
    m = re.search(r"<p>\s*Part of a series on\s*<span>\s*<a href=\"/(memes/(?:subcultures/)?[^\"]+)\"", html)
    if not m:
        m = re.search(r"Part of a series on\s*<span>\s*<a href=\"/([^\"]+)\"", html)
    if not m:
        return "none"
    p = m.group(1)
    p = p.replace("memes/subcultures/", "subcultures/")
    return p


def entry_nsfw(html):
    m = re.search(r"<[a-zA-Z0-9]+\b[^>]*\bid=[\"']entry_nsfw_notice[\"'][^>]*>", html)
    return (m is not None), (m.group(0)[:200] if m else "")


def gallery_items(html, slug):
    out = []
    for a, body in re.findall(r'(<a\b[^>]*data-is-photo="true"[^>]*>)(.*?)</a>', html, flags=re.S):
        if f'data-entry-link="/memes/{slug}"' not in a:
            continue
        url, nsfw = None, False
        for im in re.findall(r"<img\b[^>]*>", body):
            m = re.search(r'data-image="(https://i\.kym-cdn\.com/photos/images/original/[^"]+)"', im)
            if m and url is None:
                url = m.group(1)
            if re.search(r'class="[^"]*\bnsfw-img\b', im):
                nsfw = True
        out.append({"url": url, "nsfw": nsfw})
    return out


def entry_body_images(html):
    m = re.search(r"id=[\"']entry_body[\"']", html)
    if not m:
        return []
    end = html.find("id='comments'", m.end())
    body = html[m.end(): end if end > 0 else len(html)]
    urls = re.findall(r'https://i\.kym-cdn\.com/photos/images/(?:newsfeed|original)/([0-9/]+[^"\' )<]+)', body)
    seen, out = set(), []
    for u in urls:
        full = "https://i.kym-cdn.com/photos/images/original/" + u
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


# ---------------------------------------------------------------- guards
def freeze_ok():
    """Unlock only if FREEZE_RECORD.json exists, the protocol hash matches, and EVERY bundle item re-hashes
    to its recorded value with the combined hash matching (14 section 16)."""
    rec = HERE / "FREEZE_RECORD.json"
    if not rec.exists():
        return False, "no FREEZE_RECORD.json"
    r = json.loads(rec.read_text())
    if r.get("protocol_sha256") != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        return False, "protocol hash differs from freeze record"
    bpath = HERE / r["bundle_file"]
    if hashlib.sha256(bpath.read_bytes()).hexdigest() != r.get("bundle_file_sha256"):
        return False, "bundle file hash differs from freeze record"
    b = json.loads(bpath.read_text())
    an, proj = HERE.parent, HERE.parents[2]
    for rel, h in b["items"].items():
        path = RR / rel if rel == PROTOCOL.name else (an / rel if (an / rel).exists() else proj / rel)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != h:
            return False, f"bundle item changed or missing: {rel}"
    if hashlib.sha256(json.dumps(b["items"], sort_keys=True).encode()).hexdigest() != b["bundle_sha256"] or \
            b["bundle_sha256"] != r.get("bundle_sha256"):
        return False, "combined bundle hash mismatch"
    if any(v is None for v in FREEZE_PARAMS.values()) or r.get("freeze_params") != FREEZE_PARAMS:
        return False, "FREEZE_PARAMS differ from freeze record"
    return True, "ok"


def selftest():
    slugs = [r["slug"] for r in csv.DictReader(open(FRAME / "frame_slugs.csv"))]
    q = candidate_queue(slugs)
    assert len(q) == len(set(q)) == len(slugs)
    assert token_match("angry-npc-wojak", "wojak") and token_match("apu-apustaja", "apu-apustaja")
    assert not token_match("apu-nahasapeemapetilon", "apu-apustaja") and not token_match("dogecoin", "doge")
    assert anchor_of("drake-and-josh-crash") is None and anchor_of("angry-npc-wojak") == "wojak_soyjak"
    rules = robots_rules()
    assert not robots_allowed("/memes/wojak/photos/page/2", rules)
    assert robots_allowed("/memes/wojak/photos", rules) and robots_allowed("/memes/wojak", rules)
    assert not robots_allowed("/search?q=x", rules)
    sp = Path("/private/tmp")  # parser checks run on saved DEV html if provided via --devhtml
    c_queue = [s for s in q if anchor_of(s)]
    res = dict(n_frame=len(q), queue_head=q[:5], n_c_matching=len(c_queue),
               c_by_anchor={a: sum(1 for s in c_queue if anchor_of(s) == a) for a, _ in ANCHORS})
    print(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- D4 (pure, batch-equivalent by construction)
def d4_resolve(P, min_variants, thr=6):
    """P: dict family -> list of (sha256, phash_hex) = IMMUTABLE post-D3 / pre-D4 image set of every new family
    that has ever reached D4 (in any order). Duplicate relation between images of DIFFERENT families:
    identical sha256 OR pHash Hamming <= thr. Returns family -> dict(status, retained, removed, conflicts):
      - label_conflict if, for any other family g, the images of f participating in a duplicate relation with g
        exceed 50% of |P_f|, or those of g exceed 50% of |P_g| (denominators are the immutable pre-D4 sizes);
      - otherwise retained = P_f minus EVERY image of f participating in ANY cross-family duplicate relation
        (with any family, including label-conflict families); too_small if |retained| < min_variants.
    Recomputed from scratch after each addition, so results never depend on arrival order."""
    import imagehash
    fams = list(P)
    H = {f: [imagehash.hex_to_hash(ph) for _, ph in P[f]] for f in fams}
    dup = {f: set() for f in fams}                        # image indices of f in any cross-family relation
    pair = {}                                             # (f, g) -> indices of f related to g
    for a in range(len(fams)):
        for b in range(a + 1, len(fams)):
            f, g = fams[a], fams[b]
            sf, sg = set(), set()
            for i, (sha_i, _) in enumerate(P[f]):
                for j, (sha_j, _) in enumerate(P[g]):
                    if sha_i == sha_j or (H[f][i] - H[g][j]) <= thr:
                        sf.add(i); sg.add(j)
            if sf:
                pair[(f, g)], pair[(g, f)] = sf, sg
                dup[f] |= sf; dup[g] |= sg
    out = {}
    for f in fams:
        n = len(P[f])
        conflicts = sorted(g for g in fams if g != f and (f, g) in pair and
                           (len(pair[(f, g)]) > 0.5 * n or len(pair[(g, f)]) > 0.5 * len(P[g])))
        retained = [i for i in range(n) if i not in dup[f]]
        status = "label_conflict" if conflicts else ("too_small" if len(retained) < min_variants else "analyzable")
        out[f] = dict(status=status, retained=retained, removed=sorted(dup[f]), conflicts=conflicts)
    return out


# ---------------------------------------------------------------- screening loop (runs only after freeze)
class Screener:
    def __init__(self):
        import imagehash  # noqa: F401
        OUT.mkdir(exist_ok=True)
        (OUT / "images").mkdir(exist_ok=True)
        self.rules = robots_rules()
        self.log = open(OUT / "requests.log", "a")
        self.disp = []            # disposition rows
        self.fam = {}             # slug -> dict(stratum, anchor, parent, P=immutable post-D3 tuple, kept=retained after D4)
        self.screened = set()
        self.dev_parents = set(json.loads((FRAME / "dev_series_parents.json").read_text())["parents"].values())
        self.dev_hashes = self._dev_hashes()
        self.last_page = 0.0

    def _dev_hashes(self):
        import imagehash
        from PIL import Image
        sys.path.insert(0, str(HERE.parent))
        import common as C
        idx = list(csv.DictReader(C.CFD_INDEX.open(newline="")))
        shas, phs = set(), []
        for r in idx:
            b = Path(r["image_path"]).read_bytes()
            shas.add(hashlib.sha256(b).hexdigest())
            phs.append(imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"), hash_size=8))
        return shas, phs

    def fetch(self, url, binary=False, page=True):
        from urllib.parse import urlparse
        u = urlparse(url)
        if u.netloc.endswith("knowyourmeme.com"):
            pq = u.path + (("?" + u.query) if u.query else "")
            if not robots_allowed(pq, self.rules):
                self.log.write(f"{time.time():.0f}\tBLOCKED_BY_ROBOTS\t{url}\n")
                return None, "robots"
        wait = (PAGE_DELAY if page else CDN_DELAY) - (time.time() - self.last_page)
        if wait > 0:
            time.sleep(wait)
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=40)
            final = r.geturl()
            if urlparse(final).path.startswith("/sensitive/"):
                self.log.write(f"{time.time():.0f}\tSENSITIVE_GATE\t{url}\t{final}\n")
                self.last_page = time.time()
                return None, "sensitive"
            b = r.read()
            code = r.status
        except urllib.error.HTTPError as e:
            b, code = None, e.code
        except Exception as e:  # noqa: BLE001
            b, code = None, f"ERR:{type(e).__name__}"
        self.last_page = time.time()
        self.log.write(f"{self.last_page:.0f}\t{code}\t{url}\n")
        self.log.flush()
        if b is None:
            return None, code
        return (b if binary else b.decode("utf-8", "ignore")), code

    def is_dup_of_dev(self, sha, ph):
        return sha in self.dev_hashes[0] or any((ph - d) <= 6 for d in self.dev_hashes[1])

    def record(self, rank, slug, queue, anchor, disp, **kw):
        row = dict(queue_rank=rank, slug=slug, queue=queue, token_match=int(anchor is not None), anchor=anchor or "",
                   disposition=disp, **kw)
        self.disp.append(row)
        return row

    def screen(self, rank, slug, stratum, anchor):
        import imagehash
        from PIL import Image
        self.screened.add(slug)
        if slug in DEV_SLUGS or slug in EXCLUDED_SLUGS:
            return self.record(rank, slug, stratum, anchor, "dev_entry")
        html, code = self.fetch(f"https://knowyourmeme.com/memes/{slug}")
        if code == "sensitive":
            return self.record(rank, slug, stratum, anchor, "gated_entry", evidence="entry redirected to /sensitive/")
        if html is None:
            return self.record(rank, slug, stratum, anchor, "unreachable", http=code)
        parent = series_parent(html)
        if parent in {f"memes/{d}" for d in DEV_SLUGS} or f"memes/{slug}" in self.dev_parents:
            return self.record(rank, slug, stratum, anchor, "nested_dev", parent=parent)
        nsfw, snip = entry_nsfw(html)
        if nsfw:
            return self.record(rank, slug, stratum, anchor, "gated_entry", evidence=snip)
        new_an = {s for s, f in self.fam.items() if f["status"] == "analyzable"}
        new_parents = {f["parent"] for s, f in self.fam.items() if f["status"] == "analyzable"}
        if parent in {f"memes/{s}" for s in new_an} or f"memes/{slug}" in new_parents:
            return self.record(rank, slug, stratum, anchor, "nested_new", parent=parent)
        gal, gcode = self.fetch(f"https://knowyourmeme.com/memes/{slug}/photos")
        if gcode == "sensitive":
            return self.record(rank, slug, stratum, anchor, "gated_gallery", evidence="gallery redirected to /sensitive/")
        items = gallery_items(gal, slug) if gal else []
        urls = [(it["url"], it["nsfw"], "gallery") for it in items if it["url"]]
        if not urls:
            return self.record(rank, slug, stratum, anchor, "no_gallery", http=gcode, gallery_items=len(items))
        if FREEZE_PARAMS["IMAGE_SOURCES"] == "gallery+entry":
            gset = {u for u, _, _ in urls}
            urls += [(u, False, "entry") for u in entry_body_images(html) if u not in gset]
        n_gated_img = sum(1 for it in items if it["nsfw"])  # KYM gate placeholder: <img> class token nsfw-img
        n_no_url_other = sum(1 for it in items if not it["nsfw"] and not it["url"])  # non-gated items without data-image
        if FREEZE_PARAMS["EXCLUDE_GATED_IMAGES"]:
            urls = [x for x in urls if not x[1]]
        fdir = OUT / "images" / slug
        fdir.mkdir(exist_ok=True)
        kept, shas, reps, c = [], set(), [], dict(raw=len(urls), undecodable=0, d1=0, d2=0, d3=0)
        for i, (u, nf, src) in enumerate(urls):
            b, _ = self.fetch(u, binary=True, page=False)
            if b is None:
                c["undecodable"] += 1
                continue
            try:
                ph = imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"), hash_size=8)
            except Exception:  # noqa: BLE001
                c["undecodable"] += 1
                continue
            sha = hashlib.sha256(b).hexdigest()
            if sha in shas:
                c["d1"] += 1
                continue
            shas.add(sha)
            if any((ph - r) <= 6 for r in reps):
                c["d2"] += 1
                continue
            reps.append(ph)
            if self.is_dup_of_dev(sha, ph):
                c["d3"] += 1
                continue
            last = u.split("?")[0].rsplit("/", 1)[-1]          # amendment A-1: extension from the last path
            ext = last.rsplit(".", 1)[-1].lower() if "." in last else ""   # segment only; filename-only change
            if not re.fullmatch(r"[a-z0-9]{1,5}", ext):
                ext = "img"
            fp = fdir / f"{i:03d}_{sha[:16]}.{ext}"
            fp.write_bytes(b)
            kept.append(dict(url=u, source=src, sha=sha, ph=str(ph), path=str(fp), nsfw_flag=nf))
        row = self.record(rank, slug, stratum, anchor, "pending", parent=parent, n_gated_images=n_gated_img,
                          n_no_url_other=n_no_url_other,
                          n_postdedup=len(kept), **c)
        if len(kept) < FREEZE_PARAMS["MIN_VARIANTS"]:
            row["disposition"] = "too_small"
            return row
        # reached D4: register the IMMUTABLE post-D3 set, then recompute D4 for all families from scratch
        self.fam[slug] = dict(stratum=stratum, anchor=anchor, parent=parent, P=tuple(kept), kept=list(kept),
                              status="analyzable", row=row)
        self.resolve()
        return row

    def resolve(self):
        """Batch D4 over every family that ever reached D4; update statuses/retained sets/disposition rows."""
        res = d4_resolve({f: [(a["sha"], a["ph"]) for a in v["P"]] for f, v in self.fam.items()},
                         FREEZE_PARAMS["MIN_VARIANTS"])
        for f, r in res.items():
            v = self.fam[f]
            v["status"] = r["status"]
            v["kept"] = [v["P"][i] for i in r["retained"]]
            v["row"]["disposition"] = r["status"]
            v["row"]["n_after_d4"] = len(v["kept"])
            v["row"]["d4"] = len(r["removed"])
            v["row"]["d4_conflicts"] = ";".join(r["conflicts"])

    def counts(self):
        h = sum(1 for f in self.fam.values() if f["status"] == "analyzable" and f["stratum"] == "H")
        cc = sum(1 for f in self.fam.values() if f["status"] == "analyzable" and f["stratum"] == "C")
        return h, cc

    def c_at_cap(self, anchor):
        return sum(1 for f in self.fam.values() if f["status"] == "analyzable" and f["stratum"] == "C"
                   and f["anchor"] == anchor) >= ANCHOR_CAP

    def run(self):
        slugs = [r["slug"] for r in csv.DictReader(open(FRAME / "frame_slugs.csv"))]
        q = candidate_queue(slugs)
        hq = iter([(i, s) for i, s in enumerate(q)])
        cq = [(i, s) for i, s in enumerate(q) if anchor_of(s)]

        def next_h():
            for i, s in hq:
                if s not in self.screened:
                    return i, s
            return None

        def budget_left():
            return len(self.screened) < BUDGET
        cq_iter = iter(cq)
        c_exhausted = [False]

        def screen_next_c():
            for i, s in cq_iter:
                if s in self.screened:
                    continue
                an = anchor_of(s)
                if self.c_at_cap(an):
                    self.screened.add(s)
                    self.record(i, s, "C", an, "anchor_cap")
                    continue
                self.screen(i, s, "C", an)
                return True
            c_exhausted[0] = True
            return False

        def closed():
            h, c = self.counts()
            return h >= H_TARGET and (c >= C_CAP or c_exhausted[0]) and h + c >= NEW_TARGET

        # dynamic closure (14 section 5): after every D4 update, continue until H >= 60, C = 20 or C-queue
        # exhausted, and H + C >= 70; later D4 losses re-open the appropriate queue. Budget: 1,500 unique entries.
        while not closed() and budget_left():
            h, c = self.counts()
            if h < H_TARGET:
                nx = next_h()
                if nx is None:
                    break
                self.screen(nx[0], nx[1], "H", anchor_of(nx[1]))
            elif c < C_CAP and not c_exhausted[0]:
                screen_next_c()
            elif h + c < NEW_TARGET:
                nx = next_h()
                if nx is None:
                    break
                self.screen(nx[0], nx[1], "H", anchor_of(nx[1]))
            else:
                break
            self.save()
        self.save(final=True)

    def save(self, final=False):
        cols = sorted({k for r in self.disp for k in r})
        with open(OUT / "dispositions.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(self.disp)
        with open(OUT / "heldout_manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["family_slug", "stratum", "anchor", "queue_rank", "image_path", "sha256", "source", "nsfw_flag"])
            for s, f in self.fam.items():
                if f["status"] == "analyzable":
                    for a in f["kept"]:
                        w.writerow([s, f["stratum"], f["anchor"] or "", f["row"]["queue_rank"], a["path"], a["sha"],
                                    a["source"], int(a["nsfw_flag"])])
        if final:
            h, cc = self.counts()
            (OUT / "screening_summary.json").write_text(json.dumps(dict(
                screened=len(self.screened), H=h, C=cc, budget=BUDGET,
                shortfall=bool(h < H_TARGET or h + cc < NEW_TARGET),
                dispositions={d: sum(1 for r in self.disp if r["disposition"] == d)
                              for d in sorted({r["disposition"] for r in self.disp})},
                freeze_params=FREEZE_PARAMS), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--devhtml", nargs="*", default=[], help="saved DEV entry/gallery html for parser checks")
    ap.add_argument("--check-freeze", action="store_true", help="only report whether the freeze guard unlocks")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        for p in a.devhtml:
            t = Path(p).read_text(errors="ignore")
            print(Path(p).name, "parent:", series_parent(t), "| nsfw:", entry_nsfw(t)[0],
                  "| body imgs:", len(entry_body_images(t)))
        return
    ok, why = freeze_ok()
    if a.check_freeze:
        print("FREEZE GUARD:", "UNLOCKED" if ok else f"LOCKED ({why})")
        return
    if not ok:
        sys.exit(f"REFUSING TO SCREEN: {why}. Freeze 14 first.")
    Screener().run()


if __name__ == "__main__":
    main()
