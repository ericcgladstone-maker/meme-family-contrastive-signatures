# M12 cross-source leakage audit: specification (frozen before running; `09` F-59)

**Question.** Do any development-corpus (Know Your Meme) images also occur in the Reddit denominator sample? Gallery uploads could include popular Reddit posts, and the collection windows overlap (development 12–26 May 2026; Reddit 26 April–27 May 2026).

**Inputs.**
- All 680 development-corpus images (`cross_family_corrected_index.csv`, `image_path`).
- All 4,263 Reddit denominator posts (the frame in `scores_all.csv`, joined to `q2_denom_images_manifest.csv` for `local_path`).

**Rules** (the paper's existing near-duplicate definition; SI Table S2):
- (a) exact identity: SHA-256 of the file bytes;
- (b) near-duplicate: `imagehash.phash(Image.open(p).convert("RGB"), hash_size=8)` (64-bit DCT pHash), Hamming distance ≤ 6.

**Outputs** (counts only), for (a), (b) and their union:
- the number and percentage of the 4,263 Reddit posts with at least one match;
- the number of development images implicated;
- the number of development families implicated.

Match pairs are written to the output file for traceability only. They are not viewed or inspected manually.

**Stopping rule.**
- If every count is zero, close the issue with a brief reported check.
- If any count is nonzero, STOP and report the counts to Eric/GPT before any exclusion sensitivity.
- No engagement model is rerun under this specification.

**Freeze.** `FREEZE_RECORD.json` holds the SHA-256 of this specification, the script and the three input files. The script refuses to run unless every hash verifies, and it refuses to overwrite an existing output.
