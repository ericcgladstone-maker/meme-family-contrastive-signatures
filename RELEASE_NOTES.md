# Release notes

## v2.0.0-r1 (October 2026)

The revision package for the R1 version of the article, submitted to *Social Science Computer Review* on 5 October 2026.

**Layout.** As described in `README.md`: `analysis/` and `revision/` hold the revision's scripts and audited outputs, and
`submitted_package/` holds the originally submitted package (previously at the repository root),
unchanged.

**Integrity.** Every file listed in `REPRODUCIBILITY_MANIFEST.csv` that is distributed here was copied byte for byte from
the project tree and matches the SHA-256 checksum recorded in the manifest. No script, data file or output was edited for
this release.

**Not distributed.**

- Retrieved held-out images (`analysis/heldout/out/images/` and the aborted first run), as the manifest states. Per-file
  hashes are in `analysis/heldout/out/heldout_manifest.csv`.
- Four submission documents listed in the manifest: the cover letter, the response to reviewers, the change memo and the
  title page (`revision/build/out/SSCR_{Cover_Letter,Response_Letter,Change_Memo,Title_Page}_R1.docx`). The R1
  manuscript and supplement are included.

**Paths.** `analysis/common.py` points at the derived inputs of the authors' original project tree; repoint it if you
relocate the package (see `README.md`). A few logs and manifests record the absolute paths of the machine on which they
were produced.

The OSF project cited in `CITATION.cff` (DOI 10.17605/OSF.IO/BFWZE) is not yet public.
