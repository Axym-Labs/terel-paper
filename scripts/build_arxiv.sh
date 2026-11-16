#!/usr/bin/env bash
# Build the arXiv source package dist/terel-arxiv.zip: main.tex, the shipped
# main.bbl, style files, link symbols and figures; no auxiliary files.
set -euo pipefail
cd "$(dirname "$0")/.."
out=dist/arxiv
rm -rf "$out" dist/terel-arxiv.zip
mkdir -p "$out/figures"
tectonic -X compile --keep-intermediates main.tex
cp main.tex main.bbl axym-publication.sty axym-links.sty "$out"/
cp -r axym-emoji "$out"/
cp figures/terel-method-overview.pdf figures/performance-mechanisms.pdf \
  figures/representation-geometry.pdf figures/neuron-state-behavior.pdf "$out/figures"/
(cd "$out" && zip -qrX ../terel-arxiv.zip .)
echo "Wrote dist/terel-arxiv.zip"
