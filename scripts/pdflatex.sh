#!/usr/bin/env bash
# Build a LaTeX document with pdflatex, the engine arXiv uses (TeX Live 2023 in
# the axym-latex-html Docker image): pdflatex, bibtex (when the document has a
# bibliography), then pdflatex twice (up to two more runs while LaTeX asks for a
# rerun). Leaves <name>.pdf and <name>.bbl.
# Usage: scripts/pdflatex.sh <file.tex> [directory, default: .]
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <file.tex> [directory]" >&2; exit 2; }
name=$(basename "$1" .tex)
dir=$(realpath "${2:-.}")
docker run --rm -u "$(id -u):$(id -g)" -v "$dir":/w -w /w axym-latex-html:latest \
  sh -c '
    run() {
      pdflatex -interaction=nonstopmode -halt-on-error "$1.tex" > /dev/null 2>&1 || {
        grep -A8 "^!" "$1.log" >&2 || tail -40 "$1.log" >&2; exit 1; }
    }
    run "$1"
    if grep -q "\\\\bibdata" "$1.aux"; then
      bibtex "$1" > /dev/null 2>&1; status=$?
      [ "$status" -le 1 ] || { cat "$1.blg" >&2; exit 1; }
    fi
    run "$1"
    run "$1"
    for extra in 1 2; do
      grep -q "Rerun to get" "$1.log" || break
      run "$1"
    done
    grep -h "undefined references\|Rerun to get" "$1.log" >&2 || true
  ' sh "$name"
echo "Built $dir/$name.pdf (pdflatex)"
