#!/usr/bin/env bash
# Compile paper/aerograph.tex to paper/aerograph.pdf.
# Requires a working TeX distribution (TeX Live or MacTeX) with
# pdflatex and bibtex on PATH.
#
# Install on macOS:   brew install --cask mactex-no-gui
# Install on Debian:  sudo apt-get install texlive-latex-recommended texlive-publishers texlive-fonts-recommended texlive-bibtex-extra
#
# Usage: cd paper && ./compile.sh
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode aerograph.tex
bibtex aerograph
pdflatex -interaction=nonstopmode aerograph.tex
pdflatex -interaction=nonstopmode aerograph.tex

echo ""
echo "Build complete. Output: paper/aerograph.pdf"
