# Minimal Sphinx Makefile.
#
# Usage:
#   make html        Build the HTML site into build/html/
#   make clean       Remove the build directory
#   make linkcheck   Verify all external links resolve
#   make serve       Build + serve at http://localhost:8000

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html linkcheck serve

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

linkcheck:
	$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: html
	@cd "$(BUILDDIR)/html" && python -m http.server 8000
