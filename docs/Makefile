# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# this is to to autmagic gh-pages
GH_PAGES_SOURCES = docs qcodes CONTRIBUTING.rst LICENSE

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean genapi htmlfast

# we remove a few extra directories that we used to generate
# in case this is being executed from an older build
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _auto
	rm -rf api/generated
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# faster build by skipping execution of all notebooks
htmlfast:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute=never

# this is only left for backwards compatibility.
.PHONY: html-api
html-api: html

# this is only left for backwards compatibility.
.PHONY: htmlapi
htmlapi: html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
