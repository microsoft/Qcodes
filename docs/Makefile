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

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _auto
	rm -rf api/generated
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# generate api docs for instruments automatically
genapi:
	sphinx-apidoc  -o  _auto  -d 10 ../qcodes \
		../qcodes/instrument_drivers/Spectrum/pyspcm.py \
		../qcodes/instrument_drivers/Spectrum/M4i.py \
		../qcodes/instrument_drivers/keysight \
		../qcodes/instrument_drivers/Keysight/keysightb1500/* \
		../qcodes/instrument_drivers/QuantumDesign/DynaCoolPPMS/private/*
	mkdir -p api/generated/
	cp _auto/qcodes.instrument_drivers.* api/generated/

# faster build by skipping execution of all notebooks
htmlfast: genapi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute=never

# this is only left for backwards compatibility.
.PHONY: html-api
html-api: html

# this is only left for backwards compatibility.
.PHONY: htmlapi
htmlapi: html

.PHONY: gh-pages
gh-pages:
	git config --global user.email "bot@travics.com"
	git config --global user.name "Documentation Bot"
	git remote add upstream https://${GITHUB_API_KEY}@github.com/QCoDeS/Qcodes.git
	git fetch upstream
	git checkout gh-pages
	rm -rf ./*
	git checkout master $(GH_PAGES_SOURCES)
	git reset HEAD
	cd docs && make htmlapi && \
	mv -fv $(BUILDDIR)/html/* ../
	rm -rf $(GH_PAGES_SOURCES) build
	git add -A
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short \
	--abbrev-commit`" && git push https://${GITHUB_API_KEY}@github.com/QCoDeS/Qcodes.git gh-pages ; git checkout master

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile genapi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
