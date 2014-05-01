# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        = python
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest

export SHELL=/bin/bash

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  qthelp    to make HTML files and a qthelp project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf $(BUILDDIR) source/generated
	-rm generate-stamp

SPHINXEXT_PATH=$(shell python -c 'import sphinx.ext; from distutils.version import LooseVersion; print LooseVersion(sphinx.__version__) < "1.1.2" and "sphinxext" or sphinx.ext.__path__[0]')
#SPHINXEXT_PATH=sphinxext

generate: generate-stamp
generate-stamp:
	mkdir -p $(BUILDDIR)
	@MVPA_EXTERNALS_RAISE_EXCEPTION=off \
		PYTHONPATH=$(CURDIR)/sphinxext:$(CURDIR)/..:$(PYTHONPATH) \
		$(PYTHON) $(SPHINXEXT_PATH)/autosummary/generate.py \
		-t $(CURDIR)/templates -o source/generated source/*.rst
	# wipe everything that is not pymvpa to prevent suffering from the bugs
	# of other in subsequent stages
	find source/generated -type f ! -name 'mvpa*' -delete
	set -e; set -o pipefail; for man in $(BUILDDIR)/man/pymvpa*; do \
		manfile=$$(basename $${man}) ; \
		cmd=$${manfile#pymvpa2-*} ; \
		cmd=$${cmd%*.1} ; \
		man2html $${man} | sed -n '/<HR>/{:a;n;/<HR>/b;p;ba}' | sed -n '/<A NAME/{:a;n;/This document was created by/b;p;ba}' > source/generated/$${manfile}.html ; \
		underline="*****************$$(echo $${cmd} | tr A-Za-z0-9 \*)" ; \
		echo ".. index:: command line, $${cmd}\n.. _chap_cmdline_$${cmd}\n\n$${underline}\nMan page for \`\`$${cmd}\`\`\n$${underline}\n\n.. raw:: html\n   :file: $${manfile}.html\n" > source/generated/cmd_$${cmd}.rst ; \
	done
	@touch $@

html: generate
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: generate
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle: generate
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json: generate
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: generate
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: generate
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/PyMVPA.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/PyMVPA.qhc"

latex: generate
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes: generate
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck: generate
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest: generate
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."
