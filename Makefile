PROFILE_FILE=$(CURDIR)/$(BUILDDIR)/main.pstats
COVERAGE_REPORT=$(CURDIR)/$(BUILDDIR)/coverage
BUILDDIR=$(CURDIR)/build
HTML_DIR=$(BUILDDIR)/html
DOC_DIR=$(CURDIR)/doc
TUT_DIR=$(CURDIR)/datadb/tutorial_data/tutorial_data
DOCSRC_DIR=$(DOC_DIR)/source
DOCBUILD_DIR=$(BUILDDIR)/doc
MAN_DIR=$(BUILDDIR)/man
APIDOC_DIR=$(HTML_DIR)/api
PDF_DIR=$(BUILDDIR)/pdf
LATEX_DIR=$(BUILDDIR)/latex
WWW_DIR=$(BUILDDIR)/website
SWARM_DIR=$(BUILDDIR)/swarm
WWW_UPLOAD_URI=www.pymvpa.org:/home/www/www.pymvpa.org/pymvpa
WWW_UPLOAD_URI_DEV=dev.pymvpa.org:/home/www/dev.pymvpa.org/pymvpa
DATA_UPLOAD_URI=data.pymvpa.org:/home/www/data.pymvpa.org/www/datasets
DATA_URI=data.pymvpa.org::datadb
SWARMTOOL_DIR=tools/codeswarm
SWARMTOOL_DIRFULL=$(CURDIR)/$(SWARMTOOL_DIR)
RSYNC_OPTS=-az -H --no-perms --no-owner --verbose --progress --no-g --exclude prev/
RSYNC_OPTS_UP=-rzlhv --delete
# -p --chmod=Dg+s,g+rw,o+rX

#
# The Python executable to be used
#
PYTHON = python
NOSETESTS = $(PYTHON) $(shell which nosetests)

#
# Helpers for version handling.
# Note: can't be ':='-ed since location of invocation might vary
DEBCHANGELOG_VERSION = $(shell dpkg-parsechangelog | egrep ^Version | cut -d ' ' -f 2,2 | cut -d '-' -f 1,1)
SETUPPY_VERSION = $(shell $(PYTHON) setup.py -V)

#
# Automatic development version
#
#yields: LastTagName_CommitsSinceThat_AbbrvHash
DEV_VERSION := $(shell git describe --abbrev=4 HEAD |sed -e 's/-/+/g' |cut -d '/' -f 2,2)

# By default we are releasing with setup.py version
RELEASE_VERSION ?= $(SETUPPY_VERSION)
RELEASE_CODE ?=

# Conditional depends regulated from outside
#
ifdef PYMVPA_NO_3RD
	build_depends :=
else
	build_depends :=
endif

#
# Details on the Python/system
#

PYVER := $(shell $(PYTHON) -V 2>&1 | cut -d ' ' -f 2,2 | cut -d '.' -f 1,2)
DISTUTILS_PLATFORM := $(shell $(PYTHON) -c "import distutils.util; print distutils.util.get_platform()")

#
# Little helpers
#

mkdir-%:
	if [ ! -d $($*) ]; then mkdir -p $($*); fi


#
# Building
#

all: build

# build included 3rd party pieces (if present)
3rd: 3rd-stamp
3rd-stamp:
	find 3rd -mindepth 1 -maxdepth 1  -type d | \
	 while read d; do \
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d" || :; \
     done
	touch $@


debian-build:
# reuse is better than duplication (yoh)
	debian/rules build


build: build-stamp
build-stamp: $(build_depends)
	$(PYTHON) setup.py config --noisy
	$(PYTHON) setup.py build_ext --inplace
	touch $@


#
# Cleaning
#

# this target is used to clean things for a fresh build
clean:
	@echo "I: Performing clean operation"
# clean 3rd party pieces
	find 3rd -mindepth 1 -maxdepth 1  -type d | \
	 while read d; do \
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d" clean || : ; \
     done
# clean tools
	$(MAKE) -C tools clean
# clean pics
	$(MAKE) -C doc/pics clean
# clean docs
	$(MAKE) -C doc clean
	-@rm -f $(DOCSRC_DIR)/examples/*.rst
# clean all bits and pieces
	-@rm -f MANIFEST
	-@rm -f mvpa/clfs/lib*/*.so \
		mvpa/clfs/lib*/*.dylib \
		mvpa/clfs/lib*/*_wrap.* \
		mvpa/clfs/lib*/*c.py \
		mvpa/tests/*.{prof,pstats,kcache}
	@find . -name '*.py[co]' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -name 'iterate.dat' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '*.gch' \
		 -o -iname '*_flymake.*' \
		 -o -iname '#*#' | xargs -L 10 rm -f
	-@rm -rf build
	-@rm -rf dist *report
	-@rm -f *-stamp *_report.pdf *_report.log pymvpa.cfg

# this target should put the source tree into shape for building the source
# distribution
distclean: clean
# if we are on debian system - we might have left-overs from build
	-@$(MAKE) debian-clean
	-@rm -rf tools/codeswarm



debian-clean:
# remove stamps for builds since state is not really built any longer
	-fakeroot debian/rules clean


#
# Documentation
#

doc: website manpages

pics:
	$(MAKE) -C doc/pics

manpages: mkdir-MAN_DIR
	@echo "I: Creating manpages"
	PYTHONPATH=.:$(PYTHONPATH) help2man -N -n 'preprocess fMRI data for PyMVPA' \
		bin/mvpa-prep-fmri > $(MAN_DIR)/mvpa-prep-fmri.1
	PYTHONPATH=. help2man -N -n 'query stereotaxic atlases' \
		bin/atlaslabeler > $(MAN_DIR)/atlaslabeler.1

references:
	@echo "I: Generating references"
	tools/bib2rst_ref.py

htmldoc: examples2rst build pics
	@echo "I: Creating an HTML version of documentation"
	cd $(DOC_DIR) && MVPA_EXTERNALS_RAISE_EXCEPTION=off \
		PYTHONPATH=$(CURDIR):$(PYTHONPATH) \
		$(MAKE) html BUILDDIR=$(BUILDDIR) SPHINXOPTS="$(SPHINXOPTS)"
	cd $(HTML_DIR)/generated && ln -sf ../_static
	cd $(HTML_DIR)/examples && ln -sf ../_static
	cd $(HTML_DIR)/workshops && ln -sf ../_static
	cd $(HTML_DIR)/datadb && ln -sf ../_static
	cp $(DOCSRC_DIR)/pics/history_splash.png $(HTML_DIR)/_images/

pdfdoc: examples2rst build pics pdfdoc-stamp
pdfdoc-stamp:
	@echo "I: Creating a PDF version of documentation"
	cd $(DOC_DIR) && MVPA_EXTERNALS_RAISE_EXCEPTION=off \
		PYTHONPATH=$(CURDIR):$(PYTHONPATH) \
		$(MAKE) latex BUILDDIR=$(BUILDDIR) SPHINXOPTS="$(SPHINXOPTS)"
	cd $(LATEX_DIR) && $(MAKE) all-pdf
	touch $@

# Create a handy .pdf of the manual to be printed as a book
handbook: pdfdoc
	@echo "I: Creating a handbook of the manual"
	cd tools && $(MAKE) pdfbook
	build/tools/pdfbook -2 \
	 $(LATEX_DIR)/PyMVPA-Manual.pdf $(LATEX_DIR)/PyMVPA-Manual-Handbook.pdf

examples2rst: examples2rst-stamp
examples2rst-stamp: mkdir-DOCBUILD_DIR
	tools/ex2rst \
		--project PyMVPA \
		--outdir $(DOCSRC_DIR)/examples \
		--exclude doc/examples/searchlight_app.py \
		doc/examples
	touch $@

apidoc: apidoc-stamp
apidoc-stamp: build
# Disabled profiling for now, it consumes huge amounts of memory, so I doubt
# that all buildds can do it. In theory it would only be done on a single
# developer machine, because it is only necessary for the arch-all package,
# but e.g. dpkg-buildpackage runs the indep target anyway -- not sure about
# the buildds, though.
#apidoc-stamp: profile
	@echo "I: Creating an API documentation with epydoc"
	mkdir -p $(HTML_DIR)/api
	LC_ALL=C MVPA_EPYDOC_WARNINGS=once tools/epydoc --config doc/api/epydoc.conf
	touch $@

# this takes some minutes !!
profile: build mvpa/tests/__init__.py
	@echo "I: Profiling unittests"
	@PYTHONPATH=.:$(PYTHONPATH) tools/profile -K  -O $(PROFILE_FILE) mvpa/tests/__init__.py


#
# Website
#

website: website-stamp
website-stamp: mkdir-WWW_DIR htmldoc pdfdoc
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	cp $(LATEX_DIR)/PyMVPA-*.pdf $(WWW_DIR)
	tools/sitemap.sh > $(WWW_DIR)/sitemap.xml
# main icon of the website
	cp $(DOCSRC_DIR)/pics/favicon.png $(WWW_DIR)/_images/
# for those who do not care about <link> and just trying to download it
	cp $(DOCSRC_DIR)/pics/favicon.png $(WWW_DIR)/favicon.ico
# provide robots.txt to minimize unnecessary traffic
	cp $(DOCSRC_DIR)/_static/robots.txt $(WWW_DIR)/
# provide promised pylintrc
	mkdir -p $(WWW_DIR)/misc && cp $(DOC_DIR)/misc/pylintrc $(WWW_DIR)/misc
	touch $@

upload-website:
	$(MAKE) website SPHINXOPTS='-D html_theme=pymvpa_online'
	rsync $(RSYNC_OPTS_UP) $(WWW_DIR)/* $(WWW_UPLOAD_URI)/
	git update-ref refs/heads/website-updates/www HEAD

upload-htmldoc:
	$(MAKE) htmldoc SPHINXOPTS='-D html_theme=pymvpa_online'
	chmod a+rX -R $(HTML_DIR)
	rsync $(RSYNC_OPTS_UP) $(HTML_DIR)/* $(WWW_UPLOAD_URI)/
	git update-ref refs/heads/website-updates/www-html HEAD

upload-website-dev:
	sed -i -e "s,http://disqus.com/forums/pymvpa/,http://disqus.com/forums/pymvpa-dev/,g" \
		-e "s,^<!-- HEADNOTES -->,<!-- HEADNOTES --><div class='admonition note'>This content refers to an unreleased development version of PyMVPA</div>,g" \
		doc/source/_themes/pymvpa_online/page.html
	$(MAKE) website SPHINXOPTS='-D html_theme=pymvpa_online'
	sed -i -e "s,http://disqus.com/forums/pymvpa-dev/,http://disqus.com/forums/pymvpa/,g" \
		-e "s,^<!-- HEADNOTES -->.*$$,<!-- HEADNOTES -->,g" \
		doc/source/_themes/pymvpa_online/page.html
	sed -i -e "s,www.pymvpa.org,dev.pymvpa.org,g" $(WWW_DIR)/sitemap.xml
	chmod a+rX -R $(WWW_DIR)
	rsync $(RSYNC_OPTS_UP) $(WWW_DIR)/* $(WWW_UPLOAD_URI_DEV)/

upload-htmldoc-dev:
	grep -q pymvpa-dev doc/source/_themes/pymvpa_online/page.html || \
	 sed -i -e "s,http://disqus.com/forums/pymvpa/,http://disqus.com/forums/pymvpa-dev/,g" \
		-e "s,^<!-- HEADNOTES -->,<!-- HEADNOTES --><div class='admonition note'>This content refers to an unreleased development version of PyMVPA</div>,g" \
		doc/source/_themes/pymvpa_online/page.html
	$(MAKE) htmldoc SPHINXOPTS='-D html_theme=pymvpa_online'
	sed -i -e "s,http://disqus.com/forums/pymvpa-dev/,http://disqus.com/forums/pymvpa/,g" \
		-e "s,^<!-- HEADNOTES -->.*$$,<!-- HEADNOTES -->,g" \
		doc/source/_themes/pymvpa_online/page.html
	rsync $(RSYNC_OPTS_UP) $(HTML_DIR)/* $(WWW_UPLOAD_URI_DEV)/


# upload plain .rst files as descriptions to data.pympa.org as descriptions of
# each dataset
upload-datadb-descriptions:
	for ds in doc/source/datadb/*; do \
		ds=$$(basename $${ds}); ds=$${ds%*.rst}; \
		scp doc/source/datadb/$${ds}.rst $(DATA_UPLOAD_URI)/$${ds}/README.rst; \
	done

#
# Tests (unittests, docs, examples)
#

ut-%: build
	@PYTHONPATH=.:$(PYTHONPATH) $(NOSETESTS) --nocapture mvpa/tests/test_$*.py

unittest: build
	@echo "I: Running unittests (without optimization nor debug output)"
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) mvpa/tests/__init__.py


# test if PyMVPA is working if optional externals are missing
unittest-badexternals: build
	@echo "I: Running unittests under assumption of missing optional externals."
	@PYTHONPATH=mvpa/tests/badexternals:.:$(PYTHONPATH) $(PYTHON) mvpa/tests/__init__.py 2>&1 \
	| grep -v -e 'WARNING: Known dependency' -e 'Please note: w' \
              -e 'WARNING:.*SMLR.* implementation'

# only non-labile tests
unittest-nonlabile: build
	@echo "I: Running only non labile unittests. None of them should ever fail."
	@PYTHONPATH=.:$(PYTHONPATH) MVPA_TESTS_LABILE=no $(PYTHON) mvpa/tests/__init__.py

# test if no errors would result if we force enabling of all ca
unittest-ca: build
	@echo "I: Running unittests with all ca enabled."
	@PYTHONPATH=.:$(PYTHONPATH) MVPA_DEBUG=ENFORCE_CA_ENABLED $(PYTHON) mvpa/tests/__init__.py

# Run unittests with optimization on -- helps to catch unconditional
# debug calls
unittest-optimization: build
	@echo "I: Running unittests with $(PYTHON) -O."
	@PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) -O mvpa/tests/__init__.py

# Run unittests with all debug ids and some metrics (crossplatform ones) on.
#   That does:
#     additional checking,
#     debug() calls validation, etc
# Need to use /bin/bash due to use of PIPESTATUS
unittest-debug: SHELL=/bin/bash
unittest-debug: build
	@echo "I: Running unittests with debug output. No progress output."
	@PYTHONPATH=.:$(PYTHONPATH) MVPA_DEBUG=.* MVPA_DEBUG_METRICS=ALL \
       $(PYTHON) mvpa/tests/__init__.py 2>&1 \
       |  sed -n -e '/^[=-]\{60,\}$$/,$$p'; \
	   exit $${PIPESTATUS[0]}	# reaquire status of 1st command, works only in bash!


# Run all unittests
#  Run with 'make -k' if you like to sweep through all of them, so
#  failure in one of them does not stop the full sweep
unittests: unittest-nonlabile unittest unittest-badexternals \
           unittest-optimization unittest-ca unittest-debug

te-%: build
	@echo -n "I: Testing example $*: "
	@MVPA_EXAMPLES_INTERACTIVE=no PYTHONPATH=.:$(PYTHONPATH) MVPA_MATPLOTLIB_BACKEND=agg \
	 $(PYTHON) doc/examples/$*.py >| temp-$@.log 2>&1 \
	 && echo "passed" \
	 || { echo "failed:"; cat temp-$@.log; rm -f temp-$@.log; exit 1; }
	@rm -f temp-$@.log

testexamples: te-svdclf te-smlr te-sensanas te-pylab_2d \
              te-curvefitting te-projections te-kerneldemo \
              te-erp_plot te-match_distribution te-permutation_test \
              te-searchlight_minimal te-smlr te-start_easy te-topo_plot \
              te-gpr te-gpr_model_selection0 te-mri_plot te-searchlight \
              te-clfs_examples

testdocstrings: dt-mvpa

dt-%: build
	@PYTHONPATH=.:$(PYTHONPATH) \
		MVPA_MATPLOTLIB_BACKEND=agg \
		MVPA_EXTERNALS_RAISE_EXCEPTION=off \
		MVPA_DATADB_ROOT=datadb \
		$(NOSETESTS) --with-doctest \
			$(shell git grep -l __docformat__ | grep '^mvpa' \
				| grep -v filter.py | grep -v channel.py | grep "$*")

tm-%: build
	@PYTHONPATH=.:$(TUT_DIR):$(CURDIR)/doc/examples:$(PYTHONPATH) \
		MVPA_MATPLOTLIB_BACKEND=agg \
		MVPA_LOCATION_TUTORIAL_DATA=$(TUT_DIR) \
		MVPA_DATADB_ROOT=datadb \
		$(NOSETESTS) --with-doctest --doctest-extension .rst \
	                 --doctest-tests doc/source/$*.rst

testmanual: build testdocstrings
	@echo "I: Testing code samples found in documentation"
	@PYTHONPATH=.:$(TUT_DIR):$(PYTHONPATH) \
		MVPA_MATPLOTLIB_BACKEND=agg \
		MVPA_LOCATION_TUTORIAL_DATA=$(TUT_DIR) \
		MVPA_DATADB_ROOT=datadb \
		$(NOSETESTS) -v --with-doctest --doctest-extension .rst \
		             --doctest-tests doc/source

testtutorial-%: build
	@echo "I: Testing code samples found in tutorial part $*"
	@PYTHONPATH=.:$(TUT_DIR):$(PYTHONPATH) \
		MVPA_MATPLOTLIB_BACKEND=agg \
		MVPA_LOCATION_TUTORIAL_DATA=$(TUT_DIR) \
		$(NOSETESTS) --with-doctest --doctest-extension .rst \
		             --doctest-tests doc/source/tutorial_$**.rst

# Test either all # alt references in tutorials are correct
# Just outputs filenames found missing -- doesn't fail the rule
TUTORIAL_RESDIR=tutorial_data/results
testtutorials-alt:
	@grep '# *alt' doc/source/tutorial*rst | \
	 sed -e "s/.*'\(.*\)'.*/\1/g" | \
	 while read f; do \
	  fs="$$(/bin/ls $(TUT_DIR)/results/$$f.* 2>/dev/null)"; \
	  [ -z "$$fs" ] && echo "$$f missing" || :; \
	 done

testdatadb: build
	@echo "I: Testing code samples on the dataset DB website"
	@PYTHONPATH=.:$(PYTHONPATH) \
		MVPA_MATPLOTLIB_BACKEND=agg \
		MVPA_DATADB_ROOT=datadb \
		$(NOSETESTS) --with-doctest --doctest-extension .rst \
		             --doctest-tests doc/source/datadb/*.rst

# Check if everything (with few exclusions) is imported in unitests is
# known to the mvpa.suite()
# XXX remove \|spam whenever clfs.spam gets available
testsuite:
	@echo "I: Running full testsuite"
	@tfile=`mktemp -u testsuiteXXXXXXX`; \
	 git grep -h '^\W*from mvpa.*import' mvpa/tests | \
	 grep -v '^\W*#' | \
	 sed -e 's/^.*from *\(mvpa[^ ]*\) im.*/from \1 import/g' | \
	 sort | uniq | \
	 grep -v -e 'mvpa\.base\.dochelpers' \
			 -e 'mvpa\.\(tests\|testing\|support\)' \
			 -e 'mvpa\.misc\.args' \
			 -e 'mvpa\.clfs\.\(libsvmc\|sg\|spam\)' \
	| while read i; do \
	 grep -q "^ *$$i" mvpa/suite.py || \
	 { echo "E: '$$i' is missing from mvpa.suite()"; touch "$$tfile"; }; \
	 done; \
	 [ -f "$$tfile" ] && { rm -f "$$tfile"; exit 1; } || :

# Check if links to api/ within documentation are broken.
testapiref:
	@echo "I: epydoc support is depricated -- so, nothing to test"
# testapiref: apidoc
# 	@for tf in doc/*.rst; do \
# 	 out=$$(for f in `grep api/mvpa $$tf | sed -e 's|.*\(api/mvpa.*html\).*|\1|g' `; do \
# 	  ff=build/html/$$f; [ ! -f $$ff ] && echo "E: $$f missing!"; done; ); \
# 	 [ "x$$out" == "x" ] || echo -e "$$tf:\n$$out"; done

# Check if there is no WARNINGs from sphinx
testsphinx: htmldoc
	{ grep -A1 system-message build/html/*html build/html/*/*html && exit 1 || exit 0 ; }

# Check if stored cfg after whole suite is imported is safe to be
# reloaded
testcfg: build
	@echo "I: Running test to check that stored configuration is acceptable."
	-@rm -f pymvpa.cfg
	@PYTHONPATH=.:$(PYTHONPATH)	$(PYTHON) -c 'from mvpa.suite import *; cfg.save("pymvpa.cfg");'
	@PYTHONPATH=.:$(PYTHONPATH)	$(PYTHON) -c 'from mvpa.suite import *;'
	@echo "+I: Run non-labile testing to verify safety of stored configuration"
	@PYTHONPATH=.:$(PYTHONPATH) MVPA_TESTS_LABILE=no $(PYTHON) mvpa/tests/__init__.py
	@echo "+I: Check all known dependencies and store them"
	@PYTHONPATH=.:$(PYTHONPATH)	$(PYTHON) -c \
	  'from mvpa.suite import *; mvpa.base.externals.test_all_dependencies(force=False); cfg.save("pymvpa.cfg");'
	@echo "+I: Run non-labile testing to verify safety of stored configuration"
	@PYTHONPATH=.:$(PYTHONPATH) MVPA_TESTS_LABILE=no $(PYTHON) mvpa/tests/__init__.py
	-@rm -f pymvpa.cfg

testourcfg: build
	@echo "+I: Run non-labile testing to verify safety of shipped configuration"
	@PYTHONPATH=.:$(PYTHONPATH) MVPACONFIG=doc/examples/pymvpa.cfg MVPA_TESTS_LABILE=no $(PYTHON) mvpa/tests/__init__.py


test: unittests testmanual testsuite testexamples testcfg testourcfg

# Target to be called after some major refactoring
# It skips some flavors of unittests
testrefactor: unittest testmanual testsuite testexamples

coverage: $(COVERAGE_REPORT)
$(COVERAGE_REPORT): build
	@echo "I: Generating coverage data and report. Takes awhile. No progress output."
	@{ \
	  export PYTHONPATH=.:$(PYTHONPATH) MVPA_DEBUG=.* MVPA_DEBUG_METRICS=ALL; \
	  python-coverage -x mvpa/tests/__init__.py >/dev/null 2>&1; \
	  python-coverage -r -i -o /usr,/var >| $(COVERAGE_REPORT); \
	  grep -v '100%$$' $(COVERAGE_REPORT); \
	  python-coverage -a -i -o /usr,/var ; }


#
# Sources
#

pylint:
	pylint -e --rcfile doc/misc/pylintrc mvpa

#
# Generate new source distribution
# (not to be run by users, depends on debian environment)

# Check either everything was committed
check-nodirty:
	# Need to run in clean tree. If fails: commit or clean first
	[ "x$$(git diff)" = "x" ]
# || $(error "")

check-debian:
	# Need to run in a Debian packaging branch
	[ -d debian ]

check-debian-version: check-debian
	# Does debian version correspond to setup.py version?
	[ "$(DEBCHANGELOG_VERSION)" = "$(SETUPPY_VERSION)" ]

embed-dev-version: check-nodirty
	# change upstream version
	sed -i -e "s/$(SETUPPY_VERSION)/$(DEV_VERSION)/g" setup.py mvpa/__init__.py
	# change package name
	sed -i -e "s/= 'pymvpa',/= 'pymvpa-snapshot',/g" setup.py

deb-dev-autochangelog: check-debian
	# removed -snapshot from pkg name for now
	$(MAKE) check-debian-version || \
		dch --newversion $(DEV_VERSION)-1 --package pymvpa-snapshot \
		 --allow-lower-version "PyMVPA development snapshot."

deb-mergedev:
	git merge --no-commit origin/dist/debian/dev

orig-src:
	git archive --format=tar --prefix=pymvpa-$(SETUPPY_VERSION)/ HEAD | \
		gzip -9 > pymvpa_$(SETUPPY_VERSION).orig.tar.gz

devel-src: check-nodirty
	-rm -rf dist
	git clone -l . dist/pymvpa-snapshot
	#RELEASE_CODE=-snapshot
	RELEASE_VERSION=$(DEV_VERSION) \
	  $(MAKE) -C dist/pymvpa-snapshot -f ../../Makefile embed-dev-version orig-src
	mv dist/*tar.gz ..
	rm -rf dist

devel-dsc: check-nodirty
	-rm -rf dist
	git clone -l . dist/pymvpa-snapshot
	#RELEASE_CODE=-snapshot
	RELEASE_VERSION=$(DEV_VERSION) \
	  $(MAKE) -C dist/pymvpa-snapshot -f ../../Makefile embed-dev-version orig-src deb-mergedev deb-dev-autochangelog
	# create the dsc -- NOT using deb-src since it would clean the hell first
	cd dist && dpkg-source -i'\.(gbp.conf|git\.*)' -b pymvpa-snapshot
	mv dist/*.gz dist/*dsc ..
	rm -rf dist

# make Debian source package
# # DO NOT depend on orig-src here as it would generate a source tarball in a
# Debian branch and might miss patches!
deb-src: check-debian distclean
	cd .. && dpkg-source -i'\.(gbp.conf|git\.*)' -b $(CURDIR)


bdist_rpm: 3rd
	$(PYTHON) setup.py bdist_rpm \
	  --doc-files "doc data" \
	  --packager "PyMVPA Authors <pkg-exppsy-pymvpa@lists.alioth.debian.org>" \
	  --vendor "PyMVPA Authors <pkg-exppsy-pymvpa@lists.alioth.debian.org>"

# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg: 3rd
	$(PYTHON) tools/mpkg_wrapper.py setup.py build_ext
	$(PYTHON) tools/mpkg_wrapper.py setup.py install


#
# Data
#

fetch-data:
	@echo "I: fetching data from datadb"
	@rsync $(RSYNC_OPTS) $(DATA_URI)/tutorial_data $(DATA_URI)/mnist \
		$(DATA_URI)/face_inversion_demo datadb
	@for ds in datadb/*; do \
		echo " I: looking at $$ds"; \
		cd $(CURDIR)/$${ds} && \
		md5sum -c MD5SUMS && \
		tbs="$$(/bin/ls *.tar.gz 2>/dev/null)" && \
		[ ! -z "$$tbs" ] && \
		for tb in $${tbs}; do \
		 fn=$${tb%.tar.gz}; dn=$${fn%-*}; \
		 [ ! -d $$dn ] && tar xzf $$tb || : ;\
		done; \
	done

# Various other data which might be sensitive and not distribu
fetch-data-nonfree: fetch-data-nonfree-stamp
fetch-data-nonfree-stamp:
	@mkdir -p temp
# clean up previous location to make sure we don't have it
	@rm -f data/nonfree/audio/Peter_Nalitch-Guitar.mp3
# remove directories which should be bogus now
	@rmdir data/nonfree/audio data/nonfree 2>/dev/null || :
	rsync $(RSYNC_OPTS) dev.pymvpa.org:/home/data/nonfree temp/ && touch $@


#
# Various sugarings (e.g. swarm)
#

AUDIO_TRACK=temp/nonfree/audio/Peter_Nalitch-Guitar.mp3

# With permission of the author, we can use Gitar for our visual history
$(AUDIO_TRACK): fetch-data-nonfree

# Nice visual git log
# Requires: sun-java5-jdk, ffmpeg, ant
codeswarm: $(SWARM_DIR)/pymvpa-codeswarm.flv
$(SWARM_DIR)/frames: $(SWARMTOOL_DIR) $(SWARM_DIR)/git.xml
	@echo "I: Visualizing git history using codeswarm"
	@mkdir -p $(SWARM_DIR)/frames
	cd $(SWARMTOOL_DIR) && ./run.sh ../../doc/misc/codeswarm.config

$(SWARM_DIR)/pymvpa-codeswarm.flv: $(SWARM_DIR)/frames $(AUDIO_TRACK)
	@echo "I: Generating codeswarm video"
	@cd $(SWARM_DIR) && \
     ffmpeg -r $$(echo "scale=2; $$(ls -1 frames/ |wc -l) / 154" | bc) -f image2 \
      -i frames/code_swarm-%05d.png -r 15 -b 250k \
      -i ../../$(AUDIO_TRACK) -ar 22050 -ab 128k -acodec libmp3lame \
      -y -ac 2 pymvpa-codeswarm.flv

$(SWARM_DIR)/git.log: Makefile
	@echo "I: Dumping git log in codeswarm preferred format"
	@mkdir -p $(SWARM_DIR)
	@git log --name-status --all \
     --pretty=format:'%n------------------------------------------------------------------------%nr%h | %an | %ai (%aD) | x lines%nChanged paths:' | \
     perl -pe 's/Ingo .*d \|/Ingo Fruend |/' | \
     sed -e 's,Yaroslav.*Halchenko,Yaroslav O. Halchenko,g' \
         -e 's,gorlins,Scott,g' -e 's,Scott Gorlin,Scott,g' -e 's,Scott,Scott Gorlin,g' \
         -e 's,hanke,Michael Hanke,g' \
		 -e 's,swaroop,Swaroop Guntupalli,g' \
         -e 's,Per.*Sederberg,Per B. Sederberg,g' \
         -e 's,Neukom Institute,James M. Hughes,g' >| $@




$(SWARM_DIR)/git.xml: $(SWARMTOOL_DIR)/run.sh $(SWARM_DIR)/git.log
	@$(PYTHON) $(SWARMTOOL_DIR)/convert_logs/convert_logs.py \
	 -g $(SWARM_DIR)/git.log -o $(SWARM_DIR)/git.xml

$(SWARMTOOL_DIR)/run.sh:
	@echo "I: Checking out codeswarm tool source code"
	@svn checkout http://codeswarm.googlecode.com/svn/trunk/ $(SWARMTOOL_DIR)


upload-codeswarm: codeswarm
	rsync -rzhvp --delete --chmod=Dg+s,g+rw,o+r $(SWARM_DIR)/*.flv $(WWW_UPLOAD_URI)/files/


#
# Trailer
#

.PHONY: fetch-data deb-src orig-src pylint apidoc pdfdoc htmldoc doc manual \
        all profile website fetch-data-misc upload-website \
        test testsuite testmanual testapiref testexamples testrefactor \
        testcfg testourcfg \
        unittest unittest-debug unittest-optimization unittest-nonlabile \
        unittest-badexternals unittests \
        distclean debian-clean check-nodirty check-debian check-debian-version \
        handbook codeswarm upload-codeswarm coverage pics
