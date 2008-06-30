PROFILE_FILE=tests/main.pstats
COVERAGE_REPORT=coverage
HTML_DIR=build/html
APIDOC_DIR=$(HTML_DIR)/api
PDF_DIR=build/pdf
LATEX_DIR=build/latex
WWW_DIR=build/website

# should be made conditional, as pyversions id Debian specific
PYVER := $(shell pyversions -vd)
ARCH := $(shell uname -m)


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
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d"; \
     done
	touch $@


debian-build:
# reuse is better than duplication (yoh)
	debian/rules build


build: build-stamp
build-stamp: 3rd
	python setup.py config --noisy
	python setup.py build_ext
	python setup.py build_py
# to overcome the issue of not-installed svmc.so
	for ext in svm smlr; do \
		ln -sf ../../../build/lib.linux-$(ARCH)-$(PYVER)/mvpa/clfs/lib$$ext/$${ext}c.so \
		mvpa/clfs/lib$$ext/; done
	touch $@


#
# Cleaning
#

# Full clean
clean:
# clean 3rd party pieces
	find 3rd -mindepth 1 -maxdepth 1  -type d | \
	 while read d; do \
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d" clean; \
     done

# if we are on debian system - we might have left-overs from build
	-@$(MAKE) debian-clean
# if not on debian -- just distclean
	-@$(MAKE) distclean

distclean:
	-@rm -f MANIFEST
	-@rm -f mvpa/clfs/lib*/*.so \
        mvpa/clfs/lib*/*_wrap.* \
		mvpa/clfs/lib*/*c.py \
		tests/*.{prof,pstats,kcache} $(PROFILE_FILE) $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '*.[ao]' -o -iname '*.gch' \
		 -o -iname '#*#' | xargs -l10 rm -f
	-@rm -rf build
	-@rm -rf dist
	-@rm build-stamp apidoc-stamp website-stamp pdfdoc-stamp 3rd-stamp


debian-clean:
# remove stamps for builds since state is not really built any longer
	-fakeroot debian/rules clean

#
# Documentation
#
doc: website

htmldoc:
	cd doc && $(MAKE) html

pdfdoc: pdfdoc-stamp
pdfdoc-stamp:
	cd doc && $(MAKE) latex
	cd $(LATEX_DIR) && $(MAKE) all-pdf
	touch $@

apidoc: apidoc-stamp
apidoc-stamp: build
# Disabled profiling for now, it consumes huge amounts of memory, so I doubt
# that all buildds can do it. In theory it would only be done on a single
# developer machine, because it is only necessary for the arch-all package,
# but e.g. dpkg-buildpackage runs the indep target anyway -- not sure about
# the buildds, though.
#apidoc-stamp: profile
	mkdir -p $(HTML_DIR)/api
	epydoc --config doc/api/epydoc.conf
	touch $@

website: website-stamp
website-stamp: mkdir-WWW_DIR apidoc htmldoc pdfdoc
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	cp $(LATEX_DIR)/*.pdf $(WWW_DIR)
	touch $@

upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(WWW_DIR)/* alioth.debian.org:/home/groups/pkg-exppsy/htdocs/pymvpa/

upload-htmldoc: htmldoc
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(HTML_DIR)/* alioth.debian.org:/home/groups/pkg-exppsy/htdocs/pymvpa/


# this takes some minutes !!
profile: build tests/main.py
	@cd tests && PYTHONPATH=.. ../tools/profile -K  -O ../$(PROFILE_FILE) main.py

ut-%: build
	@cd tests && PYTHONPATH=.. python test_$*.py

unittest: build
	@cd tests && PYTHONPATH=.. python main.py

# Runs unittests in few additional modes:
# * with optimization on -- helps to catch unconditional debug calls
# * with all debug ids and some metrics (crossplatform ones) on.
#   That does:
#     additional checking,
#     debug() calls validation, etc
unittests: unittest
	@cd tests && PYTHONPATH=.. python -O main.py
	@echo "Running unittests with debug output. No progress output."
	@cd tests && \
      PYTHONPATH=.. MVPA_DEBUG=.* MVPA_DEBUG_METRICS=ALL \
       python main.py 2>&1 \
       |  sed -n -e '/^[=-]\{60,\}$$/,/^\(MVPA_SEED=\|OK\)/p'

te-%: build
	MVPA_EXAMPLES_INTERACTIVE=no PYTHONPATH=. python doc/examples/$*.py

testexamples: te-svdclf te-smlr te-searchlight_2d te-sensanas te-pylab_2d \
              te-curvefitting te-projections

tm-%: build
	PYTHONPATH=. nosetests --with-doctest --doctest-extension .txt \
	                       --doctest-tests doc/$*.txt

testmanual: build
	PYTHONPATH=. nosetests --with-doctest --doctest-extension .txt \
	                       --doctest-tests doc/

# Check if everything imported in unitests is known to the
# mvpa.suite()
testsuite:
	@git grep -h '^\W*from mvpa.*import' tests | \
	 sed -e 's/^\W*from *\(mvpa[^ ]*\) im.*/from \1 import/g' | \
	 sort | uniq | \
	while read i; do \
	 grep -q "^ *$$i" mvpa/suite.py || \
	 { echo "'$$i' is missing from mvpa.suite()"; exit 1; }; \
	 done

# Check if links to api/ within documentation are broken.
testapiref: apidoc
	@for tf in doc/*.txt; do \
	 out=$$(for f in `grep api/mvpa $$tf | sed -e 's|.*\(api/mvpa.*html\).*|\1|g' `; do \
	  ff=build/html/$$f; [ ! -f $$ff ] && echo " $$f missing!"; done; ); \
	 [ "x$$out" == "x" ] || echo -e "$$tf:\n$$out"; done

test: unittests testmanual testsuite testapiref testexamples

$(COVERAGE_REPORT): build
	@cd tests && { \
	  export PYTHONPATH=..; \
	  python-coverage -x main.py; \
	  python-coverage -r -i -o /usr >| ../$(COVERAGE_REPORT); \
	  grep -v '100%$$' ../$(COVERAGE_REPORT); \
	  python-coverage -a -i -o /usr; }


#
# Sources
#

pylint:
	pylint --rcfile doc/misc/pylintrc mvpa

#
# Generate new source distribution
# (not to be run by users, depends on debian environment)

orig-src: distclean debian-clean 
	# clean existing dist dir first to have a single source tarball to process
	-rm -rf dist

	if [ -f debian/changelog ]; then \
		if [ ! "$$(dpkg-parsechangelog | egrep ^Version | cut -d ' ' -f 2,2 | cut -d '-' -f 1,1)" == "$$(python setup.py -V)" ]; then \
				printf "WARNING: Changelog version does not match tarball version!\n" ;\
				exit 1; \
		fi \
	fi
	# let python create the source tarball
	# enable libsvm to get all sources!
	python setup.py sdist --formats=gztar --with-libsvm
	# rename to proper Debian orig source tarball and move upwards
	# to keep it out of the Debian diff
	file=$$(ls -1 dist); ver=$${file%*.tar.gz}; ver=$${ver#pymvpa-*}; mv dist/$$file ../pymvpa_$$ver.orig.tar.gz
	# clean leftover
	rm MANIFEST

# make Debian source package
# # DO NOT depend on orig-src here as it would generate a source tarball in a
# Debian branch and might miss patches!
debsrc:
	cd .. && dpkg-source -i'\.(gbp.conf|git\.*)' -b $(CURDIR)


bdist_rpm: 3rd
	python setup.py bdist_rpm --with-libsvm \
	  --doc-files "doc data" \
	  --packager "PyMVPA Authors <pkg-exppsy-pymvpa@lists.alioth.debian.org>" \
	  --vendor "PyMVPA Authors <pkg-exppsy-pymvpa@lists.alioth.debian.org>"


#
# Data
#

fetch-data:
	rsync -avz apsy.gse.uni-magdeburg.de:/home/hanke/public_html/software/pymvpa/data .

#
# Trailer
#

.PHONY: fetch-data debsrc orig-src pylint apidoc pdfdoc htmldoc doc manual profile website fetch-data upload-website test testsuite testmanual testapiref testexamples distclean debian-clean all unittest unittests
