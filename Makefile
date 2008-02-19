PROFILE_FILE=tests/main.pstats
COVERAGE_REPORT=coverage
HTML_DIR=build/html
APIDOC_DIR=$(HTML_DIR)/api
PDF_DIR=build/pdf
WWW_DIR=build/website


PYVER := $(shell pyversions -vd)
ARCH := $(shell uname -m)

rst2html = rst2html --date --strict --stylesheet=pymvpa.css --link-stylesheet
rst2latex=rst2latex --documentclass=scrartcl \
					--use-latex-citations \
					--strict \
					--use-latex-footnotes \
					--stylesheet ../../doc/misc/style.tex


mkdir-%:
	if [ ! -d $($*) ]; then mkdir -p $($*); fi


#
# Building
#

all: build

debian-build:
# reuse is better than duplication (yoh)
	debian/rules build


build: build-stamp
build-stamp:
	python setup.py config --noisy
	python setup.py build_ext
	python setup.py build_py
# to overcome the issue of not-installed svmc.so
	ln -sf ../../../build/lib.linux-$(ARCH)-$(PYVER)/mvpa/clfs/libsvm/svmc.so \
		mvpa/clfs/libsvm/
	touch $@

#
# Cleaning
#

# Full clean
clean:
# if we are on debian system - we might have left-overs from build
	-@$(MAKE) debian-clean
# if not on debian -- just distclean
	-@$(MAKE) distclean

distclean:
	-@rm -f MANIFEST
	-@rm -f mvpa/clfs/libsvm/*.{c,so} \
		mvpa/clfs/libsvm/svmc.py \
		mvpa/clfs/libsvm/svmc_wrap.cpp \
		tests/*.{prof,pstats,kcache} $(PROFILE_FILE) $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '#*#' | xargs -l10 rm -f
	-@rm -rf build
	-@rm -rf dist
	-@rm build-stamp apidoc-stamp


debian-clean:
# remove stamps for builds since state is not really built any longer
	-fakeroot debian/rules clean

#
# Documentation
#
doc: website

htmlindex: mkdir-HTML_DIR
	$(rst2html) doc/index.txt $(HTML_DIR)/index.html

htmlchangelog: mkdir-HTML_DIR
	$(rst2html) Changelog $(HTML_DIR)/changelog.html

htmlmanual: mkdir-HTML_DIR
	$(rst2html) doc/manual.txt $(HTML_DIR)/manual.html
	# copy images and styles
	cp -r doc/misc/{*.css,pics} $(HTML_DIR)

htmldevguide: mkdir-HTML_DIR
	$(rst2html) doc/devguide.txt $(HTML_DIR)/devguide.html

pdfmanual: mkdir-PDF_DIR
	cat doc/manual.txt Changelog | $(rst2latex) > $(PDF_DIR)/manual.tex
	-cp -r doc/misc/pics $(PDF_DIR)
	# need to run twice to get cross-refs right
	cd $(PDF_DIR) && pdflatex manual.tex && pdflatex manual.tex

pdfdevguide: mkdir-PDF_DIR
	$(rst2latex) doc/devguide.txt $(PDF_DIR)/devguide.tex
	cd $(PDF_DIR) && pdflatex devguide.tex

printables: pdfmanual pdfdevguide

apidoc: apidoc-stamp
apidoc-stamp: build
# Disabled profiling for now, it consumes huge amounts of memory, so I doubt
# that all buildds can do it. In theory it would only be done on a single
# developer machine, because it is only necessary for the arch-all package,
# but e.g. dpkg-buildpackage runs the indep target anyway -- not sure about
# the buildds, though.
#apidoc-stamp: $(PROFILE_FILE)
	mkdir -p $(HTML_DIR)/api
	epydoc --config doc/api/epydoc.conf
	touch $@

website: mkdir-WWW_DIR htmlindex htmlmanual htmlchangelog \
         htmldevguide printables apidoc
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	cp $(PDF_DIR)/*.pdf $(WWW_DIR)

upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(WWW_DIR)/* alioth.debian.org:/home/groups/pkg-exppsy/htdocs/pymvpa/


# this takes some minutes !!
$(PROFILE_FILE): build tests/main.py
	@cd tests && PYTHONPATH=.. ../tools/profile -K  -O ../$(PROFILE_FILE) main.py

test-%: build
	@cd tests && PYTHONPATH=.. python test_$*.py

test: build
	@cd tests && PYTHONPATH=.. python main.py


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

	if [ ! "$$(dpkg-parsechangelog | egrep ^Version | cut -d ' ' -f 2,2 | cut -d '-' -f 1,1)" == "$$(python setup.py -V)" ]; then \
			printf "WARNING: Changelog version does not match tarball version!\n" ;\
			exit 1; \
	fi
	# let python create the source tarball
	python setup.py sdist --formats=gztar
	# rename to proper Debian orig source tarball and move upwards
	# to keep it out of the Debian diff
	file=$$(ls -1 dist); ver=$${file%*.tar.gz}; ver=$${ver#pymvpa-*}; mv dist/$$file ../pymvpa_$$ver.orig.tar.gz

#
# Data
#

fetch-data:
	rsync -avz apsy.gse.uni-magdeburg.de:/home/hanke/public_html/software/pymvpa/data .

#
# Trailer
#

.PHONY: fetch-data orig-src pylint apidoc doc manual
