PROFILE_FILE=tests/main.pstats
COVERAGE_REPORT=coverage
HTML_DIR=build/html
PDF_DIR=build/pdf

PYVER := $(shell pyversions -vd)
ARCH := $(shell uname -m)


#
# Building
#

all: build doc

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
	-@rm -f MANIFEST Changelog
	-@rm -f mvpa/clfs/libsvm/*.{c,so} \
		mvpa/clfs/libsvm/svmc.py \
		mvpa/clfs/libsvm/svmc_wrap.cpp \
		tests/*.{prof,pstats,kcache,coverage} $(PROFILE_FILE) $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*,cover' \
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
# Misc pattern rules
#

# convert rsT documentation in doc/* to HTML.
rst2html-%:
	if [ ! -d $(HTML_DIR) ]; then mkdir -p $(HTML_DIR); fi
	for f in doc/$*/*.txt; do rst2html --date --strict --stylesheet=pymvpa.css \
		--link-stylesheet $${f} $(HTML_DIR)/$$(basename $${f%%.txt}.html); \
	done
	cp doc/misc/*.css $(HTML_DIR)
	# copy common images
	cp -r doc/misc/pics $(HTML_DIR)
	# copy local images, but ignore if there are none
	-cp -r doc/$*/pics $(HTML_DIR)

# convert rsT documentation in doc/* to PDF.
rst2pdf-%:
	if [ ! -d $(PDF_DIR) ]; then mkdir -p $(PDF_DIR); fi
	for f in doc/$*/*.txt; do \
		rst2latex --documentclass=scrartcl \
		          --use-latex-citations \
				  --strict \
				  --use-latex-footnotes \
				  --stylesheet ../../doc/misc/style.tex \
				  $${f} $(PDF_DIR)/$$(basename $${f%%.txt}.tex); \
		done
	cd $(PDF_DIR) && for f in *.tex; do pdflatex $${f}; done
# need to clean tex files or the will be rebuild again
	cd $(PDF_DIR) && rm *.tex


#
# Website
#
# put everything in one directory. Might be undesired if there are
# filename clashes. But given the website will be/should be simply, it
# might 'just work'.
website: rst2html-website rst2html-devguide rst2html-manual apidoc \
         rst2pdf-manual
	-@mkdir $(HTML_DIR)/files
	cp $(PDF_DIR)/*.pdf $(HTML_DIR)/files

printables: rst2pdf-manual

upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(HTML_DIR)/* alioth.debian.org:/home/groups/pkg-exppsy/htdocs/pymvpa/

#
# Documentation
#

doc: apidoc rst2html-devguide rst2html-manual

manual: rst2html-manual
	#cd doc/manual && pdflatex manual.tex && pdflatex manual.tex

apidoc: apidoc-stamp
apidoc-stamp: $(PROFILE_FILE)
	mkdir -p $(HTML_DIR)/api
	epydoc --config doc/api/epydoc.conf
	touch $@

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
	# the debian changelog is also the upstream changelog
	cp debian/changelog Changelog

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
