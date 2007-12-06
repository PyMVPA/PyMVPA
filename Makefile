PROFILE_FILE=tests/main.pstats

PYVER := $(shell pyversions -vd)
ARCH := $(shell uname -m)

all:

build:
# reuse is better than duplication (yoh)
	debian/rules build

# to overcome the issue of not-installed svmc.so
	ln -sf ../../../build/lib.linux-$(ARCH)-$(PYVER)/mvpa/clf/libsvm/svmc.so mvpa/clf/libsvm/

distclean: clean
	-@rm -f MANIFEST Changelog
	-@rm -f mvpa/clf/libsvm/*.{c,so} \
		mvpa/clf/libsvm/svmc.py \
		mvpa/clf/libsvm/svmc_wrap.cpp \
		tests/*.{prof,pstats,kcache} $(PROFILE_FILE)
	@find . -name '*.pyo' \
		 -o -name '*.pyc' \
		 -o -iname '*~' \
		 -o -iname '#*#' | xargs -l10 rm -f
	-@rm -rf build
	-@rm -rf dist
	-@rm -rf doc/{,devguide/}*.html
# remove all generated HTML stuff
	@find doc -mindepth 2 -maxdepth 2 -type d -name 'html' -print -exec rm -rf {} \;

clean:
# remove stamps for builds since state is not really built any longer
	-fakeroot debian/rules clean

#
# Misc pattern rules
#

# convert rsT documentation in doc/* to HTML. In the corresponding directory
# below doc/ a subdir html/ is created that contains the converted output.
rst2html-%:
	if [ ! -d doc/$*/html ]; then mkdir -p doc/$*/html; fi
	cd doc/$* && \
		for f in *.txt; do rst2html --date --strict --stylesheet=pymvpa.css \
		    --link-stylesheet $${f} html/$${f%%.txt}.html; \
		done
	cp doc/misc/*.css doc/$*/html
	# copy common images
	cp -r doc/misc/pics doc/$*/html
	# copy local images, but ignore if there are none
	-cp -r doc/$*/pics doc/$*/html

#
# Website
#

website: rst2html-website rst2html-devguide
	# put everything in one directory. Might be undesired if there are
	# filename clashes. But given the website will be/should be simply, it
	# might 'just work'.
	cp -r doc/devguide/html/* doc/website/html/

upload-website: website
	scp -r doc/website/html/* alioth:/home/groups/pkg-exppsy/htdocs/pymvpa

#
# Documentation
#

doc: apidoc rst2html-devguide rst2html-manual

manual:
	cd doc/manual && pdflatex manual.tex && pdflatex manual.tex

apidoc: $(PROFILE_FILE)
	epydoc --config doc/api/epydoc.conf

$(PROFILE_FILE): build tests/main.py
	@cd tests && PYTHONPATH=.. ../tools/profile -K  -O ../$(PROFILE_FILE) main.py

#
# Sources
#

pylint:
	pylint --rcfile doc/misc/pylintrc mvpa

orig-src: distclean 
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

.PHONY: fetch-data orig-src pylint apidoc doc manual build
