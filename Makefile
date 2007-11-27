PROFILE_FILE=tests/main.pstats

all:

distclean:
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
	-@rm -rf doc/api/html doc/*.html
	-@cd doc/manual && rm -f *.log *.aux *.pdf *.backup *.out *.toc


manual:
	cd doc/manual && pdflatex manual.tex && pdflatex manual.tex

apidoc: $(PROFILE_FILE)
	epydoc --config doc/api/epydoc.conf

$(PROFILE_FILE): tests/main.py
	@cd tests && PYTHONPATH=.. ../tools/profile -K  -O ../$(PROFILE_FILE) main.py

doc: apidoc
	@rst2html doc/NOTES.coding doc/NOTES.coding.html

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

fetch-data:
	rsync -avz apsy.gse.uni-magdeburg.de:/home/hanke/public_html/software/pymvpa/data .

.PHONY: fetch-data orig-src pylint apidoc doc manual
