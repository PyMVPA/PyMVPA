all:

distclean:
	-rm MANIFEST Changelog
	-rm mvpa/clf/libsvm/*.{c,so} \
		mvpa/clf/libsvm/svmc.py \
		mvpa/clf/libsvm/svmc_wrap.cpp
	find . -name '*.pyc' -exec rm \{\} \;
	-rm -r build
	-rm -r dist
	-rm -rf doc/api/html
	-cd doc/manual && rm *.log *.aux *.pdf *.backup *.out *.toc


manual:
	cd doc/manual && pdflatex manual.tex && pdflatex manual.tex


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
