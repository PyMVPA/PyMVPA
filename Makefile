PROFILE_FILE=tests/main.pstats
COVERAGE_REPORT=coverage
HTML_DIR=build/html
APIDOC_DIR=$(HTML_DIR)/api
PDF_DIR=build/pdf
LATEX_DIR=build/latex
WWW_DIR=build/website
SWARM_DIR=build/swarm

SWARMTOOL_DIR=tools/codeswarm
SWARMTOOL_DIRFULL=$(CURDIR)/$(SWARMTOOL_DIR)

# should be made conditional, as pyversions id Debian specific
#PYVER := $(shell pyversions -vd)
# try generic variant instead
PYVER := $(shell python -V 2>&1 | cut -d ' ' -f 2,2 | cut -d '.' -f 1,2)
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
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d" || :; \
     done
	touch $@


debian-build:
# reuse is better than duplication (yoh)
	debian/rules build


build: build-stamp
build-stamp: 3rd
	python setup.py config --noisy --with-libsvm
	python setup.py build --with-libsvm
# to overcome the issue of not-installed svmc.so
	for ext in _svmc smlrc; do \
		ln -sf ../../../build/lib.linux-$(ARCH)-$(PYVER)/mvpa/clfs/lib$${ext#_*}/$${ext}.so \
		mvpa/clfs/lib$${ext#_*}/; \
		ln -sf ../../../build/lib.linux-$(ARCH)-$(PYVER)/mvpa/clfs/lib$${ext#_*}/$${ext}.so \
		mvpa/clfs/lib$${ext#_*}/$${ext}.dylib; \
		done
	touch $@


#
# Cleaning
#

# Full clean
clean:
# clean 3rd party pieces
	find 3rd -mindepth 1 -maxdepth 1  -type d | \
	 while read d; do \
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d" clean || : ; \
     done
# clean tools
	$(MAKE) -C tools clean

# if we are on debian system - we might have left-overs from build
	-@$(MAKE) debian-clean
# if not on debian -- just distclean
	-@$(MAKE) distclean

distclean:
	-@rm -rf doc/modref doc/ex
	-@rm -f MANIFEST
	-@rm -f mvpa/clfs/lib*/*.so \
		mvpa/clfs/lib*/*.dylib \
		mvpa/clfs/lib*/*_wrap.* \
		mvpa/clfs/lib*/*c.py \
		tests/*.{prof,pstats,kcache} $(PROFILE_FILE) $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '*.[ao]' -o -iname '*.gch' \
		 -o -iname '#*#' | xargs -L 10 rm -f
	-@rm -rf build
	-@rm -rf dist
	-@rm build-stamp apidoc-stamp website-stamp pdfdoc-stamp 3rd-stamp \
		modref-templates-stamp examples2rst-stamp fetch-data-nonfree-stamp


debian-clean:
# remove stamps for builds since state is not really built any longer
	-fakeroot debian/rules clean

#
# Documentation
#
doc: website

references:
	tools/bib2rst_ref.py

htmldoc: modref-templates examples2rst build
	cd doc && MVPA_EXTERNALS_RAISE_EXCEPTION=off PYTHONPATH=.. $(MAKE) html
	cd build/html/modref && ln -sf ../_static
	cd build/html/ex && ln -sf ../_static
	cp doc/pics/history_splash.png build/html/_images/

pdfdoc: modref-templates examples2rst build pdfdoc-stamp
pdfdoc-stamp:
	cd doc && MVPA_EXTERNALS_RAISE_EXCEPTION=off PYTHONPATH=.. $(MAKE) latex
	cd $(LATEX_DIR) && $(MAKE) all-pdf
	touch $@

# Create a handy .pdf of the manual to be printed as a book
handbook: pdfdoc
	cd tools && $(MAKE) pdfbook
	tools/pdfbook -2 \
	 $(LATEX_DIR)/PyMVPA-Manual.pdf $(LATEX_DIR)/PyMVPA-Manual-Handbook.pdf

modref-templates: modref-templates-stamp
modref-templates-stamp:
	PYTHONPATH=. tools/build_modref_templates.py
	touch $@

examples2rst: examples2rst-stamp
examples2rst-stamp:
	tools/examples2rst.py
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
	LC_ALL=C MVPA_EPYDOC_WARNINGS=once tools/epydoc --config doc/api/epydoc.conf
	touch $@

website: website-stamp
website-stamp: mkdir-WWW_DIR apidoc htmldoc pdfdoc
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	cp $(LATEX_DIR)/*.pdf $(WWW_DIR)
	tools/sitemap.sh > $(WWW_DIR)/sitemap.xml
# main icon of the website
	cp doc/pics/favicon.png $(WWW_DIR)/_images/
# for those who do not care about <link> and just trying to download it
	cp doc/pics/favicon.png $(WWW_DIR)/favicon.ico
# provide robots.txt to minimize unnecessary traffic
	cp doc/_static/robots.txt $(WWW_DIR)/
	touch $@

upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(WWW_DIR)/* belka.rutgers.edu:/home/michael/www.pymvpa.org/pymvpa/

upload-htmldoc: htmldoc
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(HTML_DIR)/* belka.rutgers.edu:/home/michael/www.pymvpa.org/pymvpa/


# this takes some minutes !!
profile: build tests/main.py
	@cd tests && PYTHONPATH=.. ../tools/profile -K  -O ../$(PROFILE_FILE) main.py

ut-%: build
	@cd tests && PYTHONPATH=.. python test_$*.py

unittest: build
	@echo "I: Running unittests (without optimization nor debug output)"
	@cd tests && PYTHONPATH=.. python main.py

# test if PyMVPA is working if optional externals are missing
unittest-badexternals: build
	@echo "I: Running unittests under assumption of missing optional externals."
	@cd tests && PYTHONPATH=badexternals:.. python main.py 2>&1 \
	| grep -v -e 'WARNING: Known dependency' -e 'Please note: w' \
              -e 'WARNING:.*SMLR.* implementation'

# Runs unittests in few additional modes:
# * with optimization on -- helps to catch unconditional debug calls
# * with all debug ids and some metrics (crossplatform ones) on.
#   That does:
#     additional checking,
#     debug() calls validation, etc
unittests: unittest unittest-badexternals
	@cd tests && PYTHONPATH=.. python -O main.py
	@echo "I: Running unittests with debug output. No progress output."
	@cd tests && \
      PYTHONPATH=.. MVPA_DEBUG=.* MVPA_DEBUG_METRICS=ALL \
       python main.py 2>&1 \
       |  sed -n -e '/^[=-]\{60,\}$$/,/^\(MVPA_SEED=\|OK\)/p'

te-%: build
	@echo -n "I: Testing example $*: "
	@MVPA_EXAMPLES_INTERACTIVE=no PYTHONPATH=. python doc/examples/$*.py \
	 >| temp-$@.log 2>&1 \
	 && echo "passed" || { echo "failed:"; cat temp-$@.log; }
	@rm -f temp-$@.log

testexamples: te-svdclf te-smlr te-searchlight_2d te-sensanas te-pylab_2d \
              te-curvefitting te-projections te-kerneldemo

tm-%: build
	PYTHONPATH=. nosetests --with-doctest --doctest-extension .txt \
	                       --doctest-tests doc/$*.txt

testmanual: build
	PYTHONPATH=. nosetests --with-doctest --doctest-extension .txt \
	                       --doctest-tests doc/

# Check if everything imported in unitests is known to the
# mvpa.suite()
testsuite:
	@echo "I: Running full testsuite"
	@git grep -h '^\W*from mvpa.*import' tests | \
	 sed -e 's/^\W*from *\(mvpa[^ ]*\) im.*/from \1 import/g' | \
	 sort | uniq | \
	while read i; do \
	 grep -q "^ *$$i" mvpa/suite.py || \
	 { echo "E: '$$i' is missing from mvpa.suite()"; exit 1; }; \
	 done

# Check if links to api/ within documentation are broken.
testapiref: apidoc
	@for tf in doc/*.txt; do \
	 out=$$(for f in `grep api/mvpa $$tf | sed -e 's|.*\(api/mvpa.*html\).*|\1|g' `; do \
	  ff=build/html/$$f; [ ! -f $$ff ] && echo "E: $$f missing!"; done; ); \
	 [ "x$$out" == "x" ] || echo -e "$$tf:\n$$out"; done

test: unittests testmanual testsuite testapiref testexamples

$(COVERAGE_REPORT): build
	@echo "I: Generating coverage data and report. Takes awhile. No progress output."
	@cd tests && { \
	  export PYTHONPATH=.. MVPA_DEBUG=.* MVPA_DEBUG_METRICS=ALL; \
	  python-coverage -x main.py >/dev/null 2>&1; \
	  python-coverage -r -i -o /usr,/var >| ../$(COVERAGE_REPORT); \
	  grep -v '100%$$' ../$(COVERAGE_REPORT); \
	  python-coverage -a -i -o /usr,/var ; }


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

# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg: 3rd
	python tools/mpkg_wrapper.py setup.py build_ext
	python tools/mpkg_wrapper.py setup.py install


#
# Data
#

fetch-data:
	rsync -avz apsy.gse.uni-magdeburg.de:/home/hanke/public_html/software/pymvpa/data .

# Various other data which might be sensitive and not distribu
fetch-data-nonfree: fetch-data-nonfree-stamp
fetch-data-nonfree-stamp:
	rsync -avz dev.pymvpa.org:/home/data/nonfree data/ && touch $@
#
# Various sugarings
#
AUDIO_TRACK=data/nonfree/audio/Peter_Nalitch-Guitar.mp3

# With permission of the author, we can use Gitar for our visual history
$(AUDIO_TRACK): fetch-data-nonfree

# Nice visual git log
# Requires: sun-java5-jdk, ffmpeg, ant
codeswarm: $(SWARM_DIR)/pymvpa-codeswarm.flv
$(SWARM_DIR)/frames: $(SWARMTOOL_DIR) $(SWARM_DIR)/git.xml
	@echo "I: Visualizing git history using codeswarm"
	@mkdir -p $(SWARM_DIR)/frames
	cd $(SWARMTOOL_DIR) && ./run.sh ../../doc/misc/codeswarm.config

$(SWARM_DIR)/pymvpa-codeswarm.flv: $(SWARM_DIR)/frames
	@echo "I: Generating codeswarm video"
	@cd $(SWARM_DIR) && \
     ffmpeg -r $$(echo "scale=2; $$(ls -1 frames/ |wc -l) / 154" | bc) -f image2 \
      -i frames/code_swarm-%05d.png -r 15 -b 250k \
      -i ../../$(AUDIO_TRACK) -ar 22050 -ab 128k -acodec libmp3lame \
      -ac 2 pymvpa-codeswarm.flv

$(SWARM_DIR)/git.log:
	@echo "I: Dumping git log in codeswarm preferred format"
	@mkdir -p $(SWARM_DIR)
	@git-log --name-status \
     --pretty=format:'%n------------------------------------------------------------------------%nr%h | %ae | %ai (%aD) | x lines%nChanged paths:' | \
     sed -e 's,[a-z]*@onerussian.com,Yarik,g' \
         -e 's,\(michael\.*hanke@\(gmail.com\|mvpa1.dartmouth.edu\)\|neukom-data@neukom-data-desktop\.(none)\),Michael,g' \
         -e 's,\(per@parsec.Princeton.EDU\|per@sync.(none)\|psederberg@gmail.com\),Per,g' \
         -e 's,emanuele@relativita.com,Emanuele,g' \
         -e 's,Ingo.Fruend@gmail.com,Ingo,g' >| $@

$(SWARM_DIR)/git.xml: $(SWARMTOOL_DIR) $(SWARM_DIR)/git.log
	@python $(SWARMTOOL_DIR)/convert_logs/convert_logs.py \
	 -g $(SWARM_DIR)/git.log -o $(SWARM_DIR)/git.xml

$(SWARMTOOL_DIR):
	@echo "I: Checking out codeswarm tool source code"
	@svn checkout http://codeswarm.googlecode.com/svn/trunk/ $(SWARMTOOL_DIR)


upload-codeswarm: codeswarm
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(SWARM_DIR)/*.flv belka.rutgers.edu:/home/michael/www.pymvpa.org/pymvpa/files/

#
# Trailer
#

.PHONY: fetch-data debsrc orig-src pylint apidoc pdfdoc htmldoc doc manual profile website fetch-data-misc upload-website test testsuite testmanual testapiref testexamples distclean debian-clean all unittest unittests handbook codeswarm
