distclean:
	rm -rf doc/api/html
	-cd doc/manual && rm *.log *.aux *.pdf *.backup *.out *.toc

manual:
	cd doc/manual && pdflatex manual.tex && pdflatex manual.tex
