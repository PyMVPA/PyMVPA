TARGETS=pdfbook
TDIR=../build/tools

all: $(TARGETS)

pdfbook:
	mkdir -p $(TDIR)
	gcc pdfbook.c -o $(TDIR)/pdfbook

clean::
	rm -f $(TDIR)/pdfbook
