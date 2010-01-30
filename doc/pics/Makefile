all:: pics pymvpa_logo.jpg

# where the rendered images should go
TARGETDIR = ../source/pics

# PICS_IGNORE must contain a rule for grep
PICS_IGNORE ?= "XXX"

# For every .svg we must have a pdf
PDFPICS=$(shell find . -iname \*svg \
	| sed -e 's/svg$$/pdf/g' -e 's/\([^\]\)\([ \t:]\)/\1\\\\\2/g' \
	| grep -v -e $(PICS_IGNORE) )
# and a png
PNGPICS=$(shell find . -iname \*svg \
	| sed -e 's/svg$$/png/g' -e 's/\([^\]\)\([ \t:]\)/\1\\\\\2/g' \
	| grep -v -e $(PICS_IGNORE) )

pics: $(PDFPICS) $(PNGPICS)

clean::
	for p in *.svg; do \
		rm -f $(TARGETDIR)/$${p%*.svg}.pdf; \
		rm -f $(TARGETDIR)/$${p%*.svg}.png; \
		rm -f $(TARGETDIR)/$${p%*.svg}.jpg; \
	done
	for p in *.svg; do \
		rm -f $${p%*.svg}.pdf; \
		rm -f $${p%*.svg}.png; \
		rm -f $${p%*.svg}.jpg; \
	done

.PHONY: ignore-%
ignore-%:
	@grep -q "^$*$$" $(TARGETDIR)/.gitignore || { \
	  echo "$*" >> $(TARGETDIR)/.gitignore; echo "Ignore $@"; }
	@grep -q "^$*$$" .gitignore || { echo "$*" >> .gitignore; }

#
# Inkscape rendered figures
#
%.pdf: %.svg ignore-%.pdf
	@echo "Rendering $@"
	@inkscape -z -f "$<" -A "$@" --export-dpi=300
	@cp "$@" $(TARGETDIR)

%.png: %.svg ignore-%.png
	@echo "Rendering $@"
	@inkscape -z -f "$<" -e "$@" --export-width=500
	@cp "$@" $(TARGETDIR)

pymvpa_logo.jpg: pymvpa_logo.png ignore-pymvpa_logo.jpg
	@echo "Converting logo to JPG"
	@convert -flatten -resize x130 pymvpa_logo.png pymvpa_logo.jpg
	@cp pymvpa_logo.jpg $(TARGETDIR)



.PHONY: all pics
