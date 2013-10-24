all:: pymvpa_logo.jpg download_icon.jpg tutorial_icon.jpg \
      documentation_icon.jpg support_icon.jpg movie_icon.jpg \
      pymvpa_icon.jpg \
      pics

# where the rendered images should go
TARGETDIR = ../source/pics
STATICDIR = ../source/_static

# special case of icon svg
ICON_SVG_BASE = pymvpa_icons
ICON_HEIGHT = $(shell inkscape -z -f $(ICON_SVG_BASE).svg -H)
ICON_WIDTH = $(shell inkscape -z -f $(ICON_SVG_BASE).svg -W)
ICON_NUMBER = 6

LOGO_HEIGHT = 70

# PICS_IGNORE must contain a rule for grep
PICS_IGNORE ?= "$(ICON_SVG_BASE)"

# For every .svg we must have a pdf
PDFPICS=$(shell find . -iname \*svg \
	| sed -e 's/svg$$/pdf/g' -e 's/\([^\]\)\([ \t:]\)/\1\\\\\2/g' \
	| grep -v -e $(PICS_IGNORE) )
# and a png
PNGPICS=$(shell find . -iname \*svg \
	| sed -e 's/svg$$/png/g' -e 's/\([^\]\)\([ \t:]\)/\1\\\\\2/g' \
	| grep -v -e $(PICS_IGNORE) )
LOGOPICS=ccn-logo_h$(LOGO_HEIGHT).png uni-md-logo_h$(LOGO_HEIGHT).png dartmouth-logo_h$(LOGO_HEIGHT).png

pics: $(PDFPICS) $(PNGPICS) $(LOGOPICS)

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
		rm -f $(STATICDIR)/$${p%*.svg}.jpg; \
	done
	-rm -f $(STATICDIR)/*_icon.jpg *_icon.*

.PHONY: ignore-%
ignore-%:
	@grep -q "^$*$$" $(TARGETDIR)/.gitignore || { \
	  echo "$*" >> $(TARGETDIR)/.gitignore; echo "Ignore $*"; }
	@grep -q "^$*$$" $(STATICDIR)/.gitignore || { \
	  echo "$*" >> $(STATICDIR)/.gitignore; echo "Ignore $*"; }
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

%.jpg: %.png ignore-%.jpg
	@echo "Converting $*.png to $*.jpg"
	@convert -flatten -resize x130 $*.png $*.jpg
	@cp "$@" $(STATICDIR)

# logo sized versions
%_h70.png: %.png ignore-%_h70.png
	@echo "Generating downscaled $@"
	@convert -geometry x70 "$<" "$@"

# special cases
download_icon.png: $(ICON_SVG_BASE).svg
	inkscape -z -f $(ICON_SVG_BASE).svg -e $@ \
		--export-area=$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 0" | bc):0:$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 1" | bc):$(ICON_HEIGHT)
	$(MAKE) ignore-$@

tutorial_icon.png: $(ICON_SVG_BASE).svg
	inkscape -z -f $(ICON_SVG_BASE).svg -e $@ \
		--export-area=$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 1" | bc):0:$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 2" | bc):$(ICON_HEIGHT)
	$(MAKE) ignore-$@

documentation_icon.png: $(ICON_SVG_BASE).svg
	inkscape -z -f $(ICON_SVG_BASE).svg -e $@ \
		--export-area=$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 2" | bc):0:$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 3" | bc):$(ICON_HEIGHT)
	$(MAKE) ignore-$@

support_icon.png: $(ICON_SVG_BASE).svg
	inkscape -z -f $(ICON_SVG_BASE).svg -e $@ \
		--export-area=$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 3" | bc):0:$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 4" | bc):$(ICON_HEIGHT)
	$(MAKE) ignore-$@

movie_icon.png: $(ICON_SVG_BASE).svg
	inkscape -z -f $(ICON_SVG_BASE).svg -e $@ \
		--export-area=$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 4" | bc):0:$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 5" | bc):$(ICON_HEIGHT)
	$(MAKE) ignore-$@

pymvpa_icon.png: $(ICON_SVG_BASE).svg
	inkscape -z -f $(ICON_SVG_BASE).svg -e $@ \
		--export-area=$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 5" | bc):$(shell echo "$(ICON_HEIGHT) * 0.463/4.211" | bc):$(shell echo "$(ICON_WIDTH) / $(ICON_NUMBER) * 6" | bc):$(ICON_HEIGHT)
	$(MAKE) ignore-$@

.PHONY: all pics
