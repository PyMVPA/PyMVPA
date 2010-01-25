all:: pics

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
	done
	for p in *.svg; do \
		rm -f $${p%*.svg}.pdf; \
		rm -f $${p%*.svg}.png; \
	done

.PHONY: ignore-%
ignore-%:
	@grep -q "^$*$$" $(TARGETDIR)/.gitignore || { \
	  echo "$*" >> $(TARGETDIR)/.gitignore; echo "Ignore $@"; }
	@grep -q "^$*$$" .gitignore || { echo "$*" >> .gitignore; }

#
# Inkscape rendered figures
#
.PHONY: %.pdf
%.pdf: %.svg ignore-%.pdf
	@echo "Rendering $@"
	@inkscape -z -f "$<" -A "$@"
	@cp "$@" $(TARGETDIR)

%.png: %.svg ignore-%.png
	@echo "Rendering $@"
	@inkscape -z -f "$<" -e "$@" --export-width=500
	@cp "$@" $(TARGETDIR)


.PHONY: all pics
