FILE = template

all:
	platex $(FILE).tex
	platex $(FILE).tex
	dvipdfmx $(FILE).dvi
tex:
	platex $(FILE).tex
	platex $(FILE).tex
pdf: $(FILE).dvi
	dvipdfmx $(FILE).dvi
clean:
	@rm -f	*~ \
		$(FILE).aux $(FILE).dvi $(FILE).log \
		$(FILE).lof $(FILE).lot $(FILE).toc \
		$(FILE).ps  $(FILE).pdf
