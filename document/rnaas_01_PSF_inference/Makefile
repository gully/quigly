all: rnaas.pdf

%.pdf: %.tex
	pdflatex rnaas
	pdflatex rnaas
	rm -rf *Notes.bib
	rm -rf *.log
	rm -rf *.bbl
	rm -rf *.aux
	mv rnaas.pdf final/