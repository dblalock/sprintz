#!/bin/bash

pdflatex ${1}.tex && pdflatex ${1}.tex && bibtex ${1} && pdflatex ${1}.tex && pdflatex ${1}.tex && open ${1}.pdf
