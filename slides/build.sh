latexmk -C fi-czech.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error fi-czech.tex
