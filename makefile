.PHONY: all help venv install format lint strip_notebooks

all: install strip_notebooks format lint 

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install           Install package dependencies"
	@echo "  format            Format code using black"
	@echo "  lint              Lint code using pylint"
	@echo "  strip_notebooks   Strip notebooks using nbstripout"
	@echo "  all               Run all targets"


install: 
	@echo "Installing dependencies..."
	pip install -r requirements.txt

format: 
	@echo "Formatting code..."
	 black .
	 nbqa black .
	 isort .
	 nbqa isort .

lint: 
	@echo "Linting code..."
	pylint *.py --disable=C0114,C0115,C0116,R1725,C0103,W0621,W0603

strip_notebooks: 
	@echo "Stripping notebooks..."
	nbstripout *.ipynb

