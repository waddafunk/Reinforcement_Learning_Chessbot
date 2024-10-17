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

lint: 
	@echo "Linting code..."
	pylint *.py
	nbqa pylint . *.ipynb

strip_notebooks: 
	@echo "Stripping notebooks..."
	nbstripout *.ipynb

