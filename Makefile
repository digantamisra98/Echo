ISORT_FLAGS = --diff --atomic
BLACK_FLAGS = -v --check --diff --color
FLAKE8_FLAGS = -v


all: test

isort:
	isort $(ISORT_FLAGS) .
flake:
	flake8 $(FLAKE8_FLAGS) .	
black:
	black $(BLACK_FLAGS) .

test: flake isort black	
