ISORT_FLAGS = --atomic
BLACK_FLAGS = -v --color
AUTOFLAKE_FLAGS = -i --remove-all-unused-imports --remove-unused-variables
FLAKE8_FLAGS = 


all:

isort:
	isort $(ISORT_FLAGS) .
flake:
	find . -name '*.py' | xargs autoflake $(AUTOFLAKE_FLAGS)
	flake8 $(FLAKE8_FLAGS)	
	@echo "Passed flake8 tests"
black:
	black $(BLACK_FLAGS) .

lint: isort black flake

test: lint
