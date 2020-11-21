ISORT_FLAGS = --atomic
BLACK_FLAGS = 
AUTOFLAKE_FLAGS = -i --remove-all-unused-imports --remove-unused-variables
FLAKE8_FLAGS = 


all:

isort:
	isort $(ISORT_FLAGS) .
	@echo "Passed isort tests"
flake:
	find . -name '*.py' | xargs autoflake $(AUTOFLAKE_FLAGS)
	flake8 $(FLAKE8_FLAGS)	
	@echo "Passed flake8 tests"
black:
	black $(BLACK_FLAGS) .
	@echo "Passed black tests"

lint: isort black flake

test: lint
	python tests/test_t_ops.py
