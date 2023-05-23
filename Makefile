.PHONY: clean 

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL := /bin/bash
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

LOGG := $(PYTHON_INTERPRETER) -c "import sys, datetime; print(f'{datetime.datetime.now()}: [ENV] {sys.argv[1]}');"

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create Python Virtual Environment
create_environment:
	@if ! command -v $(PYTHON_INTERPRETER) > /dev/null; then \
		echo "ERROR: $(PYTHON_INTERPRETER) not found"; \
		exit 1; \
	fi

	@if [ -d $(PROJECT_DIR)/venv ]; then \
		$(LOGG) "Virtual Environment Already Configured"; \
	else \
		$(PYTHON_INTERPRETER) -m venv $(PROJECT_DIR)/venv; \
		$(LOGG) "Virtual Environment Created"; \
	fi

## Install Python Dependencies from PyPI
requirements: create_environment
	@( \
		source $(PROJECT_DIR)/venv/bin/activate; \
		$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel > /dev/null; \
		$(PYTHON_INTERPRETER) -m pip install -r $(PROJECT_DIR)/requirements.txt > /dev/null; \
		$(LOGG) "Installed Python Dependencies"; \
	)

## Perform Patient Matching on Full Dataset
patient_matching: requirements
	@( \
		$(LOGG) "Starting Patient Matching"; \
		source $(PROJECT_DIR)/venv/bin/activate; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIR)/patient_matching.py; \
		$(LOGG) "Completed Patient Matching"; \
	)

## Perform Patient Matching on New Records
patient_matching_update: requirements
	@( \
		$(LOGG) "Starting Patient Matching"; \
		source $(PROJECT_DIR)/venv/bin/activate; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIR)/patient_matching_update.py; \
		$(LOGG) "Completed Patient Matching"; \
	)

## Perform Patient Matching on New Records (Automated Environments)
patient_matching_update_auto: requirements
	@( \
		$(LOGG) "Starting Patient Matching"; \
		source $(PROJECT_DIR)/venv/bin/activate; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIR)/patient_matching_update.py 2> /dev/null; \
		$(LOGG) "Completed Patient Matching"; \
	);

## Remove Python Dependancies, Cache and Virtual Environment
clean:
	@( \
		find . -type f -name "*.py[co]" -delete; \
		find . -type d -name "__pycache__" -delete; \
		touch $(PROJECT_DIR)/venv; \
		rm -r $(PROJECT_DIR)/venv; \
		$(LOGG) "Removed Python Dependancies"; \
	)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
