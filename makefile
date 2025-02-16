# Makefile for AI Learning System
# ==============================

.PHONY: clean run cmd format test lint docs help

# Variables
PYTHON := python
PORT := 8000
HOST := 0.0.0.0

# Colors for pretty output
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color
INFO := @echo "$(GREEN)➜$(NC)"

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk -F ':|##' '/^[^\t].+?:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$NF }' $(MAKEFILE_LIST)

clean: ## Clean python cache files
	$(INFO) "Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -r {} +
	@find . -type d -name "*.egg" -exec rm -r {} +
	@find . -type d -name ".pytest_cache" -exec rm -r {} +
	@find . -type d -name ".coverage" -exec rm -r {} +

run: clean ## Start the AI Learning System
	$(INFO) "Starting AI Learning System..."
	uvicorn main:app --host $(HOST) --port $(PORT) --reload

cmd: clean ## Run a specific command with format name=<file_name> with `.py` excluded inside commands folder
	@if [ -z "$(name)" ]; then \
		echo "Usage: make cmd name=<command_name>"; \
		echo "Available commands:"; \
		ls commands/*.py | grep -v "__" | sed 's/commands\///;s/\.py//'; \
	else \
		if [ -f "commands/$(name).py" ]; then \
			python -m commands.$(name); \
		else \
			echo "Command '$(name)' not found."; \
			echo "Did you mean one of these?"; \
			ls commands/*.py | grep -v "__" | sed 's/commands\///;s/\.py//' | grep -i "$(name)"; \
			echo ""; \
			echo "Available commands:"; \
			ls commands/*.py | grep -v "__" | sed 's/commands\///;s/\.py//'; \
		fi \
	fi

format: ## Format code using black
	$(INFO) "Formatting code..."
	black .
