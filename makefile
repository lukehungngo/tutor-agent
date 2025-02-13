run:
	echo "Starting AI Learning System..."
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Example usage:
# make cmd name=chat
# make cmd name=display_graph
cmd:
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

