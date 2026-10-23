install:
	uv sync

install-dev:
	uv sync --all-extras

check:
	@output=$$(copier check-update --answers-file .template.yaml 2>&1) || true; \
	if echo "$$output" | grep -q "up-to-date"; then \
		true; \
	elif echo "$$output" | grep -q "New template version"; then \
		echo "Template update available, run make update-template"; \
	else \
		echo "$$output"; \
	fi

	@find . \
		-name "*.py" \
		-not -path "./.venv/*" \
		-not -path "./.git/*" \
		-exec uv run python -m py_compile {} +
	@echo "Compilation ok."

	@uv run ruff format \
		--quiet \
		--config=pyproject.toml \
		--check
	@echo "Formatting ok."

	@uv run ruff check \
		--quiet \
		--config=pyproject.toml
	@echo "Linter ok."

	@output=$$(uv run basedpyright 2>&1); exit_code=$$?; \
	if [ $$exit_code -ne 0 ]; then echo "$$output"; fi; \
	exit $$exit_code
	@echo "Typechecking ok."

	@uv run pytest \
		--quiet \
		--config-file=pyproject.toml
	@echo "Testing ok."

fix:
	@uv run ruff format \
		--quiet \
		--config=pyproject.toml

	@uv run ruff check \
		--quiet \
		--config=pyproject.toml \
		--fix

wheel:
	uv build --wheel

new-branch:
	@read -p "Branch name: " branch_name && \
	branch_name=$${branch_name// /-} && \
	base=$$(git remote show origin | sed -n '/HEAD branch/s/.*: //p') && \
	echo "Selecting $$base branch as default" && \
	git fetch origin $$base && \
	git checkout -b $$branch_name origin/$$base && \
	git push -u origin $$branch_name

update-template:
	copier update \
		--skip-answered \
		--conflict inline \
		--answers-file .template.yaml
