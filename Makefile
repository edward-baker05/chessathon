SHELL := /bin/bash

.PHONY: setup play arena zip gate test bench ab

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)") --pgn game.pgn

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

test:
	uv run pytest -q

bench:
	uv run python tests/bench.py

ab:
	uv run python tests/match.py --opponent $(OPPONENT) --games $(if $(GAMES),$(GAMES),200) --nodes $(if $(NODES),$(NODES),200000)
