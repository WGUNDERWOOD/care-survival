all: illustration

illustration:
    uv run bin/illustration_simulation.py 1
    uv run bin/illustration_simulation.py 2
