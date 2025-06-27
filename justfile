REPS := "10"

all: illustration

illustration:
    uv run bin/illustration_simulation.py 1
    uv run bin/illustration_simulation.py 2

analysis:
    seq 1 {{REPS}} | parallel --bar --lb uv run bin/analysis_simulation.py 1
    seq 1 {{REPS}} | parallel --bar --lb uv run bin/analysis_simulation.py 2
