REPS_SIM := "50"
REPS_SCORE2 := "2"
DATE := "2025-06-30"

default: simulation plot_simulation

all: simulation score2 plot

simulation: illustration_simulation analysis_simulation

score2: illustration_score2 analysis_score2

plot: plot_simulation plot_score2

illustration_simulation:
    uv run bin/illustration_simulation.py 1
    uv run bin/illustration_simulation.py 2

analysis_simulation:
    seq 1 {{REPS_SIM}} | parallel --bar --lb uv run bin/analysis_simulation.py 1
    seq 1 {{REPS_SIM}} | parallel --bar --lb uv run bin/analysis_simulation.py 2

illustration_score2:
    uv run bin/illustration_score2.py female
    uv run bin/illustration_score2.py male

analysis_score2:
    parallel --bar --lb uv run bin/analysis_score2.py 1 {1} {2} ::: \
            female male ::: $(seq 1 {{REPS_SCORE2}})
    parallel --bar --lb uv run bin/analysis_score2.py 2 {1} {2} ::: \
            female male ::: $(seq 1 {{REPS_SCORE2}})

plot_simulation:
    parallel --bar --lb uv run bin/plot/{1}.py {{DATE}} ::: \
        aggregation_simulation \
        breslow_simulation \
        estimator_simulation \
        kernel_simulation \
        regularisation_simulation \
        scatter_simulation \
        selection_simulation \
        sketch \
        validation_simulation \

plot_score2:
    parallel --bar --lb uv run bin/plot/{1}.py {{DATE}} ::: \
        aggregation_score2 \
        breslow_score2 \
        selection_score2 \
