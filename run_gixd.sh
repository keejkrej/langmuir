#!/bin/bash

# Get experiment number from argument or default to 1
EXPERIMENT=${1:-1}

# Clean up processed and plot directories for this experiment
rm -rf ~/results/langmuir/${EXPERIMENT}/gixd
rm -rf ~/plots/langmuir/${EXPERIMENT}/gixd

# Run processing script
uv run process_gixd.py --experiment ${EXPERIMENT}

# Run plotting script
uv run plot_gixd.py --experiment ${EXPERIMENT}

echo "GIXD pipeline completed for experiment ${EXPERIMENT}."
