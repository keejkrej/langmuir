#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Usage: $0 <experiment>" >&2
    echo "Example: $0 1" >&2
    exit 1
fi
EXPERIMENT=$1

# Clean up processed and plot directories for this experiment
rm -rf ~/results/langmuir/${EXPERIMENT}/gixd
rm -rf ~/plots/langmuir/${EXPERIMENT}/gixd

# Run processing script
uv run process_gixd.py --experiment ${EXPERIMENT}

# Run plotting script
uv run plot_gixd.py --experiment ${EXPERIMENT}

echo "GIXD pipeline completed for experiment ${EXPERIMENT}."
