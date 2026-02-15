#!/bin/bash
# Open Jupyter notebook for pipeline
cd "$(dirname "$0")/.."
jupyter notebook notebooks/maga_pipe.ipynb
