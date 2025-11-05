#!/bin/bash

# ============================================================================
# notebook_to_py.sh: Convert .ipynb to .py
# ============================================================================
# Usage:
#   ./notebook_to_py.sh notebook.ipynb              # Converts to notebook.py
#   ./notebook_to_py.sh notebook.ipynb output.py    # Converts to output.py
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: ./notebook_to_py.sh <notebook.ipynb> [output.py]"
    echo ""
    echo "Examples:"
    echo "  ./notebook_to_py.sh 00_esm2_attention_setup.ipynb"
    echo "  ./notebook_to_py.sh 00_esm2_attention_setup.ipynb script.py"
    exit 1
fi

INPUT=$1
OUTPUT=${2:-"${INPUT%.ipynb}.py"}

echo "Converting: $INPUT → $OUTPUT"

jupyter nbconvert --to script "$INPUT" --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "✓ Done! Run with: python $OUTPUT"
else
    echo "✗ Conversion failed"
    exit 1
fi