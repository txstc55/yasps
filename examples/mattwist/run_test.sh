#!/usr/bin/env bash

set -e  # stop on error

for num in $(seq 1 10); do
    vertices=$((num * 100))
    echo "Running with vertices=$vertices"
    python mattwist.py --num-vertices "$vertices" >> "mat_twist_${num}.log" 2>&1
done
