#!/usr/bin/env bash

set -e  # stop on error

for num in $(seq 5 10); do
    echo "Running with num-bunnies=$num"
    python dropping_in_container_no_save.py --num-bunnies "$num" >> "bunny_${num}.log" 2>&1
done
python dropping_in_container.py --num-bunnies 10
