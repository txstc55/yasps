#!/usr/bin/env bash

set -e  # stop on error

for num in $(seq 5 10); do
    echo "Running with num-bunnies=$num"
    python dropping_in_container.py --num-bunnies "$num" >> "bunny_${num}_memory_new_new.log" 2>&1
done
# python dropping_in_container.py --num-bunnies 10 --save-obj 1
