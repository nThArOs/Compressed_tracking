#!/bin/bash

cd deep_sort_residual_1720/

i=1
for file in *; do
    mv "$file" "$(printf "%04d" $i)"
    let i=i+1
done

