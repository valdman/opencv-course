#!/bin/bash
let N = 0
mkdir -p ./out/runs
while IFS='' read -r line || [[ -n "$line" ]]; do
    ((N++))
    eval $line
    mkdir -p ./out/runs/$N
    mv *.jpg ./out/runs/$N
    echo $line >> ./out/runs/$N/run.sh
done < "$1"