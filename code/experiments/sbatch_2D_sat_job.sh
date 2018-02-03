#!/bin/bash

in_file=$1
op_file=$2
timeout=$3
dims=$4

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;

python z3_find.py "${in_file}" "${op_file}"  --dims ${dims} --timeout ${timeout}
