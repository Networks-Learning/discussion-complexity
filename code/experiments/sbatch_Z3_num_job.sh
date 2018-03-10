#!/bin/bash

set -eo pipefail

M=$1
N=$2
I=$3
dims=$4
timeout=$5

source conda.sh
cd /home/utkarshu/prog/crowdjudged.git/code;
python num_sat.py $M $N $I --dims $dims --timeout ${timeout} --incremental --copy-last
