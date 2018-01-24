#!/bin/bash

set -eo pipefail

in_file=$1
n_dim=$2
timeout=$3

source conda.sh

cd /home/utkarshu/prog/crowdjudged.git/code;
python matrix-Z3-embed.py ${in_file} --dim ${n_dim} --timeout ${timeout}
