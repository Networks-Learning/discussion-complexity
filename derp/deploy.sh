#!/bin/sh
make
rsync -av dist/* contact.mpi-sws.org:~/public_html/derp-crowdjudged/
