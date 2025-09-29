#!/bin/csh

# RUN ARGO SPARSENESS MAP BUILDING EVERY MONTH FOR PREVIOUS ONE
# K. BALEM - LOPS,IFREMER 2020

cd /home/lops/users/kbalem/Argo/sparseness
python sparse_calc.py

#TRACK
date > last_update.txt
