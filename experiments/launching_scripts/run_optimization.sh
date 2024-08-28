#!/bin/bash

for i in 0 1
do
    mpiexec -np 4 python /home/ana/ML-DFT/dft_scripts/optimization.py -symbs ag8 -isomer $i -vacuum 10
    mpiexec -np 4 python /home/ana/ML-DFT/dft_scripts/optimization.py -symbs ag8 -isomer $i -cell 22
done