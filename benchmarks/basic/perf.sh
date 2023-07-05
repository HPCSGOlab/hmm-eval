#!/bin/bash -xe

exe="./basic"
N=(1000000000 2000000000 3000000000 4000000000 5000000000 6000000000)
perfdir=perf/
mkdir -p ${perfdir}

sudo systemctl stop nvidia-persistenced.service
sudo /home/tallen93/cuda_12.2.0_535.54.03_linux.run --silent --driver -m=kernel
sudo systemctl start nvidia-persistenced.service

prefix="closed"
output=${perfdir}/${prefix}.log
rm -f ${output}
for i in ${N[@]}; do
    perft=$($exe $i 1 2>&1 >/dev/null)
    echo "${i}, ${perft}" >> ${output}
done

sudo systemctl stop nvidia-persistenced.service
sudo /home/tallen93/cuda_12.2.0_535.54.03_linux.run --silent --driver -m=kernel-open 
sudo systemctl start nvidia-persistenced.service

prefix="open"
output=${perfdir}/${prefix}.log
rm -f ${output}
for i in ${N[@]}; do
    perft=$($exe $i 1 2>&1 >/dev/null)
    echo "${i}, ${perft}" >> ${output}
done

sudo systemctl stop nvidia-persistenced.service
sudo /home/tallen93/cuda_12.2.0_535.54.03_linux.run --silent --driver -m=kernel-open
sudo systemctl start nvidia-persistenced.service

prefix="hmm"
output=${perfdir}/${prefix}.log
rm -f ${output}
for i in ${N[@]}; do
    perft=$($exe $i 0 2>&1 >/dev/null)
    echo "${i}, ${perft}" >> ${output}
done
