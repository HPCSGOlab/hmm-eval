#!/bin/bash -xe

#perfexe=perf #`which perf`
#perfexe=/usr/lib/linux-tools/6.2.0-24-generic/perf
perfexe=/usr/lib/linux-tools/5.15.0-83-generic/perf

exe="./basic"
flamegraph=/home/tallen93/dev/FlameGraph/

# pre-setup to ensure all symbols readable
sudo sh -c "echo 0 > /proc/sys/kernel/kptr_restrict"
sudo sh -c "echo -1 > /proc/sys/kernel/perf_event_paranoid"


#begin open source driver
prefix="open"
#sudo systemctl stop nvidia-persistenced.service
#sudo /home/tallen93/cuda_12.2.0_535.54.03_linux.run --silent --driver -m=kernel-open
#sudo systemctl start nvidia-persistenced.service

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 1000000000 0
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_small.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 2500000000 0
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_medium.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 4000000000 0
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_oversub.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 6000000000 0
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_oversub_dub.svg

exit

#begin closed source driver
prefix="closed"
sudo systemctl stop nvidia-persistenced.service
sudo /home/tallen93/cuda_12.2.0_535.54.03_linux.run --silent --driver -m=kernel
sudo systemctl start nvidia-persistenced.service

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 1000000000 1
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_small.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 2500000000 1
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_medium.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 4000000000 1
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_oversub.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 6000000000 1
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > ${prefix}_callgraph_oversub_dub.svg
