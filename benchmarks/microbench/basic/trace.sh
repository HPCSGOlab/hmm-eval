#!/bin/bash

#perfexe=perf #`which perf`
perfexe=/usr/lib/linux-tools/6.2.0-24-generic/perf

exe="./basic"
flamegraph=/home/tallen93/dev/FlameGraph/


taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 1000000000
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > callgraph_small.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 2500000000
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > callgraph_medium.svg

taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 4000000000 
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > callgraph_oversub.svg


taskset -c 2 $perfexe record -F 100 --call-graph  dwarf -m256M  $exe 6000000000 
$perfexe script -f | ${flamegraph}/stackcollapse-perf.pl > out.folded
${flamegraph}/flamegraph.pl out.folded > callgraph_oversub_dub.svg
