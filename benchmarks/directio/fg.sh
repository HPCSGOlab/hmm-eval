#!/bin/bash -xe 

perfexe=`which perf`
fg_dir=/home/tallen93/dev/FlameGraph/ 
exe="./dio foo.bin" 
exe2="./diocpu foo.bin" 

$perfexe record -F 100 -a --call-graph  dwarf -m256M  $exe 
$perfexe script -f | ${fg_dir}/stackcollapse-perf.pl > out.folded 
${fg_dir}/flamegraph.pl out.folded > dio.svg


$perfexe record -F 100 -a --call-graph  dwarf -m256M  $exe2
$perfexe script -f | ${fg_dir}/stackcollapse-perf.pl > out.folded 
${fg_dir}/flamegraph.pl out.folded > diocpu.svg
