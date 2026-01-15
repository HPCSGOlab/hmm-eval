PARAMS=(4096 8192 16384 32768)

for N in ${PARAMS[@]}; do
	echo $N
	bash iter_run.sh $N
done
