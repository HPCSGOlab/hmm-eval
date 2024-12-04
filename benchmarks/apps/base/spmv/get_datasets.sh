#!/bin/bash -xe

mkdir data
#	      .14MB     ~2MB       7.5MB       17MB     75MB
datasets=("bcsstk34" "bcsstk38" "bcsstk37" "ct20stif" "pwtk")

for N in "${datasets[@]}"; do
	wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/$N.tar.gz

	tar xvf $N.tar.gz

	rm $N.tar.gz

	mv $N/* data/

	rmdir $N 

done
