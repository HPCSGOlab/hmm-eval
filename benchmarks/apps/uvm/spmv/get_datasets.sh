#!/bin/bash -xe

mkdir data

wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstm20.tar.gz

tar xvf bcsstm20.tar.gz

rm bcsstm20.tar.gz

mv bcsstm20/* data/

rmdir bcsstm20
