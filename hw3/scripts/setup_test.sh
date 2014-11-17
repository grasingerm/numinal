#!/bin/bash

orgdir=`pwd`

cd ~/Dev/numinal/hw3
mkdir build_test
cd build_test
cmake ../test
make

cd $orgdir