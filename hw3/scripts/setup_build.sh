#!/bin/bash

orgdir=`pwd`

cd ~/Dev/numinal/hw3
mkdir build
cd build
cmake ../src
make
cp ~/Dev/numinal/hw3/matlab/*.csv ~/Dev/numinal/hw3/build/

cd $orgdir