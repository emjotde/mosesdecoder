#!/bin/bash

all: amunmt/build/src/libmosesplugin.a

amunmt/build/src/libmosesplugin.a: amunmt
	cd $< && mkdir build && cd build && cmake .. && make -j mosesplugin


amunmt:
	git clone https://github.com/amunmt/amunmt.git -b plugin


