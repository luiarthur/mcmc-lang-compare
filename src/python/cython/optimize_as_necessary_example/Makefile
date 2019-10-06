SHELL = /bin/bash

.PHONY: build clean all

all: build

build:
	python3 setup.py build_ext --inplace

clean:
	rm -f *.so *.c
	rm -rf build/ __pychache__/

