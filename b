#!/usr/bin/env bash
mkdir -p build
cd build
clang++ -g -O3 ../kaleidoscope.cpp `llvm-config --cxxflags` -o kaleidoscope.out