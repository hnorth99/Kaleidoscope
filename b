#!/usr/bin/env bash
mkdir -p build
cd build
clang++ -g -O3 ../kaleidoscope.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o kaleidoscope