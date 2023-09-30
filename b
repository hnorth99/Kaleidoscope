#!/usr/bin/env bash
mkdir -p build
cd build
clang++ -g ./../kaleidoscope.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o kaleidoscope