# GSTL

STL made for CUDA

## Purpose

This library was designed as a "complement" to [CUB](https://github.com/NVlabs/cub). It aims to provide similar functionalities to the STL while including notions of parallelism. It therefore focuses exclusively on block or warp operations  in order to serve as primitives for your programs.

It is still in an early stage but has a simple API that aims to simplify the GPU development experience while still respecting the underlying concepts.

## Instructions to compile

With MSVC 15.9.11 and CMake 3.14.2
 - mkdir build
 - cd build
 - cmake -G "Visual Studio 15 2017" -A x64 ..

## Thanks

A special thanks to the [Catch](https://github.com/catchorg/Catch2) and [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers) libraries who are both pearls
