/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

// Slightly modified to remove redundant dependencies by Jiahui Huang

#pragma once

#define BLOCK_SIZE 512

#include <cstdio>
#include <utility>
#include <iostream>

namespace dcrf_cuda
{

        void cudaErrorCheck();

        template <class T>
        void printDeviceValues(const T *device_ptr, int count);

        // GPU version Hash Table (with fixed size)
        template <typename T, int pd, int vd>
        class HashTableGPU
        {
        public:
                int capacity;
                short *keys;
                int *entries;
                T *values;

                HashTableGPU(int capacity_);

                __device__ int modHash(unsigned int n);

                __device__ unsigned int hash(short *key);

                // Insert key into slot. Return bucket id.
                __device__ int insert(short *key, unsigned int slot);

                // Find key. Return slot id.
                __device__ int retrieve(short *key);
        };

        template <typename T>
        struct MatrixEntry
        {
                int index;
                T weight;
        };

        template <typename T, int pd, int vd>
        __global__ static void createLattice(const int n,
                                             const T *positions,
                                             const T *scaleFactor,
                                             MatrixEntry<T> *matrix,
                                             HashTableGPU<T, pd, vd> table);

        template <typename T, int pd, int vd>
        __global__ static void cleanHashTable(int n, HashTableGPU<T, pd, vd> table);

        template <typename T, int pd, int vd>
        __global__ static void splatCache(const int n, const T *values, MatrixEntry<T> *matrix, HashTableGPU<T, pd, vd> table, bool isInit);

        template <typename T, int pd, int vd>
        __global__ static void blur(int n, T *newValues, MatrixEntry<T> *matrix, int color, HashTableGPU<T, pd, vd> table);

        template <typename T, int pd, int vd>
        __global__ static void slice(const int n, T *values, MatrixEntry<T> *matrix, HashTableGPU<T, pd, vd> table);

        template <typename T, int pd, int vd>
        class PermutohedralLatticeGPU
        {
        public:
                int n; // number of pixels/voxels etc..
                T *scaleFactor;
                MatrixEntry<T> *matrix;
                HashTableGPU<T, pd, vd> hashTable;
                T *newValues;    // auxiliary array for blur stage
                int filterTimes; // Lazy mark

                void init_scaleFactor();

                void init_matrix();

                void init_newValues();

                void init_hashTable();

                PermutohedralLatticeGPU(int n_);

                ~PermutohedralLatticeGPU();

                // values and position must already be device pointers
                void prepare(const T *positions);

                // values and position must already be device pointers
                void filter(T *output, const T *inputs, bool reverse = false);
        };

}
