#pragma once
#include "densecrf_base.h"
#include "permutohedral_gpu.cuh"

namespace dcrf_cuda
{

    // Weight applying kernels for potts potential.
    template <int M, int F>
    __global__ static void pottsWeight(float *out, const float *in, const int n, const float pw);

    // Initializing kernels for potts potential.
    template <class T, int M, int F>
    __global__ static void assembleImageFeature(int w, int h, const T *features, float posdev, float featuredev, float *out);

    template <class PT, class FT, int M, int F>
    __global__ static void assembleUnorganizedFeature(int N, int pdim, const PT *positions, const FT *features, float posdev, float featuredev, float *out);

    template <int M, int F>
    class PottsPotentialGPU : public PairwisePotential
    {
    protected:
        PermutohedralLatticeGPU<float, F, M + 1> *lattice_;
        float w_;

    public:
        PottsPotentialGPU(const float *features, int N, float w);
        ~PottsPotentialGPU();
        PottsPotentialGPU(const PottsPotentialGPU &o) = delete;

        //// Factory functions:
        // Build image-based potential: if features is NULL then applying gaussian filter only.
        template <class T = float>
        static PottsPotentialGPU<M, F> *FromImage(int w, int h, float weight, float posdev, const T *features = nullptr, float featuredev = 0.0);
        // Build linear potential:
        template <class PT = float, class FT = float>
        static PottsPotentialGPU<M, F> *FromUnorganizedData(int N, float weight, const PT *positions, float posdev, int posdim,
                                                            const FT *features = nullptr, float featuredev = 0.0);

        // tmp should be larger to store normalization values. (N*(M+1))
        // All pointers are device pointers
        void apply(float *out_values, const float *in_values, float *tmp) const;
    };

}
