#include "densecrf_base.h"
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dcrf_cuda
{

    // GPU CUDA Implementation
    template <int M>
    class DenseCRFGPU : public DenseCRF
    {

    protected:
        void expAndNormalize(float *out, const float *in, float scale = 1.0, float relax = 1.0) override;
        void buildMap() override;
        void stepInit() override;

    public:
        // Create a dense CRF model of size N with M labels
        explicit DenseCRFGPU(int N);

        ~DenseCRFGPU() override;

        DenseCRFGPU(DenseCRFGPU &o) = delete;

        // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
        void setUnaryEnergy(const float *unaryGPU) override;

        // Set the unary potential via label. Length of label array should equal to N.
        void setUnaryEnergyFromLabel(const short *labelGPU, float confidence = 0.5) override;
        void setUnaryEnergyFromLabel(const short *labelGPU, float *confidences) override;
    };

    template <int M>
    __global__ static void expNormKernel(int N, float *out, const float *in, float scale, float relax);

    template <int M>
    __global__ static void unaryFromLabel(const short *inLabel, float *outUnary, int N,
                                          float u_energy, float *n_energies, float *p_energies);

    template <int M>
    __global__ void computeMAP(int N, const float *in_prob, short *out_map);

    template <int M>
    __global__ void invertKernel(int N, const float *in_unary, float *out_next);

} // end namespace
