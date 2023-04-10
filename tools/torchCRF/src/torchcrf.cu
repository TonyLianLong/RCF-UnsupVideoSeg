#include <torch/extension.h>
#include <iostream>
#include <vector>

// Including the implementation since we use template
#include "densecrf_base.cpp"
#include "densecrf_gpu.cu"
#include "pairwise_gpu.cu"
#include "permutohedral_gpu.cu"

using namespace dcrf_cuda;

// C++ interface

// We are interested in the binary version (this could be changed for multi-class)
#define M 2

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_COND(x) TORCH_CHECK(x, #x " not satisfied")

inline void addSmoothnessPairwise(DenseCRFGPU<M> &crf, const uint32_t W, const uint32_t H, const float scompSmooth = 3.0, const float sxySmooth = 3.0)
{
  if (scompSmooth > 0.0 && sxySmooth > 0.0)
  {
    // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
    // x_stddev = 3
    // y_stddev = 3
    // weight = 3
    auto *smoothnessPairwise = PottsPotentialGPU<M, 2>::FromImage<float>(W, H, scompSmooth, sxySmooth);
    crf.addPairwiseEnergy(smoothnessPairwise);
  }
}

inline void addAppearancePairwise(DenseCRFGPU<M> &crf, float *rgbFeatGPU, const uint32_t W, const uint32_t H, const float scompApp = 10.0, const float sxyApp = 60.0, const float srgbApp = 20.0)
{
  if (scompApp > 0.0 && sxyApp > 0.0)
  {
    // add a color dependent term (feature = xyrgb)
    // x_stddev = 60
    // y_stddev = 60
    // r_stddev = g_stddev = b_stddev = 20
    // weight = 10
    auto *appearancePairwise = PottsPotentialGPU<M, 5>::FromImage<float>(W, H, scompApp, sxyApp, rgbFeatGPU, srgbApp);
    crf.addPairwiseEnergy(appearancePairwise);
  }
}

short *getLabelGPU(torch::Tensor &label, const uint32_t W, const uint32_t H)
{
  CHECK_INPUT(label);
  CHECK_COND(label.dim() == 2);
  CHECK_COND(label.size(0) == H);
  CHECK_COND(label.size(1) == W);

  short *labelGPU = label.data_ptr<short>();
  return labelGPU;
}

float *getEnergyGPU(torch::Tensor &energy, const uint32_t W, const uint32_t H)
{
  CHECK_INPUT(energy);
  CHECK_COND(energy.dim() == 2);
  CHECK_COND(energy.size(0) == H * W);
  CHECK_COND(energy.size(1) == 2);

  float *energyGPU = energy.data_ptr<float>();

  return energyGPU;
}

float *getRgbFeatGPU(torch::Tensor &rgbFeat, const uint32_t W, const uint32_t H)
{
  CHECK_INPUT(rgbFeat);
  CHECK_COND(rgbFeat.dim() == 3);
  CHECK_COND(rgbFeat.size(0) == H);
  CHECK_COND(rgbFeat.size(1) == W);
  CHECK_COND(rgbFeat.size(2) == 3);

  auto rgbFeatFloat = rgbFeat.toType(at::ScalarType::Float);
  float *rgbFeatGPU = rgbFeatFloat.data_ptr<float>();

  return rgbFeatGPU;
}

at::Tensor crfInference(DenseCRFGPU<M> &crf, at::Device device, const uint32_t W, const uint32_t H, const int iters)
{
  // Do map inference
  crf.inference(iters, true);
  short *mapGPU = crf.getMap();
  
  auto options = torch::TensorOptions().dtype(torch::kInt16).layout(torch::kStrided).device(device).requires_grad(false);
  auto map = torch::empty({H, W}, options);
  
  cudaMemcpy(map.data_ptr<short>(), mapGPU, sizeof(short) * W * H, cudaMemcpyDeviceToDevice);
  cudaErrorCheck();

  return map;
}

// This is used for hard label. It could be used for multi-label with maximum number of classes in M.
at::Tensor torchcrfHard(torch::Tensor rgbFeat, torch::Tensor label, const uint32_t W, const uint32_t H, const float scompSmooth = 3.0, const float sxySmooth = 3.0,
                    const float scompApp = 10.0, const float sxyApp = 60.0, const float srgbApp = 20.0, const float confidence = 0.5, const int iters = 10)
{

  short *labelGPU = getLabelGPU(label, W, H);
  float *rgbFeatGPU = getRgbFeatGPU(rgbFeat, W, H);

  // Setup the CRF model
  DenseCRFGPU<M> crf(W * H);
  crf.setUnaryEnergyFromLabel(labelGPU, confidence);

  addSmoothnessPairwise(crf, W, H, scompSmooth, sxySmooth);
  addAppearancePairwise(crf, rgbFeatGPU, W, H, scompApp, sxyApp, srgbApp);

  auto map = crfInference(crf, label.device(), W, H, iters);

  return map;
}

// This is used for binary label smoothing (given unary energy directly)
at::Tensor torchcrfSoft(torch::Tensor rgbFeat, torch::Tensor unaryEnergy, const uint32_t W, const uint32_t H, const float scompSmooth = 3.0, const float sxySmooth = 3.0,
                    const float scompApp = 10.0, const float sxyApp = 60.0, const float srgbApp = 20.0, const int iters = 10)
{

  float *unaryEnergyGPU = getEnergyGPU(unaryEnergy, W, H);
  float *rgbFeatGPU = getRgbFeatGPU(rgbFeat, W, H);

  // Setup the CRF model
  DenseCRFGPU<M> crf(W * H);
  crf.setUnaryEnergy(unaryEnergyGPU);

  addSmoothnessPairwise(crf, W, H, scompSmooth, sxySmooth);
  addAppearancePairwise(crf, rgbFeatGPU, W, H, scompApp, sxyApp, srgbApp);

  auto map = crfInference(crf, unaryEnergy.device(), W, H, iters);

  return map;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("crf_hard", &torchcrfHard, "Run CRF for hard label smoothing");
  m.def("crf_soft", &torchcrfSoft, "Run CRF for binary soft label smoothing");
}
