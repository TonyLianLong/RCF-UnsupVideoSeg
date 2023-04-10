#include <vector>
#include "densecrf_base.h"

namespace dcrf_cuda
{
    DenseCRF::~DenseCRF()
    {
        for (auto *pPairwise : pairwise_)
        {
            delete pPairwise;
        }
    }
    // Run inference and return the probabilities
    // All returned values are managed by class
    void DenseCRF::inference(int n_iterations, bool with_map, float relax)
    {
        startInference();
        for (int it = 0; it < n_iterations; ++it)
        {
            stepInference(relax);
        }
        if (with_map)
        {
            buildMap();
        }
    }
    short *DenseCRF::getMap() const { return map_; }
    float *DenseCRF::getProbability() const { return current_; }

    // Step by step inference
    void DenseCRF::startInference()
    {
        expAndNormalize(current_, unary_, -1);
    }

    void DenseCRF::stepInference(float relax)
    {
        // Set the unary potential
        stepInit();
        // Add up all pairwise potentials
        for (unsigned int i = 0; i < pairwise_.size(); i++)
        {
            pairwise_[i]->apply(next_, current_, tmp_);
        }
        // Exponentiate and normalize
        expAndNormalize(current_, next_, 1.0, relax);
    }
}