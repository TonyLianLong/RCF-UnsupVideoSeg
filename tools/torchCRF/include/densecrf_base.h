#pragma once

#include <vector>

namespace dcrf_cuda {

enum Device {
    CPU, GPU
};

// Define various pair wise weight (compatibility) models.
class PairwisePotential {
protected:
    int N_;
public:
    PairwisePotential(int N) : N_(N) {}
    virtual ~PairwisePotential() = default;
    virtual void apply( float * out_values, const float * in_values, float * tmp ) const = 0;
};


class DenseCRF {
protected:
    // Number of variables and labels
    int N_;
    // Pre-allocated host/device memory
    float *unary_, *current_, *next_, *tmp_;
    short *map_;

    // Store all pairwise potentials
    std::vector<PairwisePotential*> pairwise_;

    // Auxillary functions
    virtual void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 ) = 0;
    virtual void buildMap() = 0;
    virtual void stepInit() = 0;

public:

    DenseCRF(int N) : N_(N), unary_(nullptr), current_(nullptr), next_(nullptr), tmp_(nullptr), map_(nullptr) {}
    virtual ~DenseCRF();

    // Add your own favorite pairwise potential (ownwership will be transfered to this class)
    void addPairwiseEnergy( PairwisePotential* potential ) { pairwise_.push_back(potential); }

    // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
    virtual void setUnaryEnergy( const float * unary ) = 0;

    // Set the unary potential via label. Length of label array should equal to N.
    virtual void setUnaryEnergyFromLabel(const short* label, float* confidences) = 0;
    virtual void setUnaryEnergyFromLabel(const short* label, float confidence = 0.5) = 0;

    // Run inference and return the probabilities
    // All returned values are managed by class
    virtual void inference( int n_iterations, bool with_map = false, float relax = 1.0 );
    short* getMap() const;
    float* getProbability() const;

    // Step by step inference
    virtual void startInference();

    virtual void stepInference( float relax = 1.0 );
};

}