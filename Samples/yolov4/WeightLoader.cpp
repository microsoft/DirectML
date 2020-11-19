#pragma once

#include "pch.h"
#include "WeightLoader.h"

#include "TensorExtents.h"
#include "TensorUtil.h"
#include "TensorView.h"

ConvWeights WeightLoader::RegisterConvWeights(dml::TensorDesc::Dimensions filterShape, bool hasBatchNorm)
{
    ConvWeights weights = {};
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;

#if DML_MANAGED_WEIGHTS
    flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
#endif

    dml::TensorDesc filterDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags, filterShape);
    weights.filter = dml::InputTensor(*m_graph, m_modelInputCount, filterDesc);
    ++m_modelInputCount;

    dml::TensorDesc::Dimensions biasShape = { 1, filterShape[0], 1, 1 };
    dml::TensorDesc biasDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags, biasShape);
    weights.bias = dml::InputTensor(*m_graph, m_modelInputCount, biasDesc);
    ++m_modelInputCount;

    m_registrations.push_back(WeightRegistration{ filterShape, hasBatchNorm });

    return weights;
}

template <typename T>
T Read(std::ifstream& is)
{
    static_assert(std::is_pod_v<T>);

    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    
    return val;
}

template <typename T>
void ReadArray(std::ifstream& is, dml::Span<T> out)
{
    static_assert(std::is_pod_v<T>);
    is.read(reinterpret_cast<char*>(out.data()), out.size_bytes());
}

WeightData WeightLoader::LoadWeightDataFromFile(const wchar_t* path, DX::DeviceResources* deviceResources)
{
    // yolov4 is expected to have 110 layers which require weights
    assert(m_registrations.size() == 110);

    std::ifstream file(path, std::ifstream::binary);
    if (!file || !file.good() || !file.is_open())
    {
        DX::ThrowIfFailed(E_FAIL);
    }

    file.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

    uint32_t major = Read<uint32_t>(file);
    uint32_t minor = Read<uint32_t>(file);
    uint32_t revision = Read<uint32_t>(file);
    uint32_t seen = Read<uint32_t>(file);
    /*uint32_t padding =*/ Read<uint32_t>(file);

    // Check that the file header has the correct magic values
    if (major != 0 || minor != 2 || revision != 5 || seen != 0x1e8c500)
    {
        DX::ThrowIfFailed(E_INVALIDARG); // Invalid file
    }

    std::vector<ConvWeightData> loadedWeights;
    loadedWeights.reserve(m_registrations.size());

    std::vector<float> scratchMemory;

    for (const WeightRegistration& registration : m_registrations)
    {
        ConvWeightData weights;

        uint32_t filterCount = registration.filterShape[0]; // N dimension is the filter count
        uint32_t filterSize =
            registration.filterShape[1] *
            registration.filterShape[2] *
            registration.filterShape[3]; // Size of each individual filter

        // Load BN/bias weights
        if (registration.hasBatchNorm)
        {
            // 4 weights per BN, one set of BN weights for each filter
            scratchMemory.resize(4 * filterCount);
            ReadArray<float>(file, scratchMemory);
        }
        else
        {
            weights.biasData.resize(filterCount);
            ReadArray<float>(file, weights.biasData);
        }

        // Load filter weights
        weights.filterData.resize(filterCount * filterSize);
        ReadArray<float>(file, weights.filterData);

        // Fuse the batch norm weights into the filter weights and biases
        if (registration.hasBatchNorm)
        {
            // Weights are laid out in memory SoA style - beta values, followed by gamma values, then mean values, then
            // variance values.
            assert(scratchMemory.size() == filterCount * 4);
            dml::Span<const float> betas(scratchMemory.data(), filterCount);
            dml::Span<const float> gammas(betas.end(), filterCount);
            dml::Span<const float> means(gammas.end(), filterCount);
            dml::Span<const float> variances(means.end(), filterCount);
            assert(variances.end() == scratchMemory.data() + scratchMemory.size());

            weights.biasData.resize(filterCount);
            for (uint32_t i = 0; i < filterCount; ++i)
            {
                float beta = betas[i];
                float gamma = gammas[i];
                float mean = means[i];
                float variance = variances[i];

                assert(variance >= 0); // Variance can't be negative...

                // Fold gamma/variance into filter
                dml::Span<float> filter(weights.filterData.data() + i * filterSize, filterSize);
                for (float& x : filter)
                {
                    x = gamma * x / sqrt(variance + FLT_EPSILON);
                }

                // Fold beta/mean into bias
                weights.biasData[i] = beta - gamma * mean / sqrt(variance + FLT_EPSILON);
            }
        }

        loadedWeights.push_back(std::move(weights));
    }

    file.exceptions(std::ifstream::badbit | std::ifstream::failbit); // Don't throw on EOF
    if (file.peek() != EOF)
    {
        DX::ThrowIfFailed(E_INVALIDARG); // We expect to have consumed the entire file
    }

    file.close();

    return WeightData(loadedWeights, deviceResources);
}
