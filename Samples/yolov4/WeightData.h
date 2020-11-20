#pragma once

#include "DeviceResources.h"

// A set of filter weights/biases for a single convolution.
struct ConvWeightData
{
    std::vector<float> filterData;
    std::vector<float> biasData;
};

class WeightData
{
public:
    WeightData(dml::Span<const ConvWeightData> weights, DX::DeviceResources* deviceResources);

    dml::Span<const DML_BUFFER_BINDING> GetBindings() const
    {
        return m_bindings;
    }

private:
    Microsoft::WRL::ComPtr<ID3D12Resource> m_weightBuffer;
    std::vector<DML_BUFFER_BINDING> m_bindings;
};
