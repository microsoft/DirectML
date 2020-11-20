#pragma once

#include "WeightData.h"
#include "DeviceResources.h"

struct ConvWeights
{
    dml::Expression filter;
    dml::Expression bias;
};

class WeightLoader
{
public:
    explicit WeightLoader(dml::Graph* graph, uint32_t modelInputCount)
        : m_graph(graph)
        , m_modelInputCount(modelInputCount)
    {}

    ConvWeights RegisterConvWeights(dml::TensorDesc::Dimensions filterShape, bool hasBatchNorm);
    WeightData LoadWeightDataFromFile(const wchar_t* path, DX::DeviceResources* deviceResources);

private:
    struct WeightRegistration
    {
        dml::TensorDesc::Dimensions filterShape;
        bool hasBatchNorm;
    };

    dml::Graph* m_graph;
    std::vector<WeightRegistration> m_registrations;
    uint32_t m_modelInputCount;
};