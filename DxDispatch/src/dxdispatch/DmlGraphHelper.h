// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "AbstractOperatorDesc.h"
#include "SchemaHelpers.h"
#include "DmlSerializedGraphDesc.h"

struct GraphEdgeIndexInfo
{
    uint32_t maxIndex = 0;
    bool hasEdge = false;
};

inline uint32_t GetConstantNodeGraphInputIndex(
    const std::string& constantName,
    const std::unordered_map<std::string_view, uint32_t>* serializedGraphConstantNameToMainGraphInputIndex,
    GraphEdgeIndexInfo& graphInputIndexInfo,
    std::unordered_map<std::string_view, uint32_t>& localConstantNameToIndexMap)
{
    if (serializedGraphConstantNameToMainGraphInputIndex == nullptr)
    {
        if (localConstantNameToIndexMap.find(constantName) == localConstantNameToIndexMap.end())
        {
            localConstantNameToIndexMap[constantName] = ++graphInputIndexInfo.maxIndex;
            graphInputIndexInfo.hasEdge = true;
        }
        return localConstantNameToIndexMap[constantName];
    }
    else
    {
        graphInputIndexInfo.maxIndex = std::max(graphInputIndexInfo.maxIndex, serializedGraphConstantNameToMainGraphInputIndex->at(constantName));
        graphInputIndexInfo.hasEdge = true;
        return serializedGraphConstantNameToMainGraphInputIndex->at(constantName);
    }
}

template <size_t ALLOCATOR_SIZE>
void ConvertGraphDesc(
    const DmlSerializedGraphDesc& graphDesc,
    _Out_ DML_GRAPH_DESC& dmlGraphDesc,
    IDMLDevice* device,
    StackAllocator<ALLOCATOR_SIZE>& allocator,
    std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
    std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
    std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
    std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges,
    std::vector<Microsoft::WRL::ComPtr<IDMLOperator>>& dmlOperators,
    const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToMainGraphInputIndex,
    const std::unordered_map<std::string_view, uint32_t>* serializedGraphConstantNameToMainGraphInputIndex,
    bool forceScaleZeroPointToConstScalars,
    std::vector<std::vector<std::uint8_t>>& constDataVectors)
{
    // Temporary code to force scale and zero point tensors to scalars.
    // This should be removed after re-quantizing test models with scalar scale and zero point
    auto ForceInputToScalar = [](DML_OPERATOR_TYPE type, uint32_t index)
    {
        switch (type)
        {
            case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION:
                return index == 1 || index == 2 || index == 4 || index == 5 || index == 7 || index == 8;

            case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY:
                return index == 1 || index == 2 || index == 4 || index == 5 || index == 7 || index == 8;

            case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR:
                return index == 1 || index == 2;

            case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR:
                return index == 1 || index == 2;

            case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD:
                return index == 1 || index == 2 || index == 4 || index == 5 || index == 6 || index == 7;

            case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING:
                return index == 1 || index == 2 || index == 3 || index == 4;
        }

        return false;
    };

    // Temporary code to track constant input indices to generate fake data for and fix tensor flags.
    // This should be removed when de/serialization is fixed.
    std::map<uint32_t, std::vector<uint32_t>> nodeIndexToConstantInputIndicesMap;
    for (uint32_t i = 0; i < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); ++i)
    {
        DmlSerializedGraphNodeDescVariant descVariant = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Desc;
        bool isConstantEdge = std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(descVariant);
        auto toNodedesc = std::get<AbstractOperatorDesc>(graphDesc.Nodes[graphDesc.IntermediateEdges[i].ToNodeIndex].Desc);
        auto toNodeTensor = toNodedesc.GetInputTensors()[graphDesc.IntermediateEdges[i].ToNodeInputIndex];

        if (isConstantEdge && ForceInputToScalar(toNodedesc.schema->OperatorType, graphDesc.IntermediateEdges[i].ToNodeInputIndex))
        {
            nodeIndexToConstantInputIndicesMap[graphDesc.IntermediateEdges[i].ToNodeIndex].push_back(graphDesc.IntermediateEdges[i].ToNodeInputIndex);
        }
    }

    std::unordered_map<uint32_t, uint32_t> oldNodeIndexToNewNodeIndexMap;
    for (uint32_t index = 0; index < static_cast<uint32_t>(graphDesc.Nodes.size()); index++)
    {
        const DmlSerializedGraphNode& node = graphDesc.Nodes[index];
        if (std::holds_alternative<AbstractOperatorDesc>(node.Desc))
        {
            oldNodeIndexToNewNodeIndexMap[index] = static_cast<uint32_t>(dmlGraphNodes.size());

            // Temporary code to fix tensor flags for constant inputs.
            // This should be removed when de/serialization is fixed.
            AbstractOperatorDesc desc = std::get<AbstractOperatorDesc>(node.Desc);
            for (uint32_t constantInputIndex : nodeIndexToConstantInputIndicesMap[index])
            {
                desc.GetInputTensors()[constantInputIndex]->strides = std::vector<uint32_t>(desc.GetInputTensors()[constantInputIndex]->sizes.size());
                desc.GetInputTensors()[constantInputIndex]->totalTensorSizeInBytes = desc.GetInputTensors()[constantInputIndex]->CalculateBufferSizeInBytes();
                desc.GetInputTensors()[constantInputIndex]->flags &= ~DML_TENSOR_FLAG_OWNED_BY_DML;
            }

            DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc<ALLOCATOR_SIZE>(desc, &allocator);
            ComPtr<IDMLOperator> op;
            THROW_IF_FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op)));
            dmlOperators.push_back(op);
            DML_OPERATOR_GRAPH_NODE_DESC* dmlOperatorGraphNode = allocator.template Allocate<DML_OPERATOR_GRAPH_NODE_DESC>();
            dmlOperatorGraphNode->Name = node.Name.data();
            dmlOperatorGraphNode->Operator = op.Get();
            dmlGraphNodes.push_back(DML_GRAPH_NODE_DESC{DML_GRAPH_NODE_TYPE_OPERATOR, dmlOperatorGraphNode});
        }
    }

    GraphEdgeIndexInfo graphInputIndexInfo = {};

    for (size_t i = 0; i < graphDesc.InputEdges.size(); ++i)
    {
        DML_INPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INPUT_GRAPH_EDGE_DESC>();
        // 1. If serializedGraphInputIndexToMainGraphInputIndex is not null:
        //      then use the corresponding main graph input index, because the caller will use corresponding 
        //      main graph input index for extracting the actual input tensor from the main graph and 
        //      the caller does not own the creation of dml bindings directly.
        //      Use Case: When the caller is ORT (DML EP) or DmlEngine.
        // 
        // 2. If serializedGraphInputIndexToMainGraphInputIndex is null:
        //      then assign the sequential graph input index, because the owns the creationg of dml bindings
        //      directly.
        edge->GraphInputIndex = serializedGraphInputIndexToMainGraphInputIndex == nullptr ?
            graphDesc.InputEdges[i].GraphInputIndex :
            serializedGraphInputIndexToMainGraphInputIndex->at(graphDesc.InputEdges[i].GraphInputIndex);
        edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.InputEdges[i].ToNodeIndex];
        edge->ToNodeInputIndex = graphDesc.InputEdges[i].ToNodeInputIndex;
        edge->Name = graphDesc.InputEdges[i].Name.data();

        graphInputIndexInfo.maxIndex = std::max(graphInputIndexInfo.maxIndex, edge->GraphInputIndex);
        graphInputIndexInfo.hasEdge = true;
        dmlInputEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, edge});
    }

    for (size_t i = 0; i < graphDesc.OutputEdges.size(); ++i)
    {
        DML_OUTPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_OUTPUT_GRAPH_EDGE_DESC>();
        edge->GraphOutputIndex = graphDesc.OutputEdges[i].GraphOutputIndex;
        edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.OutputEdges[i].FromNodeIndex];
        edge->FromNodeOutputIndex = graphDesc.OutputEdges[i].FromNodeOutputIndex;
        edge->Name = graphDesc.OutputEdges[i].Name.data();

        dmlOutputEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, edge});
    }

    std::unordered_map<std::string_view, uint32_t> localConstantNameToIndexMap;
    for (uint32_t i = 0; i < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); ++i)
    {
        DmlSerializedGraphNodeDescVariant descVariant = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Desc;
        bool isConstantEdge = std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(descVariant);

        auto toNodeDesc = std::get<AbstractOperatorDesc>(graphDesc.Nodes[graphDesc.IntermediateEdges[i].ToNodeIndex].Desc);
        auto toNodeTensor = toNodeDesc.GetInputTensors()[graphDesc.IntermediateEdges[i].ToNodeInputIndex];

        auto& constantInputs = nodeIndexToConstantInputIndicesMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
        if (isConstantEdge && 
           std::find(
            constantInputs.begin(),
            constantInputs.end(),
            graphDesc.IntermediateEdges[i].ToNodeInputIndex) == constantInputs.end())
        {
            const std::string& constantName = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Name;
            
            DML_INPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INPUT_GRAPH_EDGE_DESC>();
            edge->GraphInputIndex = GetConstantNodeGraphInputIndex(
                constantName,
                serializedGraphConstantNameToMainGraphInputIndex,
                graphInputIndexInfo,
                localConstantNameToIndexMap);
            edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
            edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
            edge->Name = graphDesc.IntermediateEdges[i].Name.data();

            dmlInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, edge});
        }
        else
        {
            size_t fromNodeIndex;

            if (isConstantEdge)
            {
                size_t dataSize = (TensorUtil::GetDataTypeSize(toNodeTensor->dataType) + 3) & ~3;

                // Temporary code to generate fake data for constant nodes.
                // This should be removed when de/serialization is fixed.
                constDataVectors.push_back(std::vector<uint8_t>(dataSize));

                DML_CONSTANT_DATA_GRAPH_NODE_DESC* dmlConstNode = allocator.template Allocate<DML_CONSTANT_DATA_GRAPH_NODE_DESC>();
                dmlConstNode->Data = constDataVectors.back().data();
                dmlConstNode->DataSize = dataSize;
                dmlConstNode->Name = nullptr;
                dmlGraphNodes.push_back(DML_GRAPH_NODE_DESC{(DML_GRAPH_NODE_TYPE) DML_GRAPH_NODE_TYPE_CONSTANT, dmlConstNode});

                fromNodeIndex = dmlGraphNodes.size() - 1;
            }
            else
            {
                fromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].FromNodeIndex];
            }

            DML_INTERMEDIATE_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INTERMEDIATE_GRAPH_EDGE_DESC>();
            edge->FromNodeIndex = fromNodeIndex;
            edge->FromNodeOutputIndex = graphDesc.IntermediateEdges[i].FromNodeOutputIndex;
            edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
            edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
            edge->Name = graphDesc.IntermediateEdges[i].Name.data();
            dmlIntermediateEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INTERMEDIATE, edge});
        }
    }

    dmlGraphDesc.InputCount = graphInputIndexInfo.hasEdge ? graphInputIndexInfo.maxIndex + 1 : 0;
    dmlGraphDesc.OutputCount = graphDesc.OutputCount;
    dmlGraphDesc.NodeCount = gsl::narrow_cast<uint32_t>(dmlGraphNodes.size());
    dmlGraphDesc.Nodes = dmlGraphNodes.data();
    dmlGraphDesc.InputEdgeCount = gsl::narrow_cast<uint32_t>(dmlInputEdges.size());
    dmlGraphDesc.InputEdges = dmlInputEdges.data();
    dmlGraphDesc.OutputEdgeCount = gsl::narrow_cast<uint32_t>(dmlOutputEdges.size());
    dmlGraphDesc.OutputEdges = dmlOutputEdges.data();
    dmlGraphDesc.IntermediateEdgeCount = gsl::narrow_cast<uint32_t>(dmlIntermediateEdges.size());
    dmlGraphDesc.IntermediateEdges = dmlIntermediateEdges.data();
}
