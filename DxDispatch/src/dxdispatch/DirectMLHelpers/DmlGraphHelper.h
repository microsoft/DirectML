// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include <queue>
#include "AbstractOperatorDesc.h"
#include "SchemaHelpers.h"
#include "DmlSerializedGraphDesc.h"
#include "Utility.h"

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
            localConstantNameToIndexMap[constantName] = graphInputIndexInfo.hasEdge ? (++graphInputIndexInfo.maxIndex) : graphInputIndexInfo.maxIndex;
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

static std::map<uint32_t, std::vector<uint32_t>> GenerateNodeIndexToForcedConstantNameInputIndicesMap(
    const DmlSerializedGraphDesc& graphDesc,
    bool forceScaleZeroPointWithout1DMetacommandSupportToConstScalars,
    bool forceAllScaleZeroPointToConstScalars)
{
    // Temporary code to force scale and zero point tensors to scalars.
    // This should be removed after re-quantizing test models with scalar scale and zero point
    auto ForceInputToScalar = [forceScaleZeroPointWithout1DMetacommandSupportToConstScalars, forceAllScaleZeroPointToConstScalars](DML_OPERATOR_TYPE type, uint32_t index)
    {
        if (forceAllScaleZeroPointToConstScalars || forceScaleZeroPointWithout1DMetacommandSupportToConstScalars)
        {
            switch (type)
            {
                case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION:
                {
                    if (forceAllScaleZeroPointToConstScalars)
                    {
                        return index == 1 || index == 2 || index == 4 || index == 5 || index == 7 || index == 8;
                    }
                    else
                    {
                        return index == 1 || index == 2 || index == 5 || index == 7 || index == 8;
                    }
                }

                case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY:
                {
                    if (forceAllScaleZeroPointToConstScalars)
                    {
                        return index == 1 || index == 2 || index == 4 || index == 5 || index == 6 || index == 7;
                    }
                    else
                    {
                        return index == 1 || index == 2 || index == 5 || index == 6 || index == 7;
                    }
                }

                case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR:
                    return index == 1 || index == 2;

                case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR:
                    return index == 1 || index == 2;

                case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD:
                    return index == 1 || index == 2 || index == 4 || index == 5 || index == 6 || index == 7;

                case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING:
                    return index == 1 || index == 2 || index == 3 || index == 4;
            }
        }

        return false;
    };

    // Temporary code to track constant input indices to generate fake data for and fix tensor flags.
    // This should be removed when de/serialization is fixed.
    std::map<uint32_t, std::vector<uint32_t>> nodeIndexToConstantInputIndicesMap;
    for (uint32_t i = 0; i < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); ++i)
    {
        DmlSerializedGraphNodeDescVariant descVariant = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Desc;
        bool isConstantNameEdge = std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(descVariant) &&
                              std::holds_alternative<ConstantName>(std::get<DmlSerializedGraphNodeConstantVariant>(descVariant));
        auto toNodedesc = std::get<AbstractOperatorDesc>(graphDesc.Nodes[graphDesc.IntermediateEdges[i].ToNodeIndex].Desc);
        
        if (isConstantNameEdge && ForceInputToScalar(toNodedesc.schema->OperatorType, graphDesc.IntermediateEdges[i].ToNodeInputIndex))
        {
            nodeIndexToConstantInputIndicesMap[graphDesc.IntermediateEdges[i].ToNodeIndex].push_back(graphDesc.IntermediateEdges[i].ToNodeInputIndex);
        }
    }

    return nodeIndexToConstantInputIndicesMap;
}

template <size_t AllocatorSize>
void ConvertGraphDesc(
    const DmlSerializedGraphDesc& graphDesc,
    const uint32_t inputCount,
    const uint32_t outputCount,
    IDMLDevice* device,
    StackAllocator<AllocatorSize>& allocator,
    const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToSubgraphInputIndex,
    const std::unordered_map<std::string_view, uint32_t>* serializedGraphLargeConstantNameToSubgraphInputIndex,
    _Out_ DML_GRAPH_DESC& dmlGraphDesc,
    _Inout_ std::vector<Microsoft::WRL::ComPtr<IDMLOperator>>& dmlOperators,
    _Inout_ std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
    _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
    _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
    _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges)
{
    std::unordered_map<uint32_t, uint32_t> oldNodeIndexToNewNodeIndexMap;
    for (uint32_t index = 0; index < static_cast<uint32_t>(graphDesc.Nodes.size()); index++)
    {
        const DmlSerializedGraphNode& node = graphDesc.Nodes[index];
        if (std::holds_alternative<AbstractOperatorDesc>(node.Desc))
        {
            oldNodeIndexToNewNodeIndexMap[index] = static_cast<uint32_t>(dmlGraphNodes.size());
            DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc<AllocatorSize>(std::get<AbstractOperatorDesc>(node.Desc), &allocator);
            ComPtr<IDMLOperator> op;
            ORT_THROW_IF_FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op)));
            dmlOperators.push_back(op);
            DML_OPERATOR_GRAPH_NODE_DESC* dmlOperatorGraphNode = allocator.template Allocate<DML_OPERATOR_GRAPH_NODE_DESC>();
            dmlOperatorGraphNode->Name = node.Name.data();
            dmlOperatorGraphNode->Operator = op.Get();
            dmlGraphNodes.push_back(DML_GRAPH_NODE_DESC{ DML_GRAPH_NODE_TYPE_OPERATOR, dmlOperatorGraphNode });
        }
        else
        {
            auto& constantNodeVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
            if (std::holds_alternative<ConstantData>(constantNodeVariant))
            {
                oldNodeIndexToNewNodeIndexMap[index] = static_cast<uint32_t>(dmlGraphNodes.size());

                auto& constantData = std::get<ConstantData>(constantNodeVariant);

                DML_CONSTANT_DATA_GRAPH_NODE_DESC* constantNode = allocator.template Allocate<DML_CONSTANT_DATA_GRAPH_NODE_DESC>();
                constantNode->Name = node.Name.data();
                constantNode->DataSize = constantData.dataSize;
                constantNode->Data = constantData.data;
                dmlGraphNodes.push_back(DML_GRAPH_NODE_DESC{ DML_GRAPH_NODE_TYPE_CONSTANT, constantNode });
            }
        }
    }

    uint32_t graphMaxInputIndex = 0;

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
        //      then assign the sequential graph input index, because it owns the creation of dml bindings
        //      directly.
        edge->GraphInputIndex = serializedGraphInputIndexToSubgraphInputIndex == nullptr ?
            graphDesc.InputEdges[i].GraphInputIndex :
            serializedGraphInputIndexToSubgraphInputIndex->at(graphDesc.InputEdges[i].GraphInputIndex);
        edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.InputEdges[i].ToNodeIndex];
        edge->ToNodeInputIndex = graphDesc.InputEdges[i].ToNodeInputIndex;
        edge->Name = graphDesc.InputEdges[i].Name.data();

        graphMaxInputIndex = std::max(graphMaxInputIndex, edge->GraphInputIndex);
        dmlInputEdges.push_back(DML_GRAPH_EDGE_DESC{ DML_GRAPH_EDGE_TYPE_INPUT, edge });
    }

    for (size_t i = 0; i < graphDesc.OutputEdges.size(); ++i)
    {
        DML_OUTPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_OUTPUT_GRAPH_EDGE_DESC>();
        edge->GraphOutputIndex = graphDesc.OutputEdges[i].GraphOutputIndex;
        edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.OutputEdges[i].FromNodeIndex];
        edge->FromNodeOutputIndex = graphDesc.OutputEdges[i].FromNodeOutputIndex;
        edge->Name = graphDesc.OutputEdges[i].Name.data();

        dmlOutputEdges.push_back(DML_GRAPH_EDGE_DESC{ DML_GRAPH_EDGE_TYPE_OUTPUT, edge });
    }

    std::unordered_map<std::string_view, uint32_t> localConstantNameToIndexMap;
    for (uint32_t i = 0; i < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); ++i)
    {
        DmlSerializedGraphNodeDescVariant descVariant = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Desc;
        bool isConstantEdge = std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(descVariant);
        if (isConstantEdge)
        {
            auto& constantNodeVariant = std::get<DmlSerializedGraphNodeConstantVariant>(descVariant);
            if (std::holds_alternative<ConstantData>(constantNodeVariant))
            {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INTERMEDIATE_GRAPH_EDGE_DESC>();
                edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].FromNodeIndex];
                edge->FromNodeOutputIndex = graphDesc.IntermediateEdges[i].FromNodeOutputIndex;
                edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
                edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
                edge->Name = graphDesc.IntermediateEdges[i].Name.data();
                dmlIntermediateEdges.push_back(DML_GRAPH_EDGE_DESC{ DML_GRAPH_EDGE_TYPE_INTERMEDIATE, edge });
            }
            else
            {
                const std::string& constantName = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Name;

                DML_INPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INPUT_GRAPH_EDGE_DESC>();
                edge->GraphInputIndex = GetConstantNodeGraphInputIndex(
                    constantName,
                    serializedGraphLargeConstantNameToSubgraphInputIndex,
                    graphMaxInputIndex,
                    localConstantNameToIndexMap);
                edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
                edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
                edge->Name = graphDesc.IntermediateEdges[i].Name.data();

                dmlInputEdges.push_back({ DML_GRAPH_EDGE_TYPE_INPUT, edge });
            }
        }
        else
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INTERMEDIATE_GRAPH_EDGE_DESC>();
            edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].FromNodeIndex];
            edge->FromNodeOutputIndex = graphDesc.IntermediateEdges[i].FromNodeOutputIndex;
            edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
            edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
            edge->Name = graphDesc.IntermediateEdges[i].Name.data();
            dmlIntermediateEdges.push_back(DML_GRAPH_EDGE_DESC{ DML_GRAPH_EDGE_TYPE_INTERMEDIATE, edge });
        }
    }

    dmlGraphDesc.InputCount = inputCount;
    dmlGraphDesc.OutputCount = outputCount;
    dmlGraphDesc.NodeCount = gsl::narrow_cast<uint32_t>(dmlGraphNodes.size());
    dmlGraphDesc.Nodes = dmlGraphNodes.data();
    dmlGraphDesc.InputEdgeCount = gsl::narrow_cast<uint32_t>(dmlInputEdges.size());
    dmlGraphDesc.InputEdges = dmlInputEdges.data();
    dmlGraphDesc.OutputEdgeCount = gsl::narrow_cast<uint32_t>(dmlOutputEdges.size());
    dmlGraphDesc.OutputEdges = dmlOutputEdges.data();
    dmlGraphDesc.IntermediateEdgeCount = gsl::narrow_cast<uint32_t>(dmlIntermediateEdges.size());
    dmlGraphDesc.IntermediateEdges = dmlIntermediateEdges.data();
}

inline void PerformTopologicalSortAndCheckIsAcyclic(
    const DmlSerializedGraphDesc& graphDesc,
    std::vector<uint32_t>& nodesInTopologicalOrder)
{
    uint32_t nodeCount = static_cast<uint32_t>(graphDesc.Nodes.size());
    std::queue<uint32_t> queue;
    std::vector<uint32_t> inDegree(nodeCount, 0);
    std::vector<std::vector<uint32_t>> children(nodeCount);

    // Don't need to iterate through InputEdges because those inputs don't represent any node
    // and the purpose of this topological sort is come up a order to correctly iterate through nodes .
    for (const DmlIntermediateSerializedGraphEdge& intermediateEdge : graphDesc.IntermediateEdges)
    {
        inDegree[intermediateEdge.ToNodeIndex]++;
        children[intermediateEdge.FromNodeIndex].push_back(intermediateEdge.ToNodeIndex);
    }

    for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
    {
        if (inDegree[nodeIndex] == 0)
        {
            queue.push(nodeIndex);
        }
    }

    uint32_t nodeIndex = 0;
    while (!queue.empty())
    {
        if (nodeIndex >= nodeCount)
        {
            throw std::invalid_argument("Given graph is not acyclic.");
        }

        uint32_t currNodeIndex = queue.front();
        queue.pop();
        nodesInTopologicalOrder[nodeIndex++] = currNodeIndex;

        for (uint32_t child : children[currNodeIndex])
        {
            if (--inDegree[child] == 0)
            {
                queue.push(child);
            }
        }
    }
}
