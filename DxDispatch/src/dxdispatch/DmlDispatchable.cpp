#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "DmlDispatchable.h"
#include "DirectMLHelpers/DmlGraphHelper.h"
#include "DirectMLHelpers/DmlGraphDeserialization.h"

using Microsoft::WRL::ComPtr;

DmlDispatchable::DmlDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::DmlDispatchableDesc& desc,
    const Dispatchable::Bindings& initBindings,
    IDxDispatchLogger* logger
    ) : m_name(name), m_device(device), m_desc(desc), 
        m_initBindings(std::move(initBindings)), m_logger(logger), m_isSerializedGraph(false)
{
    m_bindPoints = desc.bindPoints;
    THROW_IF_FAILED(m_device->DML()->CreateOperator(desc.desc, IID_PPV_ARGS(&m_operator)));
}

DmlDispatchable::DmlDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::DmlSerializedGraphDispatchableDesc& desc,
    IDxDispatchLogger* logger)
    : m_name(name), m_device(device), m_desc(desc), m_isSerializedGraph(true)
{
}

struct BindingData
{
    std::vector<DML_BUFFER_BINDING> bufferBindings;
    std::vector<DML_BINDING_DESC> bindingDescs;
};

uint64_t SafeMultiply(uint64_t a, uint64_t b)
{
    if (b != 0 && (a > std::numeric_limits<uint64_t>::max() / b))
    {
        throw std::overflow_error("Overflow in size calculation");
    }
    return a * b;
}

uint64_t CalculateSize(uint64_t elementCount, uint64_t elementSizeInBytes)
{
    uint64_t calculatedSize = SafeMultiply(elementCount, elementSizeInBytes);
    return (calculatedSize + 3) & ~3ull; // Round up to nearest 4 bytes
}


void FillBindingData(
    const std::vector<Model::DmlDispatchableDesc::BindPoint>& bindPoints,
    const Dispatchable::Bindings* initializeBindings,
    const Dispatchable::Bindings* executeBindings,
    BindingData& bindingData,
    bool isSerializedGraph,
    bool bindingForInitialization, 
    std::optional<Model::DmlDispatchableDesc::DmlCompileType> compileType = std::nullopt)
{
    const Dispatchable::Bindings& bindings = bindingForInitialization ? *initializeBindings : *executeBindings;

    uint32_t totalResourceCount = 0;
    for (size_t i = 0; i < bindPoints.size(); i++) { totalResourceCount += bindPoints[i].resourceCount; }

    bindingData.bufferBindings.resize(totalResourceCount);
    bindingData.bindingDescs.resize(totalResourceCount);

    size_t bufferIndex = 0;

    for (size_t i = 0; i < bindPoints.size(); i++)
    {
        auto bindPointName = bindPoints[i].name;
        auto bindingIterator = bindings.find(bindPointName);
        
        if (bindingIterator == bindings.end())
        {
            for (size_t j = 0; j < bindPoints[i].resourceCount; j++)
            {
                if (compileType.has_value() && *compileType == Model::DmlDispatchableDesc::DmlCompileType::DmlCompileGraph && !bindPoints[i].requiredBinding)
                {
                    // Dml Graph will fail if given DML_BINDING_TYPE_NONE for optional bindings not described in the graph.
                    bindingData.bindingDescs.pop_back();
                    bindingData.bufferBindings.pop_back();
                }
                else
                {
                    bindingData.bufferBindings[bufferIndex].Buffer = nullptr;
                    bindingData.bufferBindings[bufferIndex].Offset = 0;
                    bindingData.bufferBindings[bufferIndex].SizeInBytes = 0;
                    bindingData.bindingDescs[bufferIndex].Type = DML_BINDING_TYPE_NONE;
                    bindingData.bindingDescs[bufferIndex].Desc = nullptr;
                    bufferIndex++;
                }
            }
        }
        else
        {
            auto& sources = bindingIterator->second;

            for (auto& source : sources)
            {
                assert(source.resource != nullptr);
                if (isSerializedGraph)
                {
                    // Use SafeMultiply and CalculateSize only for DMLSerialized
                    uint64_t offset = SafeMultiply(source.elementOffset, source.elementSizeInBytes);
                    uint64_t sizeInBytes = CalculateSize(source.elementCount, source.elementSizeInBytes);

                    bindingData.bufferBindings[bufferIndex].Offset = offset;
                    bindingData.bufferBindings[bufferIndex].SizeInBytes = sizeInBytes - offset;

                    // Validation
                    if (offset + SafeMultiply(source.elementCount, source.elementSizeInBytes) > sizeInBytes)
                    {
                        throw std::invalid_argument(fmt::format(
                            "Buffer size ({} bytes) is too small for the data ({} bytes) at offset {} bytes for binding point '{}'", 
                            sizeInBytes,
                            SafeMultiply(source.elementCount, source.elementSizeInBytes),
                            offset,
                            bindPointName));
                    }
                }
                else
                {
                    assert(source.resourceDesc != nullptr);
                    
                    if (!std::holds_alternative<Model::BufferDesc>(source.resourceDesc->value))
                    {
                        throw std::invalid_argument("DML operators only support buffer bindings");
                    }

                    auto& bufferDesc = std::get<Model::BufferDesc>(source.resourceDesc->value);
                    bindingData.bufferBindings[bufferIndex].SizeInBytes = bufferDesc.sizeInBytes - bindingData.bufferBindings[bufferIndex].Offset;
                }

                bindingData.bufferBindings[bufferIndex].Buffer = source.resource;
                bindingData.bindingDescs[bufferIndex].Type = DML_BINDING_TYPE_BUFFER;
                bindingData.bindingDescs[bufferIndex].Desc = &bindingData.bufferBindings[bufferIndex];
                bufferIndex++;
            }
        }
    }
}

uint32_t GetElementSize(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
        case DML_TENSOR_DATA_TYPE_FLOAT32: return 4;
        case DML_TENSOR_DATA_TYPE_FLOAT16: return 2;
        case DML_TENSOR_DATA_TYPE_UINT32: return 4;
        case DML_TENSOR_DATA_TYPE_UINT16: return 2;
        case DML_TENSOR_DATA_TYPE_UINT8: return 1;
        case DML_TENSOR_DATA_TYPE_INT32: return 4;
        case DML_TENSOR_DATA_TYPE_INT16: return 2;
        case DML_TENSOR_DATA_TYPE_INT8: return 1;
        default: throw std::runtime_error("Unknown data type");
    }
}

static Model::DmlDispatchableDesc::BindPoints GetSerializedBindPoints(const DmlSerializedGraphDesc& serializedDesc)
{
    Model::DmlDispatchableDesc::BindPoints result;
    
    for (const auto& inputEdge : serializedDesc.InputEdges) 
    {
        result.inputs.push_back({inputEdge.Name, 1, true});
    }

    for (const auto& node : serializedDesc.Nodes)
    {
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(node.Desc))
        {
            const auto& constantVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
            if (std::holds_alternative<ConstantName>(constantVariant))
            {
                result.inputs.push_back({node.Name, 1, true});
            }
        }
    }
    
    for (const auto& outputEdge : serializedDesc.OutputEdges)
    {
        result.outputs.push_back({outputEdge.Name, 1, true});
    }
    
    return result;
}

std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> ExtractConstantDataTypes(const DmlSerializedGraphDesc& serializedDesc)
{
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> constantDataTypes;
    std::unordered_map<uint32_t, DmlIntermediateSerializedGraphEdge> constantNodeEdges;

    for (const auto& edge : serializedDesc.IntermediateEdges) 
    {
        const auto& nodeDesc = serializedDesc.Nodes[edge.FromNodeIndex].Desc;
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(nodeDesc)) 
        {
            const auto& constantVariant = std::get<DmlSerializedGraphNodeConstantVariant>(nodeDesc);
            if (std::holds_alternative<ConstantName>(constantVariant)) 
            {
                constantNodeEdges[edge.FromNodeIndex] = edge;
            }
        }
    }

        for (uint32_t nodeIndex = 0; nodeIndex < serializedDesc.Nodes.size(); nodeIndex++) 
        {
            const auto& node = serializedDesc.Nodes[nodeIndex];
            if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(node.Desc)) 
            {
                const auto& constantVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
                if (std::holds_alternative<ConstantName>(constantVariant) && constantNodeEdges.find(nodeIndex) != constantNodeEdges.end()) 
                {
                    const auto& edge = constantNodeEdges[nodeIndex];
                    const auto& operatorNode = serializedDesc.Nodes[edge.ToNodeIndex];
                    if (std::holds_alternative<AbstractOperatorDesc>(operatorNode.Desc)) 
                    {
                        const auto& opDesc = std::get<AbstractOperatorDesc>(operatorNode.Desc);
                        auto inputTensors = opDesc.GetInputTensors();
                        if (edge.ToNodeInputIndex < inputTensors.size()) 
                        {
                            const auto& constantTensor = inputTensors[edge.ToNodeInputIndex];
                            constantDataTypes[node.Name] = constantTensor->dataType;
                        }
                    }
                }
            }
        }

        return constantDataTypes;
}

Dispatchable::Bindings GenerateInitialBindingsFromGraph(const DmlSerializedGraphDesc& serializedDesc,
const std::unordered_map<std::string, DML_TENSOR_DATA_TYPE>& constantDataTypes)
{
    Dispatchable::Bindings local_bindings;

    for (const auto& node : serializedDesc.Nodes)
    {
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(node.Desc))
        {
            const auto& constantVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
            if (std::holds_alternative<ConstantName>(constantVariant))
            {
                Dispatchable::BindingSource bindingSource;
                bindingSource.resource = nullptr;  
                bindingSource.resourceDesc = nullptr;
                bindingSource.elementOffset = 0;
                bindingSource.elementCount = 0;  
                bindingSource.elementSizeInBytes = GetElementSize(constantDataTypes.at(node.Name));
                local_bindings[node.Name] = {bindingSource};
            }
        }
    }

    return local_bindings;
}

static std::vector<std::byte> LoadFileContents(const std::filesystem::path& filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + filepath.string());
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> buffer(size);
    
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        throw std::runtime_error("Could not read file: " + filepath.string());
    }
    return buffer;
}

void DmlDispatchable::CreateResourceFromConstantNode(
    const DmlSerializedGraphNode& node,
    const std::unordered_map<std::string, DML_TENSOR_DATA_TYPE>& constantDataTypes,
    const std::variant<Model::DmlDispatchableDesc, Model::DmlSerializedGraphDispatchableDesc>& m_desc,
    std::shared_ptr<Device> m_device,  
    Dispatchable::Bindings& m_initBindings)
{
    const auto& constantVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
   
    std::vector<std::byte> data;
    if (std::holds_alternative<ConstantName>(constantVariant))
    {
        const auto& constantName = std::get<ConstantName>(constantVariant);

        const auto& desc = std::get<Model::DmlSerializedGraphDispatchableDesc>(m_desc);
        std::filesystem::path filePath = desc.sourcePath.parent_path() / (constantName.name + ".bin");
        data = LoadFileContents(filePath);
    
        auto wName = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(node.Name);
        auto d3d12Resource = m_device->Upload(data.size(), data, wName);
        if (!d3d12Resource) throw std::runtime_error("Failed to create resource for constant node: " + node.Name);

        m_resources[node.Name] = std::move(d3d12Resource);

        if (m_initBindings.find(node.Name) != m_initBindings.end() && !m_initBindings[node.Name].empty())
        {
            auto& bindingSource = m_initBindings[node.Name][0];
            bindingSource.resource = m_resources[node.Name].Get();
            bindingSource.elementCount = data.size() / GetElementSize(constantDataTypes.at(node.Name));
        }
    }
}

void DmlDispatchable::BuildAndCompileGraph()
{
    const auto& desc = std::get<Model::DmlSerializedGraphDispatchableDesc>(m_desc);
    
    // Deserialize the graph
    std::ifstream inFile(desc.sourcePath, std::ios::binary | std::ios::ate);
    if (!inFile)
    {
        throw std::invalid_argument(fmt::format("Could not open the graph file for DmlSerializedGraph dispatchable", desc.sourcePath.string()));
    }
    std::streampos fileSize = inFile.tellg();
    std::vector<uint8_t> blob(gsl::narrow_cast<size_t>(fileSize));
    inFile.seekg(0, std::ios::beg);
    inFile.read(reinterpret_cast<char*>(blob.data()), fileSize);

    std::vector<std::unique_ptr<std::byte[]>> rawData;
    DmlSerializedGraphDesc serializedDesc = DeserializeDmlGraph(blob.data(), rawData);
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> constantDataTypes;

    m_bindPoints = GetSerializedBindPoints(serializedDesc);
    constantDataTypes = ExtractConstantDataTypes(serializedDesc);
    m_initBindings  = GenerateInitialBindingsFromGraph(serializedDesc, constantDataTypes);

    for (const auto& node : serializedDesc.Nodes)
    {
        const auto* constantVariantPtr = std::get_if<DmlSerializedGraphNodeConstantVariant>(&node.Desc);
        if (constantVariantPtr && std::holds_alternative<ConstantName>(*constantVariantPtr))
        {
            CreateResourceFromConstantNode(node, constantDataTypes, desc, m_device, m_initBindings);
        }
    }

    // Convert to Public Graph Description
    StackAllocator<1024> allocator;
    DML_GRAPH_DESC dmlGraphDesc = {};
    std::vector<Microsoft::WRL::ComPtr<IDMLOperator>> dmlOperators;
    std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes;
    std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges;
    std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges;
    std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges;

    ConvertGraphDesc<1024>(
        serializedDesc,
        serializedDesc.InputCount,
        serializedDesc.OutputCount,
        m_device->DML(),
        allocator,
        nullptr, 
        nullptr, 
        dmlGraphDesc,
        dmlOperators,
        dmlGraphNodes,
        dmlInputEdges,
        dmlOutputEdges,
        dmlIntermediateEdges);

    //Compile the graph
    THROW_IF_FAILED(m_device->DML()->CompileGraph(
        &dmlGraphDesc,
        DML_EXECUTION_FLAG_NONE,
        IID_PPV_ARGS(&m_compiledOperator)));
}

void DmlDispatchable::Initialize()
{
  if (std::holds_alternative<Model::DmlDispatchableDesc>(m_desc))
  {
    const auto& dmlDesc = std::get<Model::DmlDispatchableDesc>(m_desc);
    
    if (dmlDesc.compileType == Model::DmlDispatchableDesc::DmlCompileType::DmlCompileOp)
    {
        m_logger->LogInfo("Compile Op");
        THROW_IF_FAILED(m_device->DML()->CompileOperator(
            m_operator.Get(), 
            dmlDesc.executionFlags, 
            IID_PPV_ARGS(m_compiledOperator.ReleaseAndGetAddressOf())));
        m_compiledOperator->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(m_name).data());

        }
    else if (dmlDesc.compileType == Model::DmlDispatchableDesc::DmlCompileType::DmlCompileGraph)
        {
            m_logger->LogInfo("Compiling op using IDMLDevice1::CompileGraph");
            DML_GRAPH_DESC dmlGraphDesc = {};
            std::vector<DML_INPUT_GRAPH_EDGE_DESC> dmlInputGraphEdges;
            std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges;
            
            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> dmlOutputGraphEdges;
            std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges;
            DML_GRAPH_NODE_DESC dmlGraphNodeDesc = {};
            DML_OPERATOR_GRAPH_NODE_DESC nodeDesc{};

            nodeDesc.Operator = m_operator.Get();
            nodeDesc.Name = m_name.c_str();

            {
                dmlGraphNodeDesc.Type = DML_GRAPH_NODE_TYPE_OPERATOR;
                dmlGraphNodeDesc.Desc = &nodeDesc;
            }

            dmlInputGraphEdges.resize(dmlDesc.bindPoints.inputs.size());
            for (size_t i = 0; i < dmlDesc.bindPoints.inputs.size(); i++)
            {
                if (dmlDesc.bindPoints.inputs[i].requiredBinding)
                {
                    DML_INPUT_GRAPH_EDGE_DESC desc = {};
                    desc.GraphInputIndex = gsl::narrow_cast<UINT>(i);
                    desc.ToNodeIndex = 0;
                    desc.ToNodeInputIndex = gsl::narrow_cast<UINT>(i);
                    desc.Name = dmlDesc.bindPoints.inputs[i].name.c_str();
                    dmlInputGraphEdges[i] = desc;
                    dmlInputEdges.push_back({ DML_GRAPH_EDGE_TYPE_INPUT, &dmlInputGraphEdges[i] });
                }
            }

            dmlOutputGraphEdges.resize(dmlDesc.bindPoints.outputs.size());
            for (size_t i = 0; i < dmlDesc.bindPoints.outputs.size(); i++)
            {
                if (dmlDesc.bindPoints.outputs[i].requiredBinding)
                {
                    DML_OUTPUT_GRAPH_EDGE_DESC desc = {};
                    desc.GraphOutputIndex = gsl::narrow_cast<UINT>(i);
                    desc.FromNodeIndex = 0;
                    desc.FromNodeOutputIndex = gsl::narrow_cast<UINT>(i);
                    desc.Name = dmlDesc.bindPoints.outputs[i].name.c_str();
                    dmlOutputGraphEdges[i] = desc;
                    dmlOutputEdges.push_back({ DML_GRAPH_EDGE_TYPE_OUTPUT, &dmlOutputGraphEdges[i] });
                }
            }

            dmlGraphDesc.InputCount = static_cast<uint32_t>(dmlInputEdges.size());
            dmlGraphDesc.InputEdges = dmlInputEdges.data();
            dmlGraphDesc.InputEdgeCount = dmlGraphDesc.InputCount;

            dmlGraphDesc.OutputCount = static_cast<uint32_t>(dmlOutputEdges.size());
            dmlGraphDesc.OutputEdges = dmlOutputEdges.data();
            dmlGraphDesc.OutputEdgeCount = dmlGraphDesc.OutputCount;

            dmlGraphDesc.IntermediateEdgeCount = 0;
            dmlGraphDesc.IntermediateEdges = nullptr;

            dmlGraphDesc.NodeCount = 1;
            dmlGraphDesc.Nodes = &dmlGraphNodeDesc;

            THROW_IF_FAILED(m_device->DML()->CompileGraph(&dmlGraphDesc, dmlDesc.executionFlags, IID_PPV_ARGS(&m_compiledOperator)));
            m_compiledOperator->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(fmt::format("Graph_{}", m_name)).data());
        }   
   }
   else
   {
        BuildAndCompileGraph();
   }

    ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator* ops[] = { m_compiledOperator.Get() };
    THROW_IF_FAILED(m_device->DML()->CreateOperatorInitializer(
        _countof(ops),
        ops,
        IID_PPV_ARGS(&initializer)));

    auto min = initializer->GetBindingProperties().RequiredDescriptorCount;

    // Create a descriptor heap with at least one descriptor. Even if the op doesn't require any descriptors the
    // binding table expects valid descriptor handles.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = std::max(1u, initializer->GetBindingProperties().RequiredDescriptorCount);
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(descriptorHeap.ReleaseAndGetAddressOf())));

    ID3D12DescriptorHeap* descriptorHeaps[] = { descriptorHeap.Get() };
    m_device->GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = initializer.Get();
    bindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = initializer->GetBindingProperties().RequiredDescriptorCount;

    ComPtr<IDMLBindingTable> bindingTable;
    THROW_IF_FAILED(m_device->DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

    // Inputs flagged OWNED_BY_DML must be bound during initialization (and only initialization).
    if (std::holds_alternative<Model::DmlSerializedGraphDispatchableDesc>(m_desc))
    {
        // Initializers can initialize multiple inputs simultaneously, so each compiled op's inputs must
        // be bound using a separate buffer array binding.
        BindingData inputBindingData = {};
        FillBindingData(m_bindPoints.inputs, &m_initBindings, &m_initBindings, inputBindingData, m_isSerializedGraph, true);

        DML_BUFFER_ARRAY_BINDING bufferArrayBindings = {};
        if (inputBindingData.bufferBindings.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument(fmt::format("Initialization Input BindingCount '{}' is too large.", inputBindingData.bufferBindings.size()));
        }
        bufferArrayBindings.BindingCount = static_cast<uint32_t>(inputBindingData.bufferBindings.size());
        bufferArrayBindings.Bindings = inputBindingData.bufferBindings.data();

        DML_BINDING_DESC bindingDesc = {};
        bindingDesc.Desc = &bufferArrayBindings;
        bindingDesc.Type = DML_BINDING_TYPE_BUFFER_ARRAY;

        bindingTable->BindInputs(1, &bindingDesc);
    }
    else if (std::holds_alternative<Model::DmlDispatchableDesc>(m_desc))
    {
        BindingData inputBindingData = {};
        const auto& dmlDesc = std::get<Model::DmlDispatchableDesc>(m_desc);
        FillBindingData(m_bindPoints.inputs, &m_initBindings, nullptr, inputBindingData, m_isSerializedGraph, true, dmlDesc.compileType);

        DML_BUFFER_ARRAY_BINDING bufferArrayBindings = {};
        if (inputBindingData.bufferBindings.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument(fmt::format("Initialization Input BindingCount '{}' is too large.", inputBindingData.bufferBindings.size()));
        }
        bufferArrayBindings.BindingCount = static_cast<uint32_t>(inputBindingData.bufferBindings.size());
        bufferArrayBindings.Bindings = inputBindingData.bufferBindings.data();

        DML_BINDING_DESC bindingDesc = {};
        bindingDesc.Desc = &bufferArrayBindings;
        bindingDesc.Type = DML_BINDING_TYPE_BUFFER_ARRAY;

        bindingTable->BindInputs(1, &bindingDesc);
    }
    // A temporary resource may be required to initialize the operators.
    auto tempBufferSize = initializer->GetBindingProperties().TemporaryResourceSize;
    if (tempBufferSize > 0)
    {
        ComPtr<ID3D12Resource> tempBuffer = m_device->CreatePreferredDeviceMemoryBuffer(tempBufferSize);
        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindTemporaryResource(&bindingDesc);
        m_device->KeepAliveUntilNextCommandListDispatch(std::move(tempBuffer));
    }

    // Each compiled op's persistent resource is bound as an output of the initializer.
    auto persistentBufferSize = m_compiledOperator->GetBindingProperties().PersistentResourceSize;
    if (persistentBufferSize > 0)
    {
        m_persistentBuffer = m_device->CreatePreferredDeviceMemoryBuffer(persistentBufferSize);
        DML_BUFFER_BINDING bufferBinding = { m_persistentBuffer.Get(), 0, persistentBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindOutputs(1, &bindingDesc);
    }

    m_device->KeepAliveUntilNextCommandListDispatch(std::move(descriptorHeap));
    m_device->RecordInitialize(initializer.Get(), bindingTable.Get());
    m_device->ExecuteCommandListAndWait();
}

void DmlDispatchable::Bind(const Bindings& bindings, uint32_t iteration)
{
    auto bindingProps = m_compiledOperator->GetBindingProperties();

    BindingData inputBindingData = {};
    BindingData outputBindingData = {};

    if (std::holds_alternative<Model::DmlSerializedGraphDispatchableDesc>(m_desc))
    {
        FillBindingData(m_bindPoints.inputs, &m_initBindings, &bindings, inputBindingData, m_isSerializedGraph, false);
        FillBindingData(m_bindPoints.outputs, &m_initBindings, &bindings, outputBindingData, m_isSerializedGraph, false);
    }
    else if (std::holds_alternative<Model::DmlDispatchableDesc>(m_desc))
    {
        const auto& dmlDesc = std::get<Model::DmlDispatchableDesc>(m_desc);
        FillBindingData(m_bindPoints.inputs, &m_initBindings, &bindings, inputBindingData, m_isSerializedGraph, false, dmlDesc.compileType);
        FillBindingData(m_bindPoints.outputs, &m_initBindings, &bindings, outputBindingData, m_isSerializedGraph, false, dmlDesc.compileType);
    }
    else
    {
        throw std::runtime_error("Unexpected serialized graph descriptor for non-serialized graph");
    }

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = bindingProps.RequiredDescriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(
        &descriptorHeapDesc, 
        IID_GRAPHICS_PPV_ARGS(m_descriptorHeap.ReleaseAndGetAddressOf())));

    ID3D12DescriptorHeap* descriptorHeaps[] = { m_descriptorHeap.Get() };
    m_device->GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_compiledOperator.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    THROW_IF_FAILED(m_device->DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(m_bindingTable.ReleaseAndGetAddressOf())));

    if (inputBindingData.bindingDescs.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(fmt::format("BindInputs count  '{}' is too large.", inputBindingData.bindingDescs.size()));
    }
    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingData.bindingDescs.size()), inputBindingData.bindingDescs.data());

    ComPtr<ID3D12Resource> tempBuffer;
    auto tempBufferSize = bindingProps.TemporaryResourceSize;
    if (tempBufferSize > 0)
    {
        tempBuffer = m_device->CreatePreferredDeviceMemoryBuffer(tempBufferSize);

        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        m_bindingTable->BindTemporaryResource(&bindingDesc);
        m_device->KeepAliveUntilNextCommandListDispatch(std::move(tempBuffer)); 
    }

    auto persistentBufferSize = bindingProps.PersistentResourceSize;
    if (persistentBufferSize > 0)
    {
        DML_BUFFER_BINDING bufferBinding = { m_persistentBuffer.Get(), 0, persistentBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }
    if (outputBindingData.bindingDescs.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(fmt::format("BindOutputs count  '{}' is too large.", outputBindingData.bindingDescs.size()));
    }
    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingData.bindingDescs.size()), outputBindingData.bindingDescs.data());

    // DML may remove the device if invalid bindings are specified.
    THROW_IF_FAILED(m_device->DML()->GetDeviceRemovedReason());
}

void DmlDispatchable::Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings)
{
    m_device->RecordDispatch(m_compiledOperator.Get(), m_bindingTable.Get());
    m_device->ExecuteCommandListAndWait();
}