#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "DmlSerializedGraphDispatchable.h"
#include "../DirectMLHelpers/DmlGraphHelper.h"
#include "../DirectMLHelpers/DmlGraphDeserialization.h"


using Microsoft::WRL::ComPtr;

DmlSerializedGraphDispatchable::DmlSerializedGraphDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::DmlSerializedGraphDispatchableDesc& desc) :
          m_name(name), m_device(device), m_desc(desc)
{
}
struct BindPoint
    {
        std::string name;
        uint32_t resourceCount;
        bool required;
    };

struct BindPoints
{
    std::vector<BindPoint> inputs;
    std::vector<BindPoint> outputs;
};

DmlSerializedGraphDispatchable::BindPoints DmlSerializedGraphDispatchable::GetBindPoints(const DmlSerializedGraphDesc& serializedDesc)
{
    BindPoints bindPoints;
    for (const auto& inputEdge : serializedDesc.InputEdges) 
    {
        bindPoints.inputs.push_back({
            inputEdge.Name,  
            1,               
            true             
        });
    }
    for (const auto& outputEdge : serializedDesc.OutputEdges)
    {
        bindPoints.outputs.push_back({
            outputEdge.Name, 
            1,               
            true             
        });
    }
    return bindPoints;
}

std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> DmlSerializedGraphDispatchable::ExtractConstantDataTypes(const DmlSerializedGraphDesc& serializedDesc)
{
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> constantDataTypes;
    std::unordered_map<uint32_t, DmlIntermediateSerializedGraphEdge> constantNodeEdges;

    for (const auto& edge : serializedDesc.IntermediateEdges) {
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(serializedDesc.Nodes[edge.FromNodeIndex].Desc)) {
            constantNodeEdges[edge.FromNodeIndex] = edge;
        }
    }

    for (uint32_t nodeIndex = 0; nodeIndex < serializedDesc.Nodes.size(); nodeIndex++) {
        const auto& node = serializedDesc.Nodes[nodeIndex];
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(node.Desc)) {
            if (constantNodeEdges.find(nodeIndex) != constantNodeEdges.end()) {
                const auto& edge = constantNodeEdges[nodeIndex];
                const auto& operatorNode = serializedDesc.Nodes[edge.ToNodeIndex];
                if (std::holds_alternative<AbstractOperatorDesc>(operatorNode.Desc)) {
                    const auto& opDesc = std::get<AbstractOperatorDesc>(operatorNode.Desc);
                    auto inputTensors = opDesc.GetInputTensors();
                    if (edge.ToNodeInputIndex < inputTensors.size()) {
                        const auto& constantTensor = inputTensors[edge.ToNodeInputIndex];
                        constantDataTypes[node.Name] = constantTensor->dataType;
                    }
                }
            }
        }
    }

    return constantDataTypes;
}

Dispatchable::Bindings DmlSerializedGraphDispatchable::GenerateInitialBindingsFromGraph(const DmlSerializedGraphDesc& serializedDesc)
{
    Dispatchable::Bindings local_bindings;

    for (const auto& node : serializedDesc.Nodes)
    {
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(node.Desc))
        {
            Dispatchable::BindingSource bindingSource;
            bindingSource.resource = nullptr;  // will set later in CreateResourceFromConstantNode
            bindingSource.resourceDesc = nullptr;
            bindingSource.elementOffset = 0;
            bindingSource.elementCount = 0;  // will set in CreateResourceFromConstantNode
            bindingSource.elementSizeInBytes = GetElementSize(m_constantDataTypes[node.Name]);
            // ,aybe add other fields
            local_bindings[node.Name] = {bindingSource};
        }
    }

    return local_bindings;
}

std::vector<std::byte> DmlSerializedGraphDispatchable::LoadFileContents(const std::filesystem::path& filepath)
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

void DmlSerializedGraphDispatchable::CreateResourceFromConstantNode(const DmlSerializedGraphNode& node)
{
    const auto& constantVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
    
    std::vector<std::byte> data;
    if (std::holds_alternative<ConstantName>(constantVariant))
    {
        const auto& constantName = std::get<ConstantName>(constantVariant);
        std::string fileName = constantName.name + /*/"/*_0*/".bin";
        data = LoadFileContents(m_desc.sourcePath.parent_path() / fileName);
    }
    else if (std::holds_alternative<ConstantData>(constantVariant))
    {
        const auto& constantData = std::get<ConstantData>(constantVariant);
        data.assign(constantData.data, constantData.data + constantData.dataSize);
    }
    
    auto wName = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(node.Name);
    auto d3d12Resource = m_device->Upload(data.size(), data, wName);
    m_resources[node.Name] = std::move(d3d12Resource);

    //Update the binding from generate bindings  function with the created resource
    if (m_bindings.find(node.Name) != m_bindings.end() && !m_bindings[node.Name].empty())
    {
        auto& bindingSource = m_bindings[node.Name][0];
        bindingSource.resource = m_resources[node.Name].Get();
        bindingSource.elementCount = data.size() / GetElementSize(m_constantDataTypes[node.Name]);
        //Other fields update here 
    }
}

uint32_t DmlSerializedGraphDispatchable::GetElementSize(DML_TENSOR_DATA_TYPE dataType)
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

void DmlSerializedGraphDispatchable::BuildGraph() 
{
    std::ifstream inFile(m_desc.sourcePath, std::ios::binary | std::ios::ate);
    if (!inFile)
    {
        throw std::invalid_argument("Could not open the graph file for DmlSerializedGraph dispatchable");
    }
    std::streampos fileSize = inFile.tellg();
    std::vector<uint8_t> blob(gsl::narrow_cast<size_t>(fileSize));
    inFile.seekg(0, std::ios::beg);
    inFile.read(reinterpret_cast<char*>(blob.data()), fileSize);

    std::vector<std::unique_ptr<std::byte[]>> rawData;
    DmlSerializedGraphDesc serializedDesc = DeserializeDmlGraph(blob.data(), rawData);

    bindPoints = GetBindPoints(serializedDesc);
    m_constantDataTypes = ExtractConstantDataTypes(serializedDesc);
    m_bindings = GenerateInitialBindingsFromGraph(serializedDesc);

    // Create the D3D12 resources for constant nodes
    for (const auto& node : serializedDesc.Nodes)
    {
        if (std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(node.Desc))
        {
            CreateResourceFromConstantNode(node);
        }
    }
    //

    // Convert to Public Graph Description
    StackAllocator<1024> allocator;
    DML_GRAPH_DESC dmlGraphDesc = {};
    std::vector<Microsoft::WRL::ComPtr<IDMLOperator>> dmlOperators;
    std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes;
    std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges;
    std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges;
    std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges;
    std::vector<std::vector<std::uint8_t>> constDataVectors;

    // Convert the graph description
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
        IID_PPV_ARGS(&m_graphCompiled)));
}

struct BindingData
{
    std::vector<DML_BUFFER_BINDING> bufferBindings;
    std::vector<DML_BINDING_DESC> bindingDescs;
};


void FillBindingData(
    const std::vector<DmlSerializedGraphDispatchable::BindPoint>& bindPoints,
    const Dispatchable::Bindings* initializeBindings,
    const Dispatchable::Bindings* executeBindings,
    BindingData& bindingData,
    bool bindingForInitialization = false)
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
            if (bindPoints[i].required && !bindingForInitialization)
            {
                if (!initializeBindings || initializeBindings->find(bindPointName) == initializeBindings->end())
                {
                    throw std::invalid_argument(fmt::format("Nothing bound for required tensor '{}'.", bindPointName));
                }
            }

            for (size_t j = 0; j < bindPoints[i].resourceCount; j++)
            {
                bindingData.bufferBindings[bufferIndex].Buffer = nullptr;
                bindingData.bufferBindings[bufferIndex].Offset = 0;
                bindingData.bufferBindings[bufferIndex].SizeInBytes = 0;
                bindingData.bindingDescs[bufferIndex].Type = DML_BINDING_TYPE_NONE;
                bindingData.bindingDescs[bufferIndex].Desc = nullptr;
                bufferIndex++;
            }
        }
        else
        {
            auto& sources = bindingIterator->second;

            if (bindPoints[i].resourceCount != sources.size())
            {
                throw std::invalid_argument(fmt::format(
                    "Bind point '{}' requires {} resources, but {} were bound.",
                    bindPointName,
                    bindPoints[i].resourceCount,
                    sources.size()));
            }

            for (auto& source : sources)
            {
                assert(source.resource != nullptr);
                assert(source.resourceDesc != nullptr);

                if (!std::holds_alternative<Model::BufferDesc>(source.resourceDesc->value))
                {
                    throw std::invalid_argument("DML operators only support buffer bindings");
                }

                auto& bufferDesc = std::get<Model::BufferDesc>(source.resourceDesc->value);

                bindingData.bufferBindings[bufferIndex].Buffer = source.resource;
                bindingData.bufferBindings[bufferIndex].Offset = source.elementOffset * source.elementSizeInBytes;
                bindingData.bufferBindings[bufferIndex].SizeInBytes = bufferDesc.sizeInBytes - bindingData.bufferBindings[bufferIndex].Offset;
                bindingData.bindingDescs[bufferIndex].Type = DML_BINDING_TYPE_BUFFER;
                bindingData.bindingDescs[bufferIndex].Desc = &bindingData.bufferBindings[bufferIndex];
                bufferIndex++;
            }
        }
    }
}

void DmlSerializedGraphDispatchable::Initialize()
{

    BuildGraph();
    m_graphCompiled->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(m_name).data());

    ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator* ops[] = { m_graphCompiled.Get() };
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
    if (!m_bindings.empty())
    {
        // Initializers can initialize multiple inputs simultaneously, so each compiled op's inputs must
        // be bound using a separate buffer array binding.
        BindingData inputBindingData = {};
        FillBindingData(bindPoints.inputs, &m_bindings, nullptr, inputBindingData, true);

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
    auto persistentBufferSize = m_graphCompiled->GetBindingProperties().PersistentResourceSize;
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

void DmlSerializedGraphDispatchable::Bind(const Bindings& bindings, uint32_t iteration)
{
    auto bindingProps = m_graphCompiled->GetBindingProperties();

    BindingData inputBindingData = {};
    FillBindingData(bindPoints.inputs, &bindings, &bindings, inputBindingData);

    BindingData outputBindingData = {};
    FillBindingData(bindPoints.outputs, &bindings, &bindings, outputBindingData);

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
    bindingTableDesc.Dispatchable = m_graphCompiled.Get();
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

void DmlSerializedGraphDispatchable::Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings)
{
    m_device->RecordDispatch(m_graphCompiled.Get(), m_bindingTable.Get());
    m_device->ExecuteCommandListAndWait();
}
