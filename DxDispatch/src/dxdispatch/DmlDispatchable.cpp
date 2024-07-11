#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "DmlDispatchable.h"

using Microsoft::WRL::ComPtr;

DmlDispatchable::DmlDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::DmlDispatchableDesc& desc,
    const Dispatchable::Bindings& initBindings,
    IDxDispatchLogger* logger
    ) : m_name(name), m_device(device), m_desc(desc), m_initBindings(std::move(initBindings)), m_logger(logger)
{
    THROW_IF_FAILED(device->DML()->CreateOperator(desc.desc, IID_PPV_ARGS(&m_operator)));
}

struct BindingData
{
    std::vector<DML_BUFFER_BINDING> bufferBindings;
    std::vector<DML_BINDING_DESC> bindingDescs;
};

void FillBindingData(
    const std::vector<Model::DmlDispatchableDesc::BindPoint>& bindPoints,
    const Dispatchable::Bindings* initializeBindings,
    const Dispatchable::Bindings* executeBindings,
    BindingData& bindingData,
    bool bindingForInitialization, 
    Model::DmlDispatchableDesc::DmlCompileType compileType)
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
                if (compileType == Model::DmlDispatchableDesc::DmlCompileType::DmlCompileGraph && !bindPoints[i].requiredBinding)
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

void DmlDispatchable::Initialize()
{
    if(m_desc.compileType == Model::DmlDispatchableDesc::DmlCompileType::DmlCompileOp)
    {
        m_logger->LogInfo("Compile Op");
        THROW_IF_FAILED(m_device->DML()->CompileOperator(
            m_operator.Get(), 
            m_desc.executionFlags, 
            IID_PPV_ARGS(m_operatorCompiled.ReleaseAndGetAddressOf())));
        m_operatorCompiled->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(m_name).data());
    }
    else
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
            dmlGraphNodeDesc.Desc =  &nodeDesc;
        }

        dmlInputGraphEdges.resize(m_desc.bindPoints.inputs.size());
        for( size_t i = 0; i < m_desc.bindPoints.inputs.size(); i++)
        {
            if (m_desc.bindPoints.inputs[i].requiredBinding)
            {
                DML_INPUT_GRAPH_EDGE_DESC desc = {};
                desc.GraphInputIndex = gsl::narrow_cast<UINT>(i);
                desc.ToNodeIndex = 0;
                desc.ToNodeInputIndex = gsl::narrow_cast<UINT>(i);
                desc.Name = m_desc.bindPoints.inputs[i].name.c_str();
                dmlInputGraphEdges[i] = desc;
                dmlInputEdges.push_back({ DML_GRAPH_EDGE_TYPE_INPUT, &dmlInputGraphEdges[i] });
            }
        }

        dmlOutputGraphEdges.resize(m_desc.bindPoints.outputs.size());
        for( size_t i = 0; i < m_desc.bindPoints.outputs.size(); i++)
        {
            if (m_desc.bindPoints.outputs[i].requiredBinding)
            {
                DML_OUTPUT_GRAPH_EDGE_DESC desc = {};
                desc.GraphOutputIndex = gsl::narrow_cast<UINT>(i);
                desc.FromNodeIndex = 0;
                desc.FromNodeOutputIndex = gsl::narrow_cast<UINT>(i);
                desc.Name = m_desc.bindPoints.outputs[i].name.c_str();
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

        THROW_IF_FAILED(m_device->DML()->CompileGraph(&dmlGraphDesc, m_desc.executionFlags, IID_PPV_ARGS(&m_operatorCompiled)));
        m_operatorCompiled->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(fmt::format("Graph_{}", m_name)).data());
    }

    ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator* ops[] = { m_operatorCompiled.Get() };
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
    if (!m_initBindings.empty())
    {
        // Initializers can initialize multiple inputs simultaneously, so each compiled op's inputs must
        // be bound using a separate buffer array binding.
        BindingData inputBindingData = {};
        FillBindingData(m_desc.bindPoints.inputs, &m_initBindings, nullptr, inputBindingData, true, m_desc.compileType);

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
    auto persistentBufferSize = m_operatorCompiled->GetBindingProperties().PersistentResourceSize;
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
    auto bindingProps = m_operatorCompiled->GetBindingProperties();

    BindingData inputBindingData = {};
    FillBindingData(m_desc.bindPoints.inputs, &m_initBindings, &bindings, inputBindingData, false, m_desc.compileType);

    BindingData outputBindingData = {};
    FillBindingData(m_desc.bindPoints.outputs, &m_initBindings, &bindings, outputBindingData, false, m_desc.compileType);

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
    bindingTableDesc.Dispatchable = m_operatorCompiled.Get();
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

void DmlDispatchable::Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBinings)
{
    m_device->RecordDispatch(m_operatorCompiled.Get(), m_bindingTable.Get());
    m_device->ExecuteCommandListAndWait();
}