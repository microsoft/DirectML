#include "pch.h"
#include "Adapter.h"

#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "FbDispatchable.h"
#include "DirectMLHelpers/DmlGraphDeserialization.h"
#include "DirectMLHelpers/DmlGraphHelper.h"
//#include "flatbuffers/flatbuffers.h"
//#include "DmlSerializedGraphDesc.h"
//#include "Test/Common/Common.h"//basic api converter
//#include "Test/Common/Common.h"
//#include "Product/InternalInterfaces.h"
//#include "tools\DirectMLPlanParser\inc\ReadOperator.h"
//#include "SharedToolingLib/External/DmlIR/Operator/PrivateOperators.h"

//using Microsoft::WRL::ComPtr;


void BuildFolder(std::filesystem::path folderPath, std::vector<std::string> &graphFiles)
{
    std::filesystem::path parentPath = std::filesystem::current_path(); // Get the current working directory
    assert(std::filesystem::is_directory(folderPath)); // Assert that the provided path is a directory
    for (auto const& entry : std::filesystem::recursive_directory_iterator{folderPath}) // Iterate over all files in the directory recursively
    {
        auto entryPath = entry.path();
        {
            auto pathStr = entryPath.string(); 
            {
                //LogInfo("Adding: " + pathStr); 
                graphFiles.emplace_back(pathStr); 
            }
        }
    }
}

void BuildGraph(std::string graphFile, std::shared_ptr<Device> device, Microsoft::WRL::ComPtr<IDMLCompiledOperator> &compiledOp) 
{
    
    std::ifstream inFile(graphFile, std::ios::binary | std::ios::ate);
    if (!inFile)
    {
        //LogError("Failed to open file: " + graphFile);
        return;
    }
    std::streampos fileSize = inFile.tellg();
    std::vector<uint8_t> blob(gsl::narrow_cast<size_t>(fileSize));
    inFile.seekg(0, std::ios::beg);
    inFile.read(reinterpret_cast<char*>(blob.data()), fileSize);
    
    std::vector<std::unique_ptr<std::byte[]>> rawData;
    DmlSerializedGraphDesc serializedDesc = DeserializeDmlGraph(blob.data(), rawData);
    //checkED if serilized desc compares with gemm.onnx in number of input,output edges



    //// Convert to Public Graph Description
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
        device->DML(),
        allocator,
        nullptr, // check serializedGraphInputIndexToSubgraphInputIndex
        nullptr, // check serializedGraphLargeConstantNameToSubgraphInputIndex
        dmlGraphDesc,
        dmlOperators,
        dmlGraphNodes,
        dmlInputEdges,
        dmlOutputEdges,
        dmlIntermediateEdges);
    
    // std::cout<<dmlGraphDesc.InputCount<<std::endl;
    // std::cout<<dmlGraphDesc.OutputCount<<std::endl;
    // std::cout<<dmlGraphDesc.NodeCount<<std::endl;

    //Compile the graph
    THROW_IF_FAILED(device->DML()->CompileGraph(
       &dmlGraphDesc,
       DML_EXECUTION_FLAG_NONE,
       IID_PPV_ARGS(&compiledOp)
    ));
}
   

FbDispatchable::FbDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::FbDispatchableDesc& desc,//TODO: Define Model::FbDispatchableDesc 
    const Model::Bindings& initBindings) :
          m_name(name), m_device(device), m_desc(desc), m_initBindings(initBindings)
{}
    
void FbDispatchable::Initialize()
{
    std::vector<std::string> graphFiles;
   // Compile the graph
   BuildFolder(m_desc.sourcePath, graphFiles);
   BuildGraph(graphFiles.front(), m_device, m_operatorCompiled);

//    // Set the name of the compiled operator
//    m_operatorCompiled->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(m_name).data());
//
//    
//    // Create an initializer for the compiled  operator
//    ComPtr<IDMLOperatorInitializer> initializer;
//    IDMLCompiledOperator* ops[] = { m_operatorCompiled.Get() };
//    THROW_IF_FAILED(m_device->DML()->CreateOperatorInitializer(
//        _countof(ops),
//        ops,
//        IID_PPV_ARGS(&initializer)));
//
//    // Get the number of descriptors for the binding table
//    auto min = initializer->GetBindingProperties().RequiredDescriptorCount;
//
//
//    // Create a descriptor heap with at least one descriptor. Even if the op doesn't
//    // require any descriptors the binding table expects valid descriptor handles.
//    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
//    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
//    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
//    descriptorHeapDesc.NumDescriptors = std::max(1u, initializer->GetBindingProperties().RequiredDescriptorCount);
//    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
//    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(descriptorHeap.ReleaseAndGetAddressOf())));
//
//    // Set the descriptor heap on the command list
//    ID3D12DescriptorHeap* descriptorHeaps[] = { descriptorHeap.Get() };
//    m_device->GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
//
//
//    // Create the binding table description
//    DML_BINDING_TABLE_DESC bindingTableDesc = {};
//    bindingTableDesc.Dispatchable = initializer.Get();
//    bindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
//    bindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
//    bindingTableDesc.SizeInDescriptors = initializer->GetBindingProperties().RequiredDescriptorCount;
//
//    // Create the binding table
//    ComPtr<IDMLBindingTable> bindingTable;
//    THROW_IF_FAILED(m_device->DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));
//
//    // Inputs flagged OWNED_BY_DML must be bound during initialization (and only initialization).
//    if (!initBindings.bindingDescs.empty())
//    {   
////////////////////////////////////////////////////////////////////////////////////////////////////
//        // TODO: 
//        // Initializers can initialize multiple inputs simultaneously, so each compiled op's inputs must
//        // be bound using a separate buffer array binding.
////////////////////////////////////////////////////////////////////////////////////////////////////
//    }
//
//    // A temporary resource may be required to initialize the operators.
//    auto tempBufferSize = initializer->GetBindingProperties().TemporaryResourceSize;
//    if (tempBufferSize > 0)
//    {
//        ComPtr<ID3D12Resource> tempBuffer = m_device->CreatePreferredDeviceMemoryBuffer(tempBufferSize);
//        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
//        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
//        bindingTable->BindTemporaryResource(&bindingDesc);
//        m_device->KeepAliveUntilNextCommandListDispatch(std::move(tempBuffer));
//    }
//    
//    // Each compiled op's persistent resource is bound as an output of the initializer.
//    auto persistentBufferSize = m_operatorCompiled->GetBindingProperties().PersistentResourceSize;
//    if (persistentBufferSize > 0)
//    {
//        m_persistentBuffer = m_device->CreatePreferredDeviceMemoryBuffer(persistentBufferSize);
//        DML_BUFFER_BINDING bufferBinding = { m_persistentBuffer.Get(), 0, persistentBufferSize };
//        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
//        bindingTable->BindOutputs(1, &bindingDesc);
//    }
//    // Keeps descriptor heap alive, records an initialization operation, and executes the command list, waiting for completion
//    m_device->KeepAliveUntilNextCommandListDispatch(std::move(descriptorHeap));
//    m_device->RecordInitialize(initializer.Get(), bindingTable.Get());
//    m_device->ExecuteCommandListAndWait();
//
////return?
//
}

void FbDispatchable::Bind(const Bindings& bindings, uint32_t iteration)
{
//
//    auto bindingProps = m_operatorCompiled->GetBindingProperties();
////////////////////////////////////////////////////////////////////
//    //TODO: Prepare and manage bindings  
//
////////////////////////////////////////////////////////////////////
//    
//    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
//    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
//    descriptorHeapDesc.NumDescriptors = bindingProps.RequiredDescriptorCount;
//    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
//    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(
//        &descriptorHeapDesc, 
//        IID_GRAPHICS_PPV_ARGS(m_descriptorHeap.ReleaseAndGetAddressOf())));
//
//
//    ID3D12DescriptorHeap* descriptorHeaps[] = { m_descriptorHeap.Get() };
//    m_device->GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
//
//    DML_BINDING_TABLE_DESC bindingTableDesc = {};
//    bindingTableDesc.Dispatchable = m_operatorCompiled.Get();
//    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
//    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
//    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;
//
//    THROW_IF_FAILED(m_device->DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(m_bindingTable.ReleaseAndGetAddressOf())));
//
//
//    if (inputBindingData.bindingDescs.size() > std::numeric_limits<uint32_t>::max())
//    {
//        throw std::invalid_argument(fmt::format("BindInputs count  '{}' is too large.", inputBindingData.bindingDescs.size()));
//    }
//
//    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingData.bindingDescs.size()), inputBindingData.bindingDescs.data());
//   
//   
//    ComPtr<ID3D12Resource> tempBuffer;
//    auto tempBufferSize = bindingProps.TemporaryResourceSize;
//    if (tempBufferSize > 0)
//    {
//        tempBuffer = m_device->CreateDefaultBuffer(tempBufferSize);
//
//
//        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
//        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
//        m_bindingTable->BindTemporaryResource(&bindingDesc);
//        m_device->KeepAliveUntilNextCommandListDispatch(std::move(tempBuffer)); 
//    }
//
//    auto persistentBufferSize = bindingProps.PersistentResourceSize;
//    if (persistentBufferSize > 0)
//    {
//        DML_BUFFER_BINDING bufferBinding = { m_persistentBuffer.Get(), 0, persistentBufferSize };
//        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
//        m_bindingTable->BindPersistentResource(&bindingDesc);
//    }
//
//    if (outputBindingData.bindingDescs.size() > std::numeric_limits<uint32_t>::max())
//    {
//        throw std::invalid_argument(fmt::format("BindOutputs count  '{}' is too large.", outputBindingData.bindingDescs.size()));
//    }
//    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingData.bindingDescs.size()), outputBindingData.bindingDescs.data());
//
//    // DML may remove the device if invalid bindings are specified.
//    THROW_IF_FAILED(m_device->DML()->GetDeviceRemovedReason());
//
//    //return?
}

void FbDispatchable::Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings)
{
//    m_device->RecordDispatch(m_operatorCompiled.Get(), m_bindingTable.Get());
//    m_device->ExecuteCommandListAndWait();
}
