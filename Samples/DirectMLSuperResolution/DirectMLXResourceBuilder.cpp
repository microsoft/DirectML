#include "pch.h"

#include "DirectMLSuperResolution.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"
#include "Float16Compressor.h"

using Microsoft::WRL::ComPtr;

using namespace DirectX;

#pragma warning(disable : 4238)

void Sample::CreateDirectMLResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Shader for converting texture to tensor
    {
        auto computeShaderBlob = DX::ReadData(L"ImageToTensor.cso");

        // Define root table layout
        CD3DX12_DESCRIPTOR_RANGE descRange[2];
        descRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0
        descRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0); // u0

        CD3DX12_ROOT_PARAMETER rootParameters[3];
        rootParameters[e_crpIdxCB].InitAsConstants(3, 0);
        rootParameters[e_crpIdxSRV].InitAsDescriptorTable(1, &descRange[0], D3D12_SHADER_VISIBILITY_ALL);
        rootParameters[e_crpIdxUAV].InitAsDescriptorTable(1, &descRange[1], D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC rootSignature(_countof(rootParameters), rootParameters);

        ComPtr<ID3DBlob> serializedSignature;
        DX::ThrowIfFailed(
            D3D12SerializeRootSignature(&rootSignature, D3D_ROOT_SIGNATURE_VERSION_1, serializedSignature.GetAddressOf(), nullptr));

        // Create the root signature
        DX::ThrowIfFailed(
            device->CreateRootSignature(
                0,
                serializedSignature->GetBufferPointer(),
                serializedSignature->GetBufferSize(),
                IID_PPV_ARGS(m_computeRootSignature.ReleaseAndGetAddressOf())));

        m_computeRootSignature->SetName(L"Compute RS");

        // Create compute pipeline state
        D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
        descComputePSO.pRootSignature = m_computeRootSignature.Get();
        descComputePSO.CS.pShaderBytecode = computeShaderBlob.data();
        descComputePSO.CS.BytecodeLength = computeShaderBlob.size();

        DX::ThrowIfFailed(
            device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(m_computePSO.ReleaseAndGetAddressOf())));
        m_computePSO->SetName(L"Compute PSO");
    }

    // Shader for rendering DML result tensor to texture
    // This can also be done with a compute shader, depending on the app's needs.
    {
        auto vsShaderBlob = DX::ReadData(L"TensorToImageVS.cso");
        auto psShaderBlob = DX::ReadData(L"TensorToImagePS.cso");

        static const D3D12_INPUT_ELEMENT_DESC s_inputElementDesc[1] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  0 },
        };

        // Define root table layout
        CD3DX12_DESCRIPTOR_RANGE descRange[1];
        descRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE); // t0

        CD3DX12_ROOT_PARAMETER rootParameters[2];
        rootParameters[e_rrpIdxCB].InitAsConstants(3, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[e_rrpIdxSRV].InitAsDescriptorTable(1, &descRange[0], D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_SIGNATURE_DESC rootSignature(_countof(rootParameters), rootParameters,
            0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

        ComPtr<ID3DBlob> serializedSignature;
        DX::ThrowIfFailed(
            D3D12SerializeRootSignature(&rootSignature, D3D_ROOT_SIGNATURE_VERSION_1, serializedSignature.GetAddressOf(), nullptr));

        // Create the root signature
        DX::ThrowIfFailed(
            device->CreateRootSignature(
                0,
                serializedSignature->GetBufferPointer(),
                serializedSignature->GetBufferSize(),
                IID_PPV_ARGS(m_tensorRenderRootSignature.ReleaseAndGetAddressOf())));

        m_tensorRenderRootSignature->SetName(L"Tensor Render RS");

        // Create pipeline state
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { s_inputElementDesc, _countof(s_inputElementDesc) };
        psoDesc.pRootSignature = m_tensorRenderRootSignature.Get();
        psoDesc.VS = { vsShaderBlob.data(), vsShaderBlob.size() };
        psoDesc.PS = { psShaderBlob.data(), psShaderBlob.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = m_deviceResources->GetDepthBufferFormat();
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;

        DX::ThrowIfFailed(
            device->CreateGraphicsPipelineState(&psoDesc,
                IID_PPV_ARGS(m_tensorRenderPipelineState.ReleaseAndGetAddressOf())));

        m_tensorRenderPipelineState->SetName(L"Tensor Render PSO");

        // Resource to hold the rendered texture
        D3D12_RESOURCE_DESC txtDesc = {};
        txtDesc.MipLevels = txtDesc.DepthOrArraySize = 1;
        txtDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        txtDesc.SampleDesc.Count = 1;
        txtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        txtDesc.Width = m_origTextureWidth * 2;
        txtDesc.Height = m_origTextureHeight * 2;
        txtDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &txtDesc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                &CD3DX12_CLEAR_VALUE(DXGI_FORMAT_B8G8R8A8_UNORM, DirectX::Colors::Black),
                IID_PPV_ARGS(m_finalResultTexture.ReleaseAndGetAddressOf())));

        // Create an RTV for rendering to the texture, and an SRV for rendering it back to the screen
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
        rtvDesc.Format = txtDesc.Format;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(m_finalResultTexture.Get(), &rtvDesc, m_RTVDescriptorHeap->GetCpuHandle(e_descFinalResultTextureRtv));

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = txtDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(m_finalResultTexture.Get(), &srvDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descFinalResultTextureSrv));
    }

    // DirectML device
    {
#if _DEBUG
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice)));
#else
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
#endif

#if FORCE_NCHW
        m_tensorLayout = TensorLayout::Default;
#else
        // Determine the best tensor layout based on the GPU vendor.
        // This is a fairly coarse-grained method, but recent Nvidia GPUs tend to use NHWC
        // layouts, while others use the default NCHW. 
        ComPtr<IDXGIAdapter1> adapter;
        DX::ThrowIfFailed(m_deviceResources->GetDXGIFactory()->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)));

        DXGI_ADAPTER_DESC adapterDesc;
        adapter->GetDesc(&adapterDesc);

        if (adapterDesc.VendorId == 0x10DE) // Nvidia
        {
            // This is faster on recent Nvidia hardware, but may be a problem on older hardware.
            // If necessary, set FORCE_NCHW to override this.
            m_tensorLayout = TensorLayout::NHWC;
        }
        else
        {
            m_tensorLayout = TensorLayout::Default;
        }
#endif

        DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
        DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
        DX::ThrowIfFailed(m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query), &fp16Query, sizeof(fp16Supported), &fp16Supported));

        if (!fp16Supported.IsSupported)
        {
            throw std::exception("FP16 data type support is required for this sample.");
        }

        DX::ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder)));
    }

    uint64_t modelInputBufferSize = 0;
    uint64_t modelOutputBufferSize = 0;

    {
        using Dimensions = dml::TensorDesc::Dimensions;

        // Create the residual with three convolutions, an upsample, and four more convolutions
        WeightMapType weights;
        if (!LoadWeights("Assets\\weights.bin", weights))
        {
            throw std::exception("loadWeights");
        }


        // Upload weights to the GPU
        DirectX::ResourceUploadBatch weightUploadBatch(device);
        weightUploadBatch.Begin();

        CreateWeightTensors(weights, "conv1/weights", "conv1/BatchNorm/scale", "conv1/BatchNorm/shift",
            std::array<uint32_t, 4>{ 32,  3, 5, 5 }, weightUploadBatch, &m_modelConvFilterWeights[0], &m_modelConvBiasWeights[0]);
        CreateWeightTensors(weights, "conv2/weights", "conv2/BatchNorm/scale", "conv2/BatchNorm/shift",
            std::array<uint32_t, 4>{ 64, 32, 3, 3 }, weightUploadBatch, &m_modelConvFilterWeights[1], &m_modelConvBiasWeights[1]);
        CreateWeightTensors(weights, "conv3/weights", "conv3/BatchNorm/scale", "conv3/BatchNorm/shift",
            std::array<uint32_t, 4>{ 64, 64, 3, 3 }, weightUploadBatch, &m_modelConvFilterWeights[2], &m_modelConvBiasWeights[2]);
        CreateWeightTensors(weights, "conv_up1/conv/weights", "conv_up1/conv/BatchNorm/scale", "conv_up1/conv/BatchNorm/shift",
            std::array<uint32_t, 4>{ 32, 64, 5, 5 }, weightUploadBatch, &m_modelConvFilterWeights[3], &m_modelConvBiasWeights[3]);
        CreateWeightTensors(weights, "conv4/weights", "conv4/BatchNorm/scale", "conv4/BatchNorm/shift",
            std::array<uint32_t, 4>{ 32, 32, 3, 3 }, weightUploadBatch, &m_modelConvFilterWeights[4], &m_modelConvBiasWeights[4]);
        CreateWeightTensors(weights, "conv5/weights", "conv5/BatchNorm/scale", "conv5/BatchNorm/shift",
            std::array<uint32_t, 4>{ 32, 32, 3, 3 }, weightUploadBatch, &m_modelConvFilterWeights[5], &m_modelConvBiasWeights[5]);
        CreateWeightTensors(weights, "conv6/weights", nullptr, nullptr,
            std::array<uint32_t, 4>{ 3, 32, 3, 3 }, weightUploadBatch, &m_modelConvFilterWeights[6], nullptr);

        weightUploadBatch.End(m_deviceResources->GetCommandQueue());


        // Construct a DML graph of operators

        DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
#if DML_MANAGED_WEIGHTS
        flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
#endif
        
        // Select the correct tensor policy depending on our desired layout
        dml::TensorPolicy policy =
            m_tensorLayout == TensorLayout::Default
            ? dml::TensorPolicy::Default()
            : dml::TensorPolicy::InterleavedChannel();

        dml::Graph graph(m_dmlDevice.Get(), policy);

        // Set up input tensors
        Dimensions modelInputSizes = { 1, 3, m_origTextureHeight, m_origTextureWidth };
        auto modelInput = dml::InputTensor(graph, 0, dml::TensorDesc(dataType, modelInputSizes, policy));

        // conv1
        auto conv1Filter = dml::InputTensor(graph, 1, dml::TensorDesc(dataType, flags, { 32,  3, 5, 5 }, policy));
        auto conv1Bias = dml::InputTensor(graph, 2, dml::TensorDesc(dataType, flags, { 1, 32, 1, 1 }, policy));
        auto conv1 = dml::ConvolutionBuilder(modelInput, conv1Filter, conv1Bias)
            .StartPadding(std::array<uint32_t, 2>{ 2u, 2u })
            .EndPadding(std::array<uint32_t, 2>{ 2u, 2u })
            .FusedActivation(dml::FusedActivation::Relu())
            .Build();

        // conv2
        auto conv2Filter = dml::InputTensor(graph, 3, dml::TensorDesc(dataType, flags, { 64, 32, 3, 3 }, policy));
        auto conv2Bias = dml::InputTensor(graph, 4, dml::TensorDesc(dataType, flags, { 1, 64, 1, 1 }, policy));
        auto conv2 = dml::ConvolutionBuilder(conv1, conv2Filter, conv2Bias)
            .StartPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .EndPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .FusedActivation(dml::FusedActivation::Relu())
            .Build();

        // conv3
        auto conv3Filter = dml::InputTensor(graph, 5, dml::TensorDesc(dataType, flags, { 64, 64, 3, 3 }, policy));
        auto conv3Bias = dml::InputTensor(graph, 6, dml::TensorDesc(dataType, flags, { 1, 64, 1, 1 }, policy));
        auto conv3 = dml::ConvolutionBuilder(conv2, conv3Filter, conv3Bias)
            .StartPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .EndPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .FusedActivation(dml::FusedActivation::Relu())
            .Build();

        // up1 (2x nearest-neighbor upsample)
        auto up1 = dml::Upsample2D(conv3, DML_SIZE_2D{ 2, 2 }, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR);

        // conv_up1
        auto convUp1Filter = dml::InputTensor(graph, 7, dml::TensorDesc(dataType, flags, { 32, 64, 5, 5 }, policy));
        auto convUp1Bias = dml::InputTensor(graph, 8, dml::TensorDesc(dataType, flags, { 1, 32, 1, 1 }, policy));
        auto convUp1 = dml::ConvolutionBuilder(up1, convUp1Filter, convUp1Bias)
            .StartPadding(std::array<uint32_t, 2>{ 2u, 2u })
            .EndPadding(std::array<uint32_t, 2>{ 2u, 2u })
            .FusedActivation(dml::FusedActivation::Relu())
            .Build();

        // conv4
        auto conv4Filter = dml::InputTensor(graph, 9, dml::TensorDesc(dataType, flags, { 32, 32, 3, 3 }, policy));
        auto conv4Bias = dml::InputTensor(graph, 10, dml::TensorDesc(dataType, flags, { 1, 32, 1, 1 }, policy));
        auto conv4 = dml::ConvolutionBuilder(convUp1, conv4Filter, conv4Bias)
            .StartPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .EndPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .FusedActivation(dml::FusedActivation::Relu())
            .Build();

        // conv5
        auto conv5Filter = dml::InputTensor(graph, 11, dml::TensorDesc(dataType, flags, { 32, 32, 3, 3 }, policy));
        auto conv5Bias = dml::InputTensor(graph, 12, dml::TensorDesc(dataType, flags, { 1, 32, 1, 1 }, policy));
        auto conv5 = dml::ConvolutionBuilder(conv4, conv5Filter, conv5Bias)
            .StartPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .EndPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .FusedActivation(dml::FusedActivation::Relu())
            .Build();

        // conv6 (no bias or activation)
        auto conv6Filter = dml::InputTensor(graph, 13, dml::TensorDesc(dataType, flags, { 3, 32, 3, 3 }, policy));
        auto conv6 = dml::ConvolutionBuilder(conv5, conv6Filter)
            .StartPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .EndPadding(std::array<uint32_t, 2>{ 1u, 1u })
            .Build();

        // Add the output of the convolutions to an upscaled version of the original image
        auto up2 = dml::Upsample2D(modelInput, DML_SIZE_2D{ 2, 2 }, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR);
        auto output = up2 + conv6;

        modelInputBufferSize = modelInput.GetOutputDesc().totalTensorSizeInBytes;
        modelOutputBufferSize = output.GetOutputDesc().totalTensorSizeInBytes;

        // Compile the graph
        DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        m_dmlGraph = graph.Compile(executionFlags, std::array<dml::Expression, 1>{ output });
    }

    // Buffers for DML inputs and outputs
    {
        // Resource for input tensor
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(modelInputBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelInput)
        ));

        // Describe and create a UAV for the original input tensor.
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = static_cast<UINT>(modelInputBufferSize / sizeof(uint16_t));
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        device->CreateUnorderedAccessView(m_modelInput.Get(), nullptr, &uavDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descModelInput));

        // Model result tensor is 2x larger in both dimensions
        resourceDesc.Width = modelOutputBufferSize;
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelOutput)
        ));

        // Describe and create a SRV for the final result tensor.
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = static_cast<UINT>(modelOutputBufferSize / sizeof(uint16_t));
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        device->CreateShaderResourceView(m_modelOutput.Get(), &srvDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descModelOutput));
    }

    // Wait until assets have been uploaded to the GPU.
    m_deviceResources->WaitForGpu();
}

void Sample::InitializeDirectMLResources()
{
    auto commandList = m_deviceResources->GetCommandList();
    commandList->Reset(m_deviceResources->GetCommandAllocator(), nullptr);

    DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(1, m_dmlGraph.GetAddressOf(), IID_PPV_ARGS(&m_dmlOpInitializer)));

    DML_BINDING_PROPERTIES initBindingProps = m_dmlOpInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProps = m_dmlGraph->GetBindingProperties();

    m_dmlDescriptorHeap = std::make_unique<DescriptorHeap>(
        m_deviceResources->GetD3DDevice(),
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        std::max(initBindingProps.RequiredDescriptorCount, executeBindingProps.RequiredDescriptorCount));

    // Operator initialization dispatches will use this heap right away
    ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
    commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

    // Create any persistent resources required for the operators.
    if (executeBindingProps.PersistentResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            executeBindingProps.PersistentResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelPersistentResource)));
    }

    // Temporary resource for execution
    if (executeBindingProps.TemporaryResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            executeBindingProps.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelTemporaryResource)));
    }

    // If the execute temporary resource isn't big enough for initialization, create a bigger buffer
    ComPtr<ID3D12Resource> initTemporaryResource;
    if (initBindingProps.TemporaryResourceSize > executeBindingProps.TemporaryResourceSize)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            initBindingProps.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&initTemporaryResource)));
    }
    else if (initBindingProps.TemporaryResourceSize > 0)
    {
        initTemporaryResource = m_modelTemporaryResource;
    }

    Microsoft::WRL::ComPtr<IDMLBindingTable> initBindingTable;
    assert(initBindingProps.PersistentResourceSize == 0);

    DML_BINDING_TABLE_DESC tableDesc =
    {
        m_dmlOpInitializer.Get(),
        m_dmlDescriptorHeap->GetCpuHandle(0),
        m_dmlDescriptorHeap->GetGpuHandle(0),
        initBindingProps.RequiredDescriptorCount
    };
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable)));

    DML_BUFFER_BINDING bufferBindings[] =
    {
        {}, // model input
        { m_modelConvFilterWeights[0].Get(), 0, m_modelConvFilterWeights[0]->GetDesc().Width }, { m_modelConvBiasWeights[0].Get(), 0, m_modelConvBiasWeights[0]->GetDesc().Width },
        { m_modelConvFilterWeights[1].Get(), 0, m_modelConvFilterWeights[1]->GetDesc().Width }, { m_modelConvBiasWeights[1].Get(), 0, m_modelConvBiasWeights[1]->GetDesc().Width },
        { m_modelConvFilterWeights[2].Get(), 0, m_modelConvFilterWeights[2]->GetDesc().Width }, { m_modelConvBiasWeights[2].Get(), 0, m_modelConvBiasWeights[2]->GetDesc().Width },
        { m_modelConvFilterWeights[3].Get(), 0, m_modelConvFilterWeights[3]->GetDesc().Width }, { m_modelConvBiasWeights[3].Get(), 0, m_modelConvBiasWeights[3]->GetDesc().Width },
        { m_modelConvFilterWeights[4].Get(), 0, m_modelConvFilterWeights[4]->GetDesc().Width }, { m_modelConvBiasWeights[4].Get(), 0, m_modelConvBiasWeights[4]->GetDesc().Width },
        { m_modelConvFilterWeights[5].Get(), 0, m_modelConvFilterWeights[5]->GetDesc().Width }, { m_modelConvBiasWeights[5].Get(), 0, m_modelConvBiasWeights[5]->GetDesc().Width },
        { m_modelConvFilterWeights[6].Get(), 0, m_modelConvFilterWeights[6]->GetDesc().Width }, // last layer has no bias
    };

    // Bind inputs for initialization, which is only necessary if we're using OWNED_BY_DML

#if DML_MANAGED_WEIGHTS
    DML_BUFFER_ARRAY_BINDING initInputBinding = { ARRAYSIZE(bufferBindings), bufferBindings };
    initBindingTable->BindInputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &initInputBinding });
#else
    initBindingTable->BindInputs(0, nullptr);
#endif

    if (initTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { initTemporaryResource.Get(), 0, initTemporaryResource->GetDesc().Width };
        initBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    // If the operator requires a persistent resource, it must be bound as output for the initializer.
    if (m_modelPersistentResource)
    {
        DML_BUFFER_BINDING binding = { m_modelPersistentResource.Get(), 0, m_modelPersistentResource->GetDesc().Width };
        initBindingTable->BindOutputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    // Record the initialization
    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializer.Get(), initBindingTable.Get());

    DX::ThrowIfFailed(commandList->Close());
    m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    // Wait until initialization has been finished on the GPU.
    m_deviceResources->WaitForGpu();

#if DML_MANAGED_WEIGHTS
    // These have been copied to DML-managed resources and are no longer needed.
    for (int i = 0; i < c_numConvLayers; i++)
    {
        m_modelConvFilterWeights[i].Reset();
        if (i < c_numConvLayers - 1)    // Last layer has no bias
        {
            m_modelConvBiasWeights[i].Reset();
        }
    }
#endif
    
    // 
    // Now that we're done with operator initialization, set up the binding table for execution
    // 

    tableDesc.Dispatchable = m_dmlGraph.Get();
    tableDesc.SizeInDescriptors = executeBindingProps.RequiredDescriptorCount;
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&m_dmlBindingTable)));

    if (m_modelPersistentResource)
    {
        DML_BUFFER_BINDING binding = { m_modelPersistentResource.Get(), 0, m_modelPersistentResource->GetDesc().Width };
        m_dmlBindingTable->BindPersistentResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    if (m_modelTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { m_modelTemporaryResource.Get(), 0, m_modelTemporaryResource->GetDesc().Width };
        m_dmlBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    // Bind model inputs and outputs
    bufferBindings[0] = DML_BUFFER_BINDING{ m_modelInput.Get() };
#if DML_MANAGED_WEIGHTS
    // Bind only the model input
    DML_BINDING_DESC inputBindings[] =
    {
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[0] }, // model input
        { DML_BINDING_TYPE_NONE, nullptr }, { DML_BINDING_TYPE_NONE, nullptr },
        { DML_BINDING_TYPE_NONE, nullptr }, { DML_BINDING_TYPE_NONE, nullptr },
        { DML_BINDING_TYPE_NONE, nullptr }, { DML_BINDING_TYPE_NONE, nullptr },
        { DML_BINDING_TYPE_NONE, nullptr }, { DML_BINDING_TYPE_NONE, nullptr },
        { DML_BINDING_TYPE_NONE, nullptr }, { DML_BINDING_TYPE_NONE, nullptr },
        { DML_BINDING_TYPE_NONE, nullptr }, { DML_BINDING_TYPE_NONE, nullptr },
        { DML_BINDING_TYPE_NONE, nullptr }, // last layer has no bias
    };
    m_dmlBindingTable->BindInputs(ARRAYSIZE(inputBindings), inputBindings);
#else
    // Bind everything
    DML_BINDING_DESC inputBindings[] =
    {
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[0] }, // model input
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[1] }, { DML_BINDING_TYPE_BUFFER, &bufferBindings[2] },
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[3] }, { DML_BINDING_TYPE_BUFFER, &bufferBindings[4] },
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[5] }, { DML_BINDING_TYPE_BUFFER, &bufferBindings[6] },
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[7] }, { DML_BINDING_TYPE_BUFFER, &bufferBindings[8] },
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[9] }, { DML_BINDING_TYPE_BUFFER, &bufferBindings[10] },
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[11] }, { DML_BINDING_TYPE_BUFFER, &bufferBindings[12] },
        { DML_BINDING_TYPE_BUFFER, &bufferBindings[13] }, // last layer has no bias
    };
    m_dmlBindingTable->BindInputs(ARRAYSIZE(inputBindings), inputBindings);
#endif

    DML_BUFFER_BINDING outputBinding = { m_modelOutput.Get(), 0, m_modelOutput->GetDesc().Width };
    m_dmlBindingTable->BindOutputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &outputBinding });
}
