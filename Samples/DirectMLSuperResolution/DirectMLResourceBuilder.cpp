#include "pch.h"

#include "DirectMLSuperResolution.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"
#include "Float16Compressor.h"

#include "DirectMLX.h"

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
    uint64_t intermediateBufferMaxSize[] = { 0, 0 };

    // DirectML operator resources--implementation of the super-resolution model
    {
        // Create an upscaled (nearest neighbor) version of the image first
        uint32_t modelInputSizes[] = { 1, 3, m_origTextureHeight, m_origTextureWidth };
        uint32_t upscaledInputSizes[4];
        CreateUpsampleLayer(modelInputSizes, &modelInputBufferSize, &modelOutputBufferSize, upscaledInputSizes, &m_dmlUpsampleOps[0]);

        // Create the residual with three convolutions, an upsample, and four more convolutions
        WeightMapType weights;
        if (!LoadWeights("Assets\\weights.bin", weights))
        {
            throw std::exception("loadWeights");
        }

        DirectX::ResourceUploadBatch weightUploadBatch(device);
        weightUploadBatch.Begin();

        uint32_t const filterSizes1[] = { 32, 3, 5, 5 };
        uint32_t intermediateInputSizes[2][4];
        CreateConvolutionLayer(modelInputSizes, filterSizes1, true, &modelInputBufferSize,
            &intermediateBufferMaxSize[0], intermediateInputSizes[0], &m_dmlConvOps[0]);
        CreateWeightTensors(weights, "conv1/weights", "conv1/BatchNorm/scale", "conv1/BatchNorm/shift",
            filterSizes1, weightUploadBatch, &m_modelConvFilterWeights[0], &m_modelConvBiasWeights[0]);

        // Which intermediate resource to use as input for the current operation. The other will be
        // used as output. Then the next op will swap the order.
        int inputIndex = 0;

        uint32_t const filterSizes2[] = { 64, 32, 3, 3 };	// output filters
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes2, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[1]);
        CreateWeightTensors(weights, "conv2/weights", "conv2/BatchNorm/scale", "conv2/BatchNorm/shift",
            filterSizes2, weightUploadBatch, &m_modelConvFilterWeights[1], &m_modelConvBiasWeights[1]);
        inputIndex = 1 - inputIndex;

        uint32_t const filterSizes3[] = { 64, 64, 3, 3 };
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes3, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[2]);
        CreateWeightTensors(weights, "conv3/weights", "conv3/BatchNorm/scale", "conv3/BatchNorm/shift",
            filterSizes3, weightUploadBatch, &m_modelConvFilterWeights[2], &m_modelConvBiasWeights[2]);
        inputIndex = 1 - inputIndex;

        CreateUpsampleLayer(intermediateInputSizes[inputIndex], &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlUpsampleOps[1]);
        inputIndex = 1 - inputIndex;

        uint32_t const filterSizes4[] = { 32, 64, 5, 5 };
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes4, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[3]);
        CreateWeightTensors(weights, "conv_up1/conv/weights", "conv_up1/conv/BatchNorm/scale", "conv_up1/conv/BatchNorm/shift",
            filterSizes4, weightUploadBatch, &m_modelConvFilterWeights[3], &m_modelConvBiasWeights[3]);
        inputIndex = 1 - inputIndex;

        uint32_t const filterSizes5[] = { 32, 32, 3, 3 };
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes5, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[4]);
        CreateWeightTensors(weights, "conv4/weights", "conv4/BatchNorm/scale", "conv4/BatchNorm/shift",
            filterSizes5, weightUploadBatch, &m_modelConvFilterWeights[4], &m_modelConvBiasWeights[4]);
        inputIndex = 1 - inputIndex;

        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes5, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[5]);
        CreateWeightTensors(weights, "conv5/weights", "conv5/BatchNorm/scale", "conv5/BatchNorm/shift",
            filterSizes5, weightUploadBatch, &m_modelConvFilterWeights[5], &m_modelConvBiasWeights[5]);
        inputIndex = 1 - inputIndex;

        uint32_t const filterSizes6[] = { 3, 32, 3, 3 };
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes6, false, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[6]);
        CreateWeightTensors(weights, "conv6/weights", nullptr, nullptr, filterSizes6, weightUploadBatch,
            &m_modelConvFilterWeights[6], nullptr);
        inputIndex = 1 - inputIndex;

        // Finally add the residual to the original upsampled image
        assert(memcmp(upscaledInputSizes, intermediateInputSizes[inputIndex], 4 * sizeof(uint16_t)) == 0);

        CreateAdditionLayer(upscaledInputSizes, &m_dmlAddResidualOp);

        weightUploadBatch.End(m_deviceResources->GetCommandQueue());
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

        // Create two resources for intermediate layer results. Each layer will ping-pong between these. They're each large
        // enough to hold the largest intermediate result required.
        for (int i = 0; i < 2; i++)
        {
            resourceDesc.Width = intermediateBufferMaxSize[i];
            DX::ThrowIfFailed(device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&m_modelIntermediateResult[i])
            ));
        }
    }

    // Wait until assets have been uploaded to the GPU.
    m_deviceResources->WaitForGpu();
}

UINT GetDescriptorCount(size_t numOps, IDMLCompiledOperator** ops, IDMLOperatorInitializer* initializer)
{
    auto bindingProps = initializer->GetBindingProperties();

    UINT requiredDescriptorCount = bindingProps.RequiredDescriptorCount;

    for (size_t i = 0; i < numOps; i++)
    {
        bindingProps = ops[i]->GetBindingProperties();
        requiredDescriptorCount = std::max(requiredDescriptorCount, bindingProps.RequiredDescriptorCount);
    }

    return requiredDescriptorCount;
}

void Sample::InitializeDirectMLResources()
{
    auto commandList = m_deviceResources->GetCommandList();
    commandList->Reset(m_deviceResources->GetCommandAllocator(), nullptr);

    // Create operator initializers and descriptor heap for binding
    size_t upsampleOpDescriptorCount, convOpDescriptorCount, additionOpDescriptorCount;
    size_t upsampleDescriptorsIdx, convDescriptorsIdx, additionDescriptorsIdx;

    {
        // The same descriptor heap will be used for both initializing and executing operators. These each happen
        // at different times, so we reuse the same descriptor slots. GetDescriptorCount() ensures there are enough
        // slots for both cases.
        DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(c_numUpsampleLayers, m_dmlUpsampleOps[0].GetAddressOf(), IID_PPV_ARGS(m_dmlOpInitializers[e_opUpsample].GetAddressOf())));
        upsampleOpDescriptorCount = GetDescriptorCount(c_numUpsampleLayers, m_dmlUpsampleOps[0].GetAddressOf(), m_dmlOpInitializers[e_opUpsample].Get());

        DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(c_numConvLayers, m_dmlConvOps[0].GetAddressOf(), IID_PPV_ARGS(m_dmlOpInitializers[e_opConv].GetAddressOf())));
        convOpDescriptorCount = GetDescriptorCount(c_numConvLayers, m_dmlConvOps[0].GetAddressOf(), m_dmlOpInitializers[e_opConv].Get());

        DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(1, m_dmlAddResidualOp.GetAddressOf(), IID_PPV_ARGS(m_dmlOpInitializers[e_opAdd].GetAddressOf())));
        additionOpDescriptorCount = GetDescriptorCount(1, m_dmlAddResidualOp.GetAddressOf(), m_dmlOpInitializers[e_opAdd].Get());

        upsampleDescriptorsIdx = 0;
        convDescriptorsIdx = upsampleDescriptorsIdx + upsampleOpDescriptorCount * c_numUpsampleLayers;
        additionDescriptorsIdx = convDescriptorsIdx + convOpDescriptorCount * c_numConvLayers;
        size_t descriptorCount = additionDescriptorsIdx + additionOpDescriptorCount;

        m_dmlDescriptorHeap = std::make_unique<DirectX::DescriptorHeap>(m_deviceResources->GetD3DDevice(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            descriptorCount);

        // Operator initialization dispatches will use this heap right away
        ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);
    }

    // Create any persistent resources required for the operators.
    {
        for (int i = 0; i < c_numUpsampleLayers + c_numConvLayers + 1; i++)
        {
            IDMLCompiledOperator* currentOp;
            ID3D12Resource** persistentResource;
            if (i < c_numUpsampleLayers)
            {
                currentOp = m_dmlUpsampleOps[i].Get();
                persistentResource = m_modelUpsamplePersistentResources[i].ReleaseAndGetAddressOf();
            }
            else if (i < c_numUpsampleLayers + c_numConvLayers)
            {
                currentOp = m_dmlConvOps[i - c_numUpsampleLayers].Get();
                persistentResource = m_modelConvPersistentResources[i - c_numUpsampleLayers].ReleaseAndGetAddressOf();
            }
            else
            {
                currentOp = m_dmlAddResidualOp.Get();
                persistentResource = m_modelAddPersistentResource.ReleaseAndGetAddressOf();
            }

            auto bindingProps = currentOp->GetBindingProperties();

            if (bindingProps.PersistentResourceSize > 0)
            {
                D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bindingProps.PersistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
                    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &resourceDesc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(persistentResource)));
            }
        }
    }

    // When binding input and output resources, take note of which temp resource is used at the time:
    // Layer		| Input							| Output
    // Upsample[0]	| m_modelInput					| m_modelOutput
    // Conv[0]		| m_modelInput					| m_modelIntermediateResult[0]
    // Conv[1]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Conv[2]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
    // Upsample[1]	| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Conv[3]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
    // Conv[4]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Conv[5]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
    // Conv[6]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Addition		| m_modelIntermediateResult[1], m_modelOutput | m_modelOutput

    const DML_BUFFER_BINDING emptyBufferBinding = { nullptr, 0, 0 };
    const DML_BINDING_DESC emptyBindingDesc = { DML_BINDING_TYPE_NONE, nullptr };

    // Bind D3D resources

    Microsoft::WRL::ComPtr<IDMLBindingTable> initBindingTable;

    // Upsample layers
    {
        // Bind resources for initialization.
        auto bindingProps = m_dmlOpInitializers[e_opUpsample]->GetBindingProperties();
        // The DML API guarantees that initialization never uses a persistent resource.
        assert(bindingProps.PersistentResourceSize == 0);

        DML_BINDING_TABLE_DESC tableDesc = {
            m_dmlOpInitializers[e_opUpsample].Get(),
            m_dmlDescriptorHeap->GetCpuHandle(upsampleDescriptorsIdx),
            m_dmlDescriptorHeap->GetGpuHandle(upsampleDescriptorsIdx),
            bindingProps.RequiredDescriptorCount
        };
        DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable)));

        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        DML_BUFFER_BINDING upsamplePersistentBuffers[c_numUpsampleLayers];
        DML_BINDING_DESC upsamplePersistentBindings[c_numUpsampleLayers];
        for (int i = 0; i < c_numUpsampleLayers; i++)
        {
            if (m_modelUpsamplePersistentResources[i].Get() != nullptr)
            {
                upsamplePersistentBuffers[i] = { m_modelUpsamplePersistentResources[i].Get(), 0, m_modelUpsamplePersistentResources[i]->GetDesc().Width };
                upsamplePersistentBindings[i] = { DML_BINDING_TYPE_BUFFER, &upsamplePersistentBuffers[i] };
            }
            else
                upsamplePersistentBindings[i] = emptyBindingDesc;
        }

        // The inputs will vary each frame, so don't bind inputs at initialization.
        initBindingTable->BindInputs(0, nullptr);
        initBindingTable->BindOutputs(c_numUpsampleLayers, upsamplePersistentBindings);
        BindTempResourceIfNeeded(bindingProps, initBindingTable.Get(), m_modelInitTemporaryResources[e_opUpsample].ReleaseAndGetAddressOf());

        // Run initialization
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializers[e_opUpsample].Get(), initBindingTable.Get());

        // Bind resources for execution
        for (int i = 0; i < c_numUpsampleLayers; i++)
        {
            bindingProps = m_dmlUpsampleOps[i]->GetBindingProperties();

            tableDesc = {
                m_dmlUpsampleOps[i].Get(),
                m_dmlDescriptorHeap->GetCpuHandle(upsampleDescriptorsIdx + i * upsampleOpDescriptorCount),
                m_dmlDescriptorHeap->GetGpuHandle(upsampleDescriptorsIdx + i * upsampleOpDescriptorCount),
                bindingProps.RequiredDescriptorCount
            };
            DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(m_dmlUpsampleBindings[i].ReleaseAndGetAddressOf())));

            auto inputResource = (i == 0) ? m_modelInput : m_modelIntermediateResult[0];
            auto outputResource = (i == 0) ? m_modelOutput : m_modelIntermediateResult[1];

            DML_BUFFER_BINDING inputBufferBinding = { inputResource.Get(), 0, inputResource->GetDesc().Width };
            DML_BINDING_DESC inputBinding = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
            DML_BUFFER_BINDING outputBufferBinding = { outputResource.Get(), 0, outputResource->GetDesc().Width };
            DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBufferBinding };

            m_dmlUpsampleBindings[i]->BindInputs(1, &inputBinding);
            m_dmlUpsampleBindings[i]->BindOutputs(1, &outputBinding);
            BindTempResourceIfNeeded(bindingProps, m_dmlUpsampleBindings[i].Get(), m_modelUpsampleTemporaryResources[i].ReleaseAndGetAddressOf());

            if (m_modelUpsamplePersistentResources[i].Get() != nullptr)
                m_dmlUpsampleBindings[i]->BindPersistentResource(&upsamplePersistentBindings[i]);
        }
    }

    // Convolution layers
    {
        // Bind resources for initialization
        auto bindingProps = m_dmlOpInitializers[e_opConv]->GetBindingProperties();
        assert(bindingProps.PersistentResourceSize == 0);

        DML_BINDING_TABLE_DESC tableDesc = {
            m_dmlOpInitializers[e_opConv].Get(),
            m_dmlDescriptorHeap->GetCpuHandle(convDescriptorsIdx),
            m_dmlDescriptorHeap->GetGpuHandle(convDescriptorsIdx),
            bindingProps.RequiredDescriptorCount
        };
        DX::ThrowIfFailed(initBindingTable->Reset(&tableDesc));

#if DML_MANAGED_WEIGHTS
        // Bind the weight tensors at initialization instead of at execution. This lets DirectML reformat them
        // and improve performance on some hardware.
        DML_BUFFER_BINDING convBufferBindings[][3] = {
            { emptyBufferBinding, { m_modelConvFilterWeights[0].Get(), 0, m_modelConvFilterWeights[0]->GetDesc().Width }, { m_modelConvBiasWeights[0].Get(), 0, m_modelConvBiasWeights[0]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[1].Get(), 0, m_modelConvFilterWeights[1]->GetDesc().Width }, { m_modelConvBiasWeights[1].Get(), 0, m_modelConvBiasWeights[1]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[2].Get(), 0, m_modelConvFilterWeights[2]->GetDesc().Width }, { m_modelConvBiasWeights[2].Get(), 0, m_modelConvBiasWeights[2]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[3].Get(), 0, m_modelConvFilterWeights[3]->GetDesc().Width }, { m_modelConvBiasWeights[3].Get(), 0, m_modelConvBiasWeights[3]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[4].Get(), 0, m_modelConvFilterWeights[4]->GetDesc().Width }, { m_modelConvBiasWeights[4].Get(), 0, m_modelConvBiasWeights[4]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[5].Get(), 0, m_modelConvFilterWeights[5]->GetDesc().Width }, { m_modelConvBiasWeights[5].Get(), 0, m_modelConvBiasWeights[5]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[6].Get(), 0, m_modelConvFilterWeights[6]->GetDesc().Width }, emptyBufferBinding }	// last layer has no bias
        };

        DML_BUFFER_ARRAY_BINDING convBufferArrayBindings[] = {
            { 3, convBufferBindings[0] },
            { 3, convBufferBindings[1] },
            { 3, convBufferBindings[2] },
            { 3, convBufferBindings[3] },
            { 3, convBufferBindings[4] },
            { 3, convBufferBindings[5] },
            { 3, convBufferBindings[6] },
        };

        DML_BINDING_DESC convInBindings[] = {
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[0] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[1] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[2] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[3] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[4] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[5] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[6] }
        };

        initBindingTable->BindInputs(c_numConvLayers, convInBindings);
#else
        initBindingTable->BindInputs(0, nullptr);
#endif

        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        DML_BUFFER_BINDING convPersistentBuffers[c_numConvLayers];
        DML_BINDING_DESC convPersistentBindings[c_numConvLayers];
        for (int i = 0; i < c_numConvLayers; i++)
        {
            if (m_modelConvPersistentResources[i].Get() != nullptr)
            {
                convPersistentBuffers[i] = { m_modelConvPersistentResources[i].Get(), 0, m_modelConvPersistentResources[i]->GetDesc().Width };
                convPersistentBindings[i] = { DML_BINDING_TYPE_BUFFER, &convPersistentBuffers[i] };
            }
            else
                convPersistentBindings[i] = emptyBindingDesc;
        }

        initBindingTable->BindOutputs(c_numConvLayers, convPersistentBindings);
        BindTempResourceIfNeeded(bindingProps, initBindingTable.Get(), m_modelInitTemporaryResources[e_opConv].ReleaseAndGetAddressOf());

        // Run initialization
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializers[e_opConv].Get(), initBindingTable.Get());

        // Bind resources for execution
        for (int i = 0; i < c_numConvLayers; i++)
        {
            bindingProps = m_dmlConvOps[i]->GetBindingProperties();

            tableDesc = {
                m_dmlConvOps[i].Get(),
                m_dmlDescriptorHeap->GetCpuHandle(convDescriptorsIdx + i * convOpDescriptorCount),
                m_dmlDescriptorHeap->GetGpuHandle(convDescriptorsIdx + i * convOpDescriptorCount),
                bindingProps.RequiredDescriptorCount
            };
            DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(m_dmlConvBindings[i].ReleaseAndGetAddressOf())));

            // See table at the beginning of the function for the mapping of ops to resources.
            auto inputResource = (i == 0) ? m_modelInput : ((i == 1 || i == 4 || i == 6) ? m_modelIntermediateResult[0] : m_modelIntermediateResult[1]);
            auto outputResource = (i == 1 || i == 4 || i == 6) ? m_modelIntermediateResult[1] : m_modelIntermediateResult[0];

            DML_BUFFER_BINDING inputBufferBinding = { inputResource.Get(), 0, inputResource->GetDesc().Width };
            DML_BINDING_DESC inputBinding = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };

            DML_BUFFER_BINDING outputBufferBinding = { outputResource.Get(), 0, outputResource->GetDesc().Width };
            DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBufferBinding };

#if DML_MANAGED_WEIGHTS
            // The weights are stored in the persistent resource and shouldn't be bound separately.
            DML_BINDING_DESC inputBindings[] = { inputBinding, emptyBindingDesc, emptyBindingDesc };
#else
            // Bind the weight resources
            DML_BUFFER_BINDING filterBufferBinding = { m_modelConvFilterWeights[i].Get(), 0, m_modelConvFilterWeights[i]->GetDesc().Width };
            DML_BINDING_DESC filterBinding = { DML_BINDING_TYPE_BUFFER, &filterBufferBinding };

            DML_BUFFER_BINDING biasBufferBinding;
            DML_BINDING_DESC biasBinding;
            if (i == 6)
            {
                biasBinding = emptyBindingDesc;	// last layer has no bias
            }
            else
            {
                biasBufferBinding = { m_modelConvBiasWeights[i].Get(), 0, m_modelConvBiasWeights[i]->GetDesc().Width };
                biasBinding = { DML_BINDING_TYPE_BUFFER, &biasBufferBinding };
            }

            DML_BINDING_DESC inputBindings[] = { inputBinding, filterBinding, biasBinding };
#endif
            m_dmlConvBindings[i]->BindInputs(3, inputBindings);
            m_dmlConvBindings[i]->BindOutputs(1, &outputBinding);
            BindTempResourceIfNeeded(bindingProps, m_dmlConvBindings[i].Get(), m_modelConvTemporaryResources[i].ReleaseAndGetAddressOf());

            if (m_modelConvPersistentResources[i].Get() != nullptr)
                m_dmlConvBindings[i]->BindPersistentResource(&convPersistentBindings[i]);
        }
    }

    // Addition layer
    {
        // Bind resources for initialization.
        auto bindingProps = m_dmlOpInitializers[e_opAdd]->GetBindingProperties();
        assert(bindingProps.PersistentResourceSize == 0);

        DML_BINDING_TABLE_DESC tableDesc = {
            m_dmlOpInitializers[e_opAdd].Get(),
            m_dmlDescriptorHeap->GetCpuHandle(additionDescriptorsIdx),
            m_dmlDescriptorHeap->GetGpuHandle(additionDescriptorsIdx),
            bindingProps.RequiredDescriptorCount
        };
        DX::ThrowIfFailed(initBindingTable->Reset(&tableDesc));

        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        DML_BUFFER_BINDING addPersistentBuffer;
        DML_BINDING_DESC addPersistentBinding;
        if (m_modelAddPersistentResource.Get() != nullptr)
        {
            addPersistentBuffer = { m_modelAddPersistentResource.Get(), 0, m_modelAddPersistentResource->GetDesc().Width };
            addPersistentBinding = { DML_BINDING_TYPE_BUFFER, &addPersistentBuffer };
        }
        else
            addPersistentBinding = emptyBindingDesc;

        initBindingTable->BindInputs(0, nullptr);
        initBindingTable->BindOutputs(1, &addPersistentBinding);
        BindTempResourceIfNeeded(bindingProps, initBindingTable.Get(), m_modelInitTemporaryResources[e_opAdd].ReleaseAndGetAddressOf());

        // Run initialization
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializers[e_opAdd].Get(), initBindingTable.Get());

        // Bind resources for execution
        {
            bindingProps = m_dmlAddResidualOp->GetBindingProperties();

            tableDesc = {
                m_dmlAddResidualOp.Get(),
                m_dmlDescriptorHeap->GetCpuHandle(additionDescriptorsIdx),
                m_dmlDescriptorHeap->GetGpuHandle(additionDescriptorsIdx),
                bindingProps.RequiredDescriptorCount
            };
            DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(m_dmlAddResidualBinding.ReleaseAndGetAddressOf())));

            // m_modelOutput will already hold the result of the first upsample operation. We add the result of
            // the last convolution (the residual) to it in-place to get the final result.
            DML_BUFFER_BINDING input0BufferBinding = { m_modelIntermediateResult[1].Get(), 0, m_modelIntermediateResult[1]->GetDesc().Width };
            DML_BINDING_DESC input0Binding = { DML_BINDING_TYPE_BUFFER, &input0BufferBinding };
            DML_BUFFER_BINDING input1BufferBinding = { m_modelOutput.Get(), 0, m_modelOutput->GetDesc().Width };
            DML_BINDING_DESC input1Binding = { DML_BINDING_TYPE_BUFFER, &input1BufferBinding };
            DML_BUFFER_BINDING outputBufferBinding = { m_modelOutput.Get(), 0, m_modelOutput->GetDesc().Width };
            DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBufferBinding };

            DML_BINDING_DESC inputBindings[] = { input0Binding, input1Binding };
            m_dmlAddResidualBinding->BindInputs(2, inputBindings);
            m_dmlAddResidualBinding->BindOutputs(1, &outputBinding);
            BindTempResourceIfNeeded(bindingProps, m_dmlAddResidualBinding.Get(), m_modelAddTemporaryResource.ReleaseAndGetAddressOf());

            if (m_modelAddPersistentResource.Get() != nullptr)
                m_dmlAddResidualBinding->BindPersistentResource(&addPersistentBinding);
        }
    }

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
}

void Sample::CreateUpsampleLayer(
    _In_reads_(4) const uint32_t* inputSizes,
    _Inout_updates_(1) uint64_t* inputBufferRequiredSize,
    _Inout_updates_(1) uint64_t* outputBufferRequiredSize,
    _Out_writes_(4) uint32_t* outputSizesOut,
    _Out_writes_(1) IDMLCompiledOperator** compiledOpOut)
{
    // Describe input and output tensors
    uint32_t inputStrides[4];
    Sample::GetStrides(inputSizes, m_tensorLayout, inputStrides);

    uint64_t inputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, inputSizes, inputStrides);
    // Because we can resuse resources for tensor storage, this tracks the resource size needed to hold the
    // largest possible tensor requested.
    *inputBufferRequiredSize = std::max(inputBufferSize, *inputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC inputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, inputSizes, inputStrides, inputBufferSize, 0 };
    DML_TENSOR_DESC inputDesc = { DML_TENSOR_TYPE_BUFFER, &inputBufferDesc };

    // Output size is double in height and width
    outputSizesOut[0] = inputSizes[0];
    outputSizesOut[1] = inputSizes[1];
    outputSizesOut[2] = inputSizes[2] * 2;
    outputSizesOut[3] = inputSizes[3] * 2;

    uint32_t outputStrides[4];
    Sample::GetStrides(outputSizesOut, m_tensorLayout, outputStrides);

    uint64_t outputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, outputSizesOut, outputStrides);
    *outputBufferRequiredSize = std::max(outputBufferSize, *outputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC outputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, outputSizesOut, outputStrides, outputBufferSize, 0 };
    DML_TENSOR_DESC outputDesc = { DML_TENSOR_TYPE_BUFFER, &outputBufferDesc };

    // Describe, create, and compile upsample operator
    DML_UPSAMPLE_2D_OPERATOR_DESC upsampleDesc = { &inputDesc, &outputDesc, {2, 2}, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR };
    DML_OPERATOR_DESC opDesc = { DML_OPERATOR_UPSAMPLE_2D, &upsampleDesc };

    ComPtr<IDMLOperator> op;
    DX::ThrowIfFailed(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(op.ReleaseAndGetAddressOf())));
    DX::ThrowIfFailed(m_dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(compiledOpOut)));
}

void Sample::CreateConvolutionLayer(
    _In_reads_(4) const uint32_t* inputSizes,
    _In_reads_(4) const uint32_t* filterSizes,
    bool useBiasAndActivation,
    _Inout_updates_(1) uint64_t* inputBufferRequiredSize,
    _Inout_updates_(1) uint64_t* outputBufferRequiredSize,
    _Out_writes_(4) uint32_t* outputSizesOut,
    _Out_writes_(1) IDMLCompiledOperator** compiledOpOut)
{
    // Describe input and output tensors    
    uint32_t inputStrides[4];
    Sample::GetStrides(inputSizes, m_tensorLayout, inputStrides);

    uint64_t inputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, inputSizes, inputStrides);
    *inputBufferRequiredSize = std::max(inputBufferSize, *inputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC inputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, inputSizes, inputStrides, inputBufferSize, 0 };
    DML_TENSOR_DESC inputDesc = { DML_TENSOR_TYPE_BUFFER, &inputBufferDesc };

    // The output shape has as many channels as there are convolution filters.
    outputSizesOut[0] = inputSizes[0];
    outputSizesOut[1] = filterSizes[0];
    outputSizesOut[2] = inputSizes[2];
    outputSizesOut[3] = inputSizes[3];

    uint32_t outputStrides[4];
    Sample::GetStrides(outputSizesOut, m_tensorLayout, outputStrides);

    uint64_t outputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, outputSizesOut, outputStrides);
    *outputBufferRequiredSize = std::max(outputBufferSize, *outputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC outputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, outputSizesOut, outputStrides, outputBufferSize, 0 };
    DML_TENSOR_DESC outputDesc = { DML_TENSOR_TYPE_BUFFER, &outputBufferDesc };

    // Describe weight tensors
    uint32_t filterStrides[4];
    Sample::GetStrides(filterSizes, m_tensorLayout, filterStrides);
    uint64_t filterBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, filterSizes, filterStrides);

#if DML_MANAGED_WEIGHTS
    DML_BUFFER_TENSOR_DESC filterBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_OWNED_BY_DML, 4, filterSizes, filterStrides, filterBufferSize, 0 };
#else
    DML_BUFFER_TENSOR_DESC filterBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, filterSizes, filterStrides, filterBufferSize, 0 };
#endif
    DML_TENSOR_DESC filterDesc = { DML_TENSOR_TYPE_BUFFER, &filterBufferDesc };

    uint32_t biasSizes[] = { 1, filterSizes[0], 1, 1 };	// One bias per output channel    
    uint32_t biasStrides[4];
    Sample::GetStrides(biasSizes, m_tensorLayout, biasStrides);
    uint64_t biasBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, biasSizes, biasStrides);

#if DML_MANAGED_WEIGHTS
    DML_BUFFER_TENSOR_DESC biasBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_OWNED_BY_DML, 4, biasSizes, biasStrides, biasBufferSize, 0 };
#else
    DML_BUFFER_TENSOR_DESC biasBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, biasSizes, biasStrides, biasBufferSize, 0 };
#endif
    DML_TENSOR_DESC biasDesc = { DML_TENSOR_TYPE_BUFFER, &biasBufferDesc };

    // Describe, create, and compile convolution operator

    // The output size of a convolution operation is given by:
    //  height = (inputHeight - filterHeight + 2*paddingHeight) / filterStride + 1
    //  width  = (inputWidth  - filterWidth  + 2*paddingWidth ) / filterStride + 1
    //
    // We want to preserve the height and width, so assuming stride is 1, we get:
    //  paddingHeight = (filterHeight - 1) / 2
    //  paddingWidth  = (filterWidth  - 1) / 2
    // If padding is fractional, we pad unevenly with ceil/floor.
    UINT paddingHeightTop = static_cast<UINT>(ceil((filterSizes[2] - 1) / 2.0f));
    UINT paddingHeightBottom = static_cast<UINT>(floor((filterSizes[2] - 1) / 2.0f));
    UINT paddingWidthLeft = static_cast<UINT>(ceil((filterSizes[3] - 1) / 2.0f));
    UINT paddingWidthRight = static_cast<UINT>(floor((filterSizes[3] - 1) / 2.0f));

    UINT strides[] = { 1, 1 };
    UINT dilations[] = { 1, 1 };
    UINT startPadding[] = { paddingHeightTop, paddingWidthLeft };
    UINT endPadding[] = { paddingHeightBottom, paddingWidthRight };
    UINT outputPadding[] = { 0, 0 };

    DML_ACTIVATION_RELU_OPERATOR_DESC fusedReluDesc = { 0 };
    DML_OPERATOR_DESC activationDesc = { DML_OPERATOR_ACTIVATION_RELU, &fusedReluDesc };

    DML_CONVOLUTION_OPERATOR_DESC convDesc = {
        &inputDesc,
        &filterDesc,
        useBiasAndActivation ? &biasDesc : nullptr,
        &outputDesc,
        DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        DML_CONVOLUTION_DIRECTION_FORWARD,
        2,
        strides,
        dilations,
        startPadding,
        endPadding,
        outputPadding,
        1,
        useBiasAndActivation ? &activationDesc : nullptr
    };
    DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CONVOLUTION, &convDesc };

    ComPtr<IDMLOperator> op;
    DX::ThrowIfFailed(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(op.ReleaseAndGetAddressOf())));
    DX::ThrowIfFailed(m_dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(compiledOpOut)));
}

void Sample::CreateAdditionLayer(
    _In_reads_(4) const uint32_t* inputSizes,
    _Out_writes_(1) IDMLCompiledOperator** compiledOpOut)
{
    // Describe input and output tensors
    uint32_t strides[4];
    Sample::GetStrides(inputSizes, m_tensorLayout, strides);
    uint64_t bufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, inputSizes, strides);

    DML_BUFFER_TENSOR_DESC bufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, inputSizes, strides, bufferSize, 0 };
    DML_TENSOR_DESC tensorDesc = { DML_TENSOR_TYPE_BUFFER, &bufferDesc };

    // Describe, create, and compile elementwise addition operator
    // Inputs and output are all the same size and use the same tensor desc.
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc = { &tensorDesc, &tensorDesc, &tensorDesc };
    DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc };

    ComPtr<IDMLOperator> op;
    DX::ThrowIfFailed(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(op.ReleaseAndGetAddressOf())));
    DX::ThrowIfFailed(m_dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(compiledOpOut)));
}


