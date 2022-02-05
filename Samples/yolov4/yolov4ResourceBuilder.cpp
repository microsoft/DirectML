#include "pch.h"

#include "yolov4.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"
#include "WeightLoader.h"

using Microsoft::WRL::ComPtr;

using namespace DirectX;

class YoloV4
{
public:
    struct ModelOutputs
    {
        dml::Expression convSBBox;
        dml::Expression convMBBox;
        dml::Expression convLBBox;
    };

    explicit YoloV4(dml::Graph* graph, dml::Expression input, uint32_t numClasses)
        : m_graph(graph)
        , m_weightLoader(graph, 1)
    {
        m_modelOutputs = BuildModel(input, numClasses);
    }

    WeightData LoadWeightDataFromFile(const wchar_t* path, DX::DeviceResources* deviceResources)
    {
        return m_weightLoader.LoadWeightDataFromFile(path, deviceResources);
    }

    ModelOutputs GetModelOutputs() const
    {
        return m_modelOutputs;
    }

private:
    dml::Graph* m_graph;
    ModelOutputs m_modelOutputs;
    WeightLoader m_weightLoader;

private:
    struct Backbone
    {
        dml::Expression route1;
        dml::Expression route2;
        dml::Expression conv;
    };

    enum class Activation
    {
        None,
        LeakyRelu,
        Mish,
    };

    static dml::Expression Mish(dml::Expression x)
    {
        return x * dml::ActivationTanh(dml::ActivationSoftplus(x));
    }

    dml::Expression Convolutional(
        dml::Expression input,
        dml::TensorDesc::Dimensions filterShape,
        bool downsample = false,
        bool hasBatchNorm = true,
        Activation activation = Activation::LeakyRelu)
    {
        auto weights = m_weightLoader.RegisterConvWeights(filterShape, hasBatchNorm);

        uint32_t filterHeight = weights.filter.GetOutputDesc().sizes[2];
        uint32_t filterWidth = weights.filter.GetOutputDesc().sizes[3];
        std::array<uint32_t, 2> padding = { filterHeight / 2, filterWidth / 2 };

        std::array<uint32_t, 2> strides = {};
        if (downsample)
        {
            strides = { 2, 2 };
        }
        else
        {
            strides = { 1, 1 };
        }

        dml::FusedActivation fusedActivation = dml::FusedActivation::None();
        if (activation == Activation::LeakyRelu)
        {
            // LeakyRelu gets fused into the conv
            fusedActivation = dml::FusedActivation::LeakyRelu(0.1f);
        }

        auto conv = dml::ConvolutionBuilder(input, weights.filter, weights.bias)
            .StartPadding(padding)
            .EndPadding(padding)
            .Strides(strides)
            .FusedActivation(fusedActivation)
            .Build();

        if (activation == Activation::Mish)
        {
            conv = Mish(conv);
        }

        return conv;
    }

    dml::Expression ResidualBlock(
        dml::Expression input,
        uint32_t inputChannel,
        uint32_t filterCount1,
        uint32_t filterCount2,
        Activation activation)
    {
        auto shortcut = input;
        auto conv = input;

        conv = Convolutional(conv, { filterCount1, inputChannel, 1, 1 }, false, true, activation);
        conv = Convolutional(conv, { filterCount2, filterCount1, 3, 3 }, false, true, activation);

        return (shortcut + conv);
    }

    dml::Expression MaxPool(dml::Expression input, uint32_t windowHeight, uint32_t windowWidth)
    {
        uint32_t paddingH = windowHeight / 2;
        uint32_t paddingW = windowWidth / 2;

        auto [output, _] = dml::MaxPoolingBuilder(input, { windowHeight, windowWidth })
            .Strides({ 1, 1 })
            .StartPadding({ paddingH, paddingW })
            .EndPadding({ paddingH, paddingW })
            .Build();
        return output;
    }

    dml::Expression Upsample(dml::Expression input)
    {
        return dml::Upsample2D(input, { 2, 2 }, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR);
    }

    Backbone CspDarknet53(dml::Expression input)
    {
        const uint32_t joinAxis = 1; // Concatenate along channels
        dml::Expression route;

        input = Convolutional(input, { 32, 3, 3, 3 }, false, true, Activation::Mish);
        input = Convolutional(input, { 64, 32, 3, 3 }, true, true, Activation::Mish);

        route = Convolutional(input, { 64, 64, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 64, 64, 1, 1 }, false, true, Activation::Mish);
        for (uint32_t i = 0; i < 1; ++i)
            input = ResidualBlock(input, 64, 32, 64, Activation::Mish);
        input = Convolutional(input, { 64, 64, 1, 1 }, false, true, Activation::Mish);
        input = dml::Join({ input, route }, joinAxis);

        input = Convolutional(input, { 64, 128, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 128, 64, 3, 3 }, true, true, Activation::Mish);
        route = Convolutional(input, { 64, 128, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 64, 128, 1, 1 }, false, true, Activation::Mish);
        for (uint32_t i = 0; i < 2; ++i)
            input = ResidualBlock(input, 64, 64, 64, Activation::Mish);
        input = Convolutional(input, { 64, 64, 1, 1 }, false, true, Activation::Mish);
        input = dml::Join({ input, route }, joinAxis);

        input = Convolutional(input, { 128, 128, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 256, 128, 3, 3 }, true, true, Activation::Mish);
        route = Convolutional(input, { 128, 256, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 128, 256, 1, 1 }, false, true, Activation::Mish);
        for (uint32_t i = 0; i < 8; ++i)
            input = ResidualBlock(input, 128, 128, 128, Activation::Mish);
        input = Convolutional(input, { 128, 128, 1, 1 }, false, true, Activation::Mish);
        input = dml::Join({ input, route }, joinAxis);

        input = Convolutional(input, { 256, 256, 1, 1 }, false, true, Activation::Mish);
        auto route1 = input;
        input = Convolutional(input, { 512, 256, 3, 3 }, true, true, Activation::Mish);
        route = Convolutional(input, { 256, 512, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 256, 512, 1, 1 }, false, true, Activation::Mish);
        for (uint32_t i = 0; i < 8; ++i)
            input = ResidualBlock(input, 256, 256, 256, Activation::Mish);
        input = Convolutional(input, { 256, 256, 1, 1 }, false, true, Activation::Mish);
        input = dml::Join({ input, route }, joinAxis);

        input = Convolutional(input, { 512, 512, 1, 1 }, false, true, Activation::Mish);
        auto route2 = input;
        input = Convolutional(input, { 1024, 512, 3, 3 }, true, true, Activation::Mish);
        route = Convolutional(input, { 512, 1024, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 512, 1024, 1, 1 }, false, true, Activation::Mish);
        for (uint32_t i = 0; i < 4; ++i)
            input = ResidualBlock(input, 512, 512, 512, Activation::Mish);
        input = Convolutional(input, { 512, 512, 1, 1 }, false, true, Activation::Mish);
        input = dml::Join({ input, route }, joinAxis);

        input = Convolutional(input, { 1024, 1024, 1, 1 }, false, true, Activation::Mish);
        input = Convolutional(input, { 512, 1024, 1, 1 }, false, true, Activation::LeakyRelu);
        input = Convolutional(input, { 1024, 512, 3, 3}, false, true, Activation::LeakyRelu);
        input = Convolutional(input, { 512, 1024, 1, 1}, false, true, Activation::LeakyRelu);

        auto pool1 = MaxPool(input, 13, 13);
        auto pool2 = MaxPool(input, 9, 9);
        auto pool3 = MaxPool(input, 5, 5);
        input = dml::Join({ pool1, pool2, pool3, input }, joinAxis);

        input = Convolutional(input, { 512, 2048, 1, 1});
        input = Convolutional(input, { 1024, 512, 3, 3});
        input = Convolutional(input, { 512, 1024, 1, 1});

        return Backbone{ route1, route2, input };
    }

    ModelOutputs BuildModel(dml::Expression input, uint32_t numClasses)
    {
        auto [route1, route2, conv] = CspDarknet53(input);

        auto route = conv;
        const uint32_t joinAxis = 1; // Concatenate along channels
        
        conv = Convolutional(conv, { 256, 512, 1, 1 });
        conv = Upsample(conv);
        route2 = Convolutional(route2, { 256, 512, 1, 1 });
        conv = dml::Join({ route2, conv }, joinAxis);

        conv = Convolutional(conv, { 256, 512, 1, 1 });
        conv = Convolutional(conv, { 512, 256, 3, 3 });
        conv = Convolutional(conv, { 256, 512, 1, 1 });
        conv = Convolutional(conv, { 512, 256, 3, 3 });
        conv = Convolutional(conv, { 256, 512, 1, 1 });

        route2 = conv;
        conv = Convolutional(conv, { 128, 256, 1, 1 });
        conv = Upsample(conv);
        route1 = Convolutional(route1, { 128, 256, 1, 1 });
        conv = dml::Join({ route1, conv }, joinAxis);

        conv = Convolutional(conv, { 128, 256, 1, 1 });
        conv = Convolutional(conv, { 256, 128, 3, 3 });
        conv = Convolutional(conv, { 128, 256, 1, 1 });
        conv = Convolutional(conv, { 256, 128, 3, 3 });
        conv = Convolutional(conv, { 128, 256, 1, 1 });

        route1 = conv;
        conv = Convolutional(conv, { 256, 128, 3, 3 });
        auto convSBBox = Convolutional(conv, { 3 * (numClasses + 5), 256, 1, 1 }, false, false, Activation::None);

        conv = Convolutional(route1, { 256, 128, 3, 3 }, true);
        conv = dml::Join({ conv, route2 }, joinAxis);

        conv = Convolutional(conv, { 256, 512, 1, 1 });
        conv = Convolutional(conv, { 512, 256, 3, 3 });
        conv = Convolutional(conv, { 256, 512, 1, 1 });
        conv = Convolutional(conv, { 512, 256, 3, 3 });
        conv = Convolutional(conv, { 256, 512, 1, 1 });

        route2 = conv;
        conv = Convolutional(conv, { 512, 256, 3, 3 });
        auto convMBBox = Convolutional(conv, { 3 * (numClasses + 5), 512, 1, 1 }, false, false, Activation::None);

        conv = Convolutional(route2, { 512, 256, 3, 3 }, true);
        conv = dml::Join({ conv, route }, joinAxis);

        conv = Convolutional(conv, { 512, 1024, 1, 1 });
        conv = Convolutional(conv, { 1024, 512, 3, 3 });
        conv = Convolutional(conv, { 512, 1024, 1, 1 });
        conv = Convolutional(conv, { 1024, 512, 3, 3 });
        conv = Convolutional(conv, { 512, 1024, 1, 1 });

        conv = Convolutional(conv, { 1024, 512, 3, 3 });
        auto convLBBox = Convolutional(conv, { 3 * (numClasses + 5), 1024, 1, 1 }, false, false, Activation::None);

        return ModelOutputs{ convSBBox, convMBBox, convLBBox };
    }
};

// Takes a tensor of size [1, 3 * (5 + numClasses), H, W] and returns a tensor of size [3, H, W, 7]. 
// Sigmoid activation is applied to all channels that represent probabilities (which are not all of them).
dml::Expression DecodeModelOutput(dml::Expression output, uint32_t numClasses)
{
    const auto& outputSizes = output.GetOutputDesc().sizes;

    assert(outputSizes.size() == 4); // Expect 4 dimensions
    assert(outputSizes[0] == 1); // Expect batch of 1
    assert(outputSizes[1] == 3 * (numClasses + 5)); // Expect # of channels to equal 3 * (numClasses+5)
    assert(outputSizes[2] == outputSizes[3]); // Expect width == height

    // Expand the channel into the batch, so that instead of:
    //    [1, 3 * (5 + numClasses), H, W] 
    // The shape is now:
    //    [3, 5 + numClasses, H, W]
    // Since this doesn't transform the data any, this can be accomplished with a simple reinterpret.
    output = dml::Reinterpret(output, { 3, numClasses + 5, outputSizes[2], outputSizes[3] }, dml::NullOpt);

    // Split the new channel (of size 5+numClasses) into 4 different tensors with channels of 2, 2, 1, numClasses.
    // These represent the box xy, box wh, confidence, and probabilities for each class.
    const uint32_t channelDim = 1;
    std::vector<dml::Expression> split = dml::Split(output, channelDim, { 2, 2, 1, numClasses });
    assert(split.size() == 4);

    // Convenience
    auto convXy = split[0];
    auto convWh = split[1];
    auto convConf = split[2];
    auto convProb = split[3];

    // Apply final activations
    convXy = dml::ActivationSigmoid(convXy);
    convWh = dml::Exp(convWh);
    convConf = dml::ActivationSigmoid(convConf);
    convProb = dml::ActivationSigmoid(convProb);

    // Compute the max and argmax of the probabilities. The argmax outputs UINT32 indices which
    // are reinterpreted as float so they can be joined into the same output tensor.
    auto convProbMax = dml::Reduce(convProb, DML_REDUCE_FUNCTION_MAX, { channelDim });
    auto convProbArgMax = dml::Reduce(convProb, DML_REDUCE_FUNCTION_ARGMAX, { channelDim });
    convProbArgMax = dml::Reinterpret(convProbArgMax, DML_TENSOR_DATA_TYPE_FLOAT32);

    // Join the tensors along channel dimension.
    auto joined = dml::Join({ convXy, convWh, convConf, convProbMax, convProbArgMax }, channelDim);

    // Transpose from NCHW to NHWC for faster reading on the CPU (converts output from SoA to AoS).
    dml::TensorDimensions sizesNchw = joined.GetOutputDesc().sizes;
    dml::TensorDimensions sizesNhwc = { sizesNchw[0], sizesNchw[3], sizesNchw[2], sizesNchw[1] };
    dml::TensorStrides stridesNhwc = { sizesNchw[1] * sizesNchw[2] * sizesNchw[3], sizesNchw[3], 1, sizesNchw[2] * sizesNchw[3] };
    return dml::Identity(dml::Reinterpret(joined, sizesNhwc, stridesNhwc));
}

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
    }

    // DirectML device
    {
#if _DEBUG
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice)));
#else
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
#endif

        DX::ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder)));
    }

    // Build the DirectML graph
    {
        dml::Graph graph(m_dmlDevice.Get());

        dml::TensorDesc::Dimensions inputSizes = { 1, 3, m_origTextureHeight, m_origTextureWidth };
        auto input = dml::InputTensor(graph, 0, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, inputSizes));

        uint64_t modelInputBufferSize = input.GetOutputDesc().totalTensorSizeInBytes;

        // Bilinearly rescale the input image to 608x608, which is what yolov4 expects
        auto modelInputSizes = { 1u, 3u, YoloV4Constants::c_inputHeight, YoloV4Constants::c_inputWidth };
        input = dml::Resample(input, modelInputSizes, DML_INTERPOLATION_MODE_LINEAR);

        // Construct the yolov4 model
        YoloV4 model(&graph, input, YoloV4Constants::c_numClasses);
        auto [convSBBox, convMBBox, convLBBox] = model.GetModelOutputs();

        // Decode the outputs of the model
        auto sbbox = DecodeModelOutput(convSBBox, YoloV4Constants::c_numClasses);
        auto mbbox = DecodeModelOutput(convMBBox, YoloV4Constants::c_numClasses);
        auto lbbox = DecodeModelOutput(convLBBox, YoloV4Constants::c_numClasses);

        // Load the model weights from file
        m_modelWeights = model.LoadWeightDataFromFile(LR"(.\Data\yolov4.weights)", m_deviceResources.get());

        // Compile the model into a DML graph
        DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        m_dmlGraph = graph.Compile(executionFlags, { sbbox, mbbox, lbbox });


        // Buffers for DML inputs and outputs

        // Resource for input tensor
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(modelInputBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelInput)));

        // Describe and create a UAV for the original input tensor.
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = static_cast<UINT>(modelInputBufferSize / sizeof(float));
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        device->CreateUnorderedAccessView(m_modelInput.Get(), nullptr, &uavDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descModelInput));

        // Create resources to hold the model outputs and to read them back from the GPU
        m_modelSOutput.desc = sbbox.GetOutputDesc();
        uint64_t sbboxResourceSize = m_modelSOutput.desc.totalTensorSizeInBytes;
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(sbboxResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelSOutput.output)));
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(sbboxResourceSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_modelSOutput.readback)));

        m_modelMOutput.desc = mbbox.GetOutputDesc();
        uint64_t mbboxResourceSize = m_modelMOutput.desc.totalTensorSizeInBytes;
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(mbboxResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelMOutput.output)));
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(mbboxResourceSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_modelMOutput.readback)));

        m_modelLOutput.desc = lbbox.GetOutputDesc();
        uint64_t lbboxResourceSize = m_modelLOutput.desc.totalTensorSizeInBytes;
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(lbboxResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelLOutput.output)));
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(lbboxResourceSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_modelLOutput.readback)));
    }
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
        std::max(executeBindingProps.RequiredDescriptorCount, 1u));

    auto initDescriptorHeap = std::make_unique<DescriptorHeap>(
        m_deviceResources->GetD3DDevice(),
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        std::max(initBindingProps.RequiredDescriptorCount, 1u));

    // Operator initialization dispatches will use this heap right away
    ID3D12DescriptorHeap* pHeaps[] = { initDescriptorHeap->Heap() };
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
        initDescriptorHeap->GetCpuHandle(0),
        initDescriptorHeap->GetGpuHandle(0),
        initBindingProps.RequiredDescriptorCount
    };
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable)));

    // Create the binding table for execution
    tableDesc =
    {
        m_dmlGraph.Get(),
        m_dmlDescriptorHeap->GetCpuHandle(0),
        m_dmlDescriptorHeap->GetGpuHandle(0),
        executeBindingProps.RequiredDescriptorCount
    };
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&m_dmlBindingTable)));

    DML_BUFFER_BINDING inputBufferBinding{ m_modelInput.Get(), 0, m_modelInput->GetDesc().Width };
    dml::Span<const DML_BUFFER_BINDING> weightBufferBindings = m_modelWeights->GetBindings();

    // Bind inputs for initialization, which is only necessary if we're using OWNED_BY_DML

#if DML_MANAGED_WEIGHTS
    {
        std::vector<DML_BUFFER_BINDING> initBufferBindings;
        initBufferBindings.push_back(DML_BUFFER_BINDING{}); // Model input
        initBufferBindings.insert(initBufferBindings.end(), weightBufferBindings.begin(), weightBufferBindings.end()); // Weights

        DML_BUFFER_ARRAY_BINDING initInputBinding = { (UINT)initBufferBindings.size(), initBufferBindings.data() };
        initBindingTable->BindInputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &initInputBinding });
    }
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

        m_dmlBindingTable->BindPersistentResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    if (m_modelTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { m_modelTemporaryResource.Get(), 0, m_modelTemporaryResource->GetDesc().Width };
        m_dmlBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    // Bind model inputs and outputs
    std::vector<DML_BINDING_DESC> inputBindings(1 + weightBufferBindings.size());
#if DML_MANAGED_WEIGHTS
    // Bind only the model input
    inputBindings[0] = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    m_dmlBindingTable->BindInputs((UINT)inputBindings.size(), inputBindings.data());
#else
    // Bind everything
    inputBindings[0] = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    for (size_t i = 0; i < weightBufferBindings.size(); ++i)
    {
        inputBindings[i + 1] = { DML_BINDING_TYPE_BUFFER, &weightBufferBindings[i] };
    }
    m_dmlBindingTable->BindInputs((UINT)inputBindings.size(), inputBindings.data());
#endif

    DML_BUFFER_BINDING outputBufferBindings[] =
    {
        { m_modelSOutput.output.Get(), 0, m_modelSOutput.output->GetDesc().Width },
        { m_modelMOutput.output.Get(), 0, m_modelMOutput.output->GetDesc().Width },
        { m_modelLOutput.output.Get(), 0, m_modelLOutput.output->GetDesc().Width },
    };

    DML_BINDING_DESC outputBindings[] =
    {
        { DML_BINDING_TYPE_BUFFER, &outputBufferBindings[0] },
        { DML_BINDING_TYPE_BUFFER, &outputBufferBindings[1] },
        { DML_BINDING_TYPE_BUFFER, &outputBufferBindings[2] },
    };

    m_dmlBindingTable->BindOutputs(ARRAYSIZE(outputBindings), outputBindings);

    // Record the initialization
    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializer.Get(), initBindingTable.Get());

    DX::ThrowIfFailed(commandList->Close());
    m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    // Wait until initialization has been finished on the GPU.
    m_deviceResources->WaitForGpu();

#if DML_MANAGED_WEIGHTS
    m_modelWeights.reset();
#endif
}
