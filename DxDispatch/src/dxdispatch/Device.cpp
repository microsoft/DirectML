#include "pch.h"
#include "CommandLineArgs.h"
#include "Device.h"
#include "DmlTracing.h"

using Microsoft::WRL::ComPtr;

// {0059DA69-B561-43D9-A39B-3355074B1082}
static const GUID PIX_EVAL_CAPTURABLE_WORK_GUID =
{ 0x59da69, 0xb561, 0x43d9, { 0xa3, 0x9b, 0x33, 0x55, 0x7, 0x4b, 0x10, 0x82 } };

// Callback to log D3D12/DirectML debug messages.
#ifdef _GAMING_XBOX
static bool DebugMessageCallback(void* context, void* commandList, DWORD messageId, const CHAR* message)
{
    LogError(message);
    return true;
}
#else
static void __stdcall DebugMessageCallback(D3D12_MESSAGE_CATEGORY cat, D3D12_MESSAGE_SEVERITY sev, D3D12_MESSAGE_ID id, LPCSTR message, void* context)
{
    if (context)
    {
        if (id == D3D12_MESSAGE_ID_META_COMMAND_UNSUPPORTED_PARAMS)
        {
            // Ignore this message, since DML internally may try to create metacommands that are not supported.
            return;
        }

        auto logger = (IDxDispatchLogger*)context;
        auto fmtMessage = fmt::format("{} {} {}", int(cat), int(id), message);
        if ((D3D12_MESSAGE_SEVERITY_INFO == sev) ||
            (D3D12_MESSAGE_SEVERITY_MESSAGE == sev))
        {
            logger->LogInfo(fmtMessage.c_str());
        }
        else if (D3D12_MESSAGE_SEVERITY_WARNING == sev)
        {
            logger->LogWarning(fmtMessage.c_str());
        }
        else
        {
            logger->LogError(fmtMessage.c_str());
        }
    }
}
#endif

Device::Device(
    IAdapter* adapter, 
    D3D_FEATURE_LEVEL featureLevel,
    DML_FEATURE_LEVEL dmlFeatureLevel,
    bool debugLayersEnabled, 
    D3D12_COMMAND_LIST_TYPE commandListType, 
    uint32_t dispatchRepeat,
    bool uavBarrierAfterDispatch,
    bool aliasingBarrierAfterDispatch,
    bool clearShaderCaches,
    bool disableGpuTimeout,
    bool enableDred,
    bool disableBackgroundProcessing,
    bool setStablePowerState,
    bool preferCustomHeaps,
    bool usePresentSeparator,
    uint32_t maxGpuTimeMeasurements,
    std::shared_ptr<PixCaptureHelper> pixCaptureHelper,
    std::shared_ptr<D3d12Module> d3dModule,
    std::shared_ptr<DmlModule> dmlModule,
    IDxDispatchLogger *logger,
    const CommandLineArgs& args
    ) : m_pixCaptureHelper(std::move(pixCaptureHelper)),
        m_d3dModule(std::move(d3dModule)),
        m_dmlModule(std::move(dmlModule)),
        m_dispatchRepeat(dispatchRepeat),
        m_logger(logger),
        m_restoreBackgroundProcessing(disableBackgroundProcessing),
        m_restoreStablePowerState(setStablePowerState),
        m_useCustomHeaps(preferCustomHeaps),
        m_args(args)
{
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = debugLayersEnabled ? DML_CREATE_DEVICE_FLAG_DEBUG : DML_CREATE_DEVICE_FLAG_NONE;

#ifdef _GAMING_XBOX
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
    params.Version = D3D12_SDK_VERSION;

    if (debugLayersEnabled)
    {
        params.ProcessDebugFlags = D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED;
    }
    if (m_pixCaptureHelper->GetPixCaptureType() != PixCaptureType::None)
    {
        // Enable the instrumented driver.
        params.ProcessDebugFlags = D3D12XBOX_PROCESS_DEBUG_FLAG_INSTRUMENTED;
    }

    params.GraphicsCommandQueueRingSizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    params.GraphicsScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    params.ComputeScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);

    THROW_IF_FAILED(D3D12XboxCreateDevice(adapter, &params, IID_GRAPHICS_PPV_ARGS(m_d3d.ReleaseAndGetAddressOf())));

    if (debugLayersEnabled)
    {
        m_d3d->SetDebugCallbackX(DebugMessageCallback, /*context*/logger);
    }
#else // !_GAMING_XBOX
    if (debugLayersEnabled)
    {
        ComPtr<ID3D12Debug3> d3dDebug;
        THROW_IF_FAILED(m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&d3dDebug)));
        d3dDebug->EnableDebugLayer();
        d3dDebug->SetEnableGPUBasedValidation(true);
    }

    if (featureLevel == D3D_FEATURE_LEVEL_1_0_GENERIC)
    {
        // Attempt to create a D3D_FEATURE_LEVEL_1_0_CORE device first, in case the device supports this
        // feature level and the D3D runtime does not support D3D_FEATURE_LEVEL_1_0_GENERIC
        HRESULT hrUnused = m_d3dModule->CreateDevice(
            adapter,
            D3D_FEATURE_LEVEL_1_0_CORE,
            IID_PPV_ARGS(&m_d3d));
    }

    if (!m_d3d)
    {
        THROW_IF_FAILED(m_d3dModule->CreateDevice(
            adapter,
            featureLevel,
            IID_PPV_ARGS(&m_d3d)));
    }

    if (enableDred)
    {
        // Enables more debug info for TDRs, can be used with Debugger 
        // extension see following link for more info:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> pDredSettings;
        if (SUCCEEDED(m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&pDredSettings))))
        {
            pDredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            pDredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
    }

    if (debugLayersEnabled)
    {
        THROW_IF_FAILED(m_d3d->QueryInterface(m_infoQueue.GetAddressOf()));
        m_infoQueue->RegisterMessageCallback(
            DebugMessageCallback, 
            D3D12_MESSAGE_CALLBACK_FLAG_NONE, 
            logger, 
            &m_callbackCookie);
    }

    if (disableBackgroundProcessing)
    {
        HRESULT hr = m_d3d->SetBackgroundProcessingMode(
            D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
            D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
            nullptr,
            nullptr
        );

        if (FAILED(hr))
        {
            m_logger->LogError("Failed to disable background processing. Do you have developer mode enabled?");
            THROW_HR(hr);
        }
    }

    if (setStablePowerState)
    {
        HRESULT hr = m_d3d->SetStablePowerState(TRUE);
        if (FAILED(hr))
        {
            m_logger->LogError("Failed to set stable power state. Do you have developer mode enabled?");
            THROW_HR(hr);
        }
    }

    D3D12_FEATURE_DATA_ARCHITECTURE1 archData = {};
    if (SUCCEEDED(m_d3d->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &archData, sizeof(archData))))
    {
        m_architectureSupport = archData;
    }

    // Custom heaps should only be used on UMA systems with cache-coherent memory.
    m_useCustomHeaps = m_useCustomHeaps && m_architectureSupport->UMA && m_architectureSupport->CacheCoherentUMA;

    D3D_FEATURE_LEVEL featureLevelsList[] = {
        D3D_FEATURE_LEVEL_1_0_GENERIC,
        D3D_FEATURE_LEVEL_1_0_CORE,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_12_1
    };

    D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};
    featureLevels.NumFeatureLevels = _countof(featureLevelsList);
    featureLevels.pFeatureLevelsRequested = featureLevelsList;
    THROW_IF_FAILED(m_d3d->CheckFeatureSupport(
        D3D12_FEATURE_FEATURE_LEVELS,
        &featureLevels,
        sizeof(featureLevels)
    ));

    // Custom heaps are optional for MCDM devices, so we also need to check for support.
    if (featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE || 
        featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_GENERIC)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS19 options = {};
        if (SUCCEEDED(m_d3d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options, sizeof(options))))
        {
            m_useCustomHeaps = m_useCustomHeaps && options.ComputeOnlyCustomHeapSupported;
        }
    }

#endif // !_GAMING_XBOX

    THROW_IF_FAILED(m_d3d->CreateFence(
        0, 
        D3D12_FENCE_FLAG_NONE, 
        IID_GRAPHICS_PPV_ARGS(m_fence.ReleaseAndGetAddressOf())));

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = disableGpuTimeout ? D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT : D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = commandListType;
    THROW_IF_FAILED(m_d3d->CreateCommandQueue(
        &queueDesc, 
        IID_GRAPHICS_PPV_ARGS(m_queue.ReleaseAndGetAddressOf())));
    m_commandListType = queueDesc.Type;

#if defined(INCLUDE_DXGI)
    // Create dummy swapchain for frame indication
    if (usePresentSeparator)
    {
        ComPtr<IDXGIFactory2> factory;
        THROW_IF_FAILED(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)));
        DXGI_SWAP_CHAIN_DESC1 desc = {0};
        desc.Width = 2;
        desc.Height = 2;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.Stereo = FALSE;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER;
        desc.BufferCount = 3;
        desc.Scaling = DXGI_SCALING_STRETCH;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        const auto hr = factory->CreateSwapChainForComposition(m_queue.Get(), &desc, nullptr, m_dummySwapChain.GetAddressOf());
        if (FAILED(hr))
        {
            m_logger->LogWarning("Creating dummy swap chain for present seperator failed");
        }
    }
#endif

    ComPtr<IDMLDevice1> dmlDevice;
    THROW_IF_FAILED(m_dmlModule->CreateDevice1(
        m_d3d.Get(), 
        dmlCreateDeviceFlags, 
        dmlFeatureLevel, 
        IID_PPV_ARGS(&dmlDevice)));

    // TODO: only wrap if required based on command line arg
    m_dmlWrapper = Microsoft::WRL::Make<WrappedDmlDevice>(
        dmlDevice.Get(), 
        m_logger.Get(),
        m_args);
    THROW_IF_FAILED(m_dmlWrapper.As<IDMLDevice1>(&m_dml));

    THROW_IF_FAILED(m_d3d->CreateCommandAllocator(
        m_commandListType,
        IID_GRAPHICS_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3d->CreateCommandList(
        0,
        m_commandListType,
        m_commandAllocator.Get(),
        nullptr,
        IID_GRAPHICS_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_dml->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));

    // Each GPU time measurement requires a pair of timestamps
    m_timestampCapacity = maxGpuTimeMeasurements * 2;

    if (m_timestampCapacity > 0)
    {
        D3D12_QUERY_HEAP_DESC queryHeapDesc;
        queryHeapDesc.Count = m_timestampCapacity;
        queryHeapDesc.NodeMask = 0;
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;

        THROW_IF_FAILED(m_d3d->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&m_timestampHeap)));
    }

    m_pixCaptureHelper->Initialize(m_queue.Get());

    if (uavBarrierAfterDispatch)
    {
        m_postDispatchBarriers.emplace_back(CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    }

    if (aliasingBarrierAfterDispatch)
    {
        m_postDispatchBarriers.emplace_back(CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr));
    }

    if (clearShaderCaches)
    {
        ClearShaderCaches();
    }
}

Device::~Device()
{
    if (m_callbackCookie != 0)
    {
        m_infoQueue->UnregisterMessageCallback(m_callbackCookie);
        m_callbackCookie = 0;
    }

    if (m_d3d)
    {
        // Restore state for certain features that may have been toggled. Normally this isn't required,
        // since the state changes don't persist beyond the process lifetime, but the real D3D12 device 
        // is a singleton that may have other refs in the process (e.g., DxDispatch instance used in 
        // a test DLL).
        
        if (m_restoreBackgroundProcessing)
        {
            (void)m_d3d->SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                nullptr,
                nullptr
            );
        }

        if (m_restoreStablePowerState)
        {
            (void)m_d3d->SetStablePowerState(FALSE);
        }
    }
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::CreatePreferredDeviceMemoryBuffer(
    uint64_t sizeInBytes, 
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    return m_useCustomHeaps ? CreateCustomBuffer(sizeInBytes, resourceFlags, alignment, heapFlags) :
                              CreateDefaultBuffer(sizeInBytes, resourceFlags, alignment, heapFlags);
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::CreateCustomBuffer(
    uint64_t sizeInBytes, 
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0, 0, 0);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps, 
        heapFlags, 
        &resourceDesc, 
        D3D12_RESOURCE_STATE_COMMON, 
        nullptr, 
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

ComPtr<ID3D12Resource> Device::CreateDefaultBuffer(
    uint64_t sizeInBytes, 
    D3D12_RESOURCE_FLAGS resourceFlags, 
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps, 
        heapFlags, 
        &resourceDesc, 
        D3D12_RESOURCE_STATE_COMMON, 
        nullptr, 
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

ComPtr<ID3D12Resource> Device::CreateUploadBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps,
        heapFlags,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

ComPtr<ID3D12Resource> Device::CreateReadbackBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps,
        heapFlags,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

void Device::WaitForGpuWorkToComplete()
{
    uint64_t nextFenceValue = m_fence->GetCompletedValue() + 1;
    THROW_IF_FAILED(m_queue->Signal(m_fence.Get(), nextFenceValue));
    THROW_IF_FAILED(m_fence->SetEventOnCompletion(nextFenceValue, nullptr));
}

void Device::RecordInitialize(IDMLDispatchable* dispatchable, IDMLBindingTable* bindingTable)
{
    m_commandRecorder->RecordDispatch(m_commandList.Get(), dispatchable, bindingTable);
}

void Device::RecordDispatch(IDMLDispatchable* dispatchable, IDMLBindingTable* bindingTable)
{
    RecordTimestamp();

    for (uint32_t i = 0; i < m_dispatchRepeat; i++)
    {
        m_commandRecorder->RecordDispatch(m_commandList.Get(), dispatchable, bindingTable);
        if (!m_postDispatchBarriers.empty())
        {
            if (m_postDispatchBarriers.size() > std::numeric_limits<uint32_t>::max())
            {
                throw std::invalid_argument(fmt::format("ResourceBarrier '{}' is too large.", m_postDispatchBarriers.size()));
            }
            m_commandList->ResourceBarrier(static_cast<uint32_t>(m_postDispatchBarriers.size()), m_postDispatchBarriers.data());
        }
    }

    RecordTimestamp();
}

void Device::RecordDispatch(const char* name, uint32_t threadGroupX, uint32_t threadGroupY, uint32_t threadGroupZ)
{
    PIXBeginEvent(m_commandList.Get(), PIX_COLOR(255, 255, 0), "HLSL: '%s'", name);
    RecordTimestamp();
    
    for (uint32_t i = 0; i < m_dispatchRepeat; i++)
    {
        m_commandList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
        if (!m_postDispatchBarriers.empty())
        {
            if (m_postDispatchBarriers.size() > std::numeric_limits<uint32_t>::max())
            {
                throw std::invalid_argument(fmt::format("ResourceBarrier '{}' is too large.", m_postDispatchBarriers.size()));
            }
            m_commandList->ResourceBarrier(static_cast<uint32_t>(m_postDispatchBarriers.size()), m_postDispatchBarriers.data());
        }
    }

    RecordTimestamp();
    PIXEndEvent(m_commandList.Get());
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::Upload(uint64_t totalSize, gsl::span<const std::byte> data, std::wstring_view name)
{
    if (data.size() > totalSize)
    {
        throw std::invalid_argument("Attempting to upload more data than the size of the buffer");
    }

    ComPtr<ID3D12Resource> buffer;
    ComPtr<ID3D12Resource> uploadBuffer;
    ComPtr<ID3D12Resource> resourceToMap;

    if (m_useCustomHeaps)
    {
        buffer = CreateCustomBuffer(totalSize);
        resourceToMap = data.empty() ? nullptr : buffer;
    }
    else
    {
        buffer = CreateDefaultBuffer(totalSize);
        uploadBuffer = data.empty() ? nullptr : CreateUploadBuffer(totalSize);
        uploadBuffer->SetName(L"Device::Upload");
        resourceToMap = uploadBuffer;
    }

    if (!name.empty())
    {
        buffer->SetName(name.data());
    }

    if (resourceToMap)
    {
        void* mappedBufferData = nullptr;
        THROW_IF_FAILED(resourceToMap->Map(0, nullptr, &mappedBufferData));
        memcpy(mappedBufferData, data.data(), data.size());
        resourceToMap->Unmap(0, nullptr);

        if (resourceToMap == uploadBuffer)
        {
            D3D12_RESOURCE_BARRIER barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition(
                    buffer.Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST)
            };

            m_commandList->ResourceBarrier(_countof(barriers), barriers);
            m_commandList->CopyResource(buffer.Get(), uploadBuffer.Get());
            std::swap(barriers[0].Transition.StateBefore, barriers[0].Transition.StateAfter);
            m_commandList->ResourceBarrier(_countof(barriers), barriers);

            m_temporaryResources.push_back(std::move(uploadBuffer));
        }
    }

    return buffer;
}

std::vector<std::byte> Device::Download(Microsoft::WRL::ComPtr<ID3D12Resource> buffer)
{
    if (buffer->GetDesc().Width > std::numeric_limits<size_t>::max())
    {
        throw std::invalid_argument(fmt::format("Buffer width '{}' is too large.", buffer->GetDesc().Width));
    }

    ComPtr<ID3D12Resource> resourceToMap;

    // Can't assume the input buffer was created as a custom heap (e.g., ONNX dispatchable with a deferred
    // resource allocated by the DML EP), so check the heap properties.
    D3D12_HEAP_PROPERTIES heapProps = {};
    D3D12_HEAP_FLAGS heapFlags = {};

    if (SUCCEEDED(buffer->GetHeapProperties(&heapProps, &heapFlags)) && 
        heapProps.MemoryPoolPreference == D3D12_MEMORY_POOL_L0 && 
        heapProps.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE)
    {
        resourceToMap = buffer;
    }
    else
    {
        resourceToMap = CreateReadbackBuffer(buffer->GetDesc().Width);
        resourceToMap->SetName(L"Device::Download");

        D3D12_RESOURCE_BARRIER barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition(
                buffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
        };

        m_commandList->ResourceBarrier(_countof(barriers), barriers);
        m_commandList->CopyResource(resourceToMap.Get(), buffer.Get());
        std::swap(barriers[0].Transition.StateBefore, barriers[0].Transition.StateAfter);
        m_commandList->ResourceBarrier(_countof(barriers), barriers);
        ExecuteCommandListAndWait();
    }

    std::vector<std::byte> outputBuffer(static_cast<size_t>(buffer->GetDesc().Width));
    
    size_t dataSize = gsl::narrow<size_t>(buffer->GetDesc().Width);
    CD3DX12_RANGE readRange(0, dataSize);
    void* mappedBufferData = nullptr;
    THROW_IF_FAILED(resourceToMap->Map(0, &readRange, &mappedBufferData));
    memcpy(outputBuffer.data(), mappedBufferData, dataSize);
    resourceToMap->Unmap(0, nullptr);

    return outputBuffer;
}

void Device::ExecuteCommandList()
{
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_queue->ExecuteCommandLists(_countof(commandLists), commandLists);
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}

void Device::ExecuteCommandListAndWait()
{
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_queue->ExecuteCommandLists(_countof(commandLists), commandLists);
    WaitForGpuWorkToComplete();
    THROW_IF_FAILED(m_d3d->GetDeviceRemovedReason());
    THROW_IF_FAILED(m_commandAllocator->Reset());
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));

    m_temporaryResources.clear();
}

void Device::RecordTimestamp()
{
    if (!GpuTimingEnabled())
    {
        return;
    }

    m_commandList->EndQuery(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, m_timestampHeadIndex);
    m_timestampHeadIndex = (m_timestampHeadIndex + 1) % m_timestampCapacity;
    if (m_timestampCount < m_timestampCapacity)
    {
        m_timestampCount++;
    }
}

std::vector<uint64_t> Device::ResolveTimestamps()
{
    assert(m_timestampCount <= m_timestampCapacity);

    if (!GpuTimingEnabled())
    {
        return {};
    }

    auto timestampReadbackBuffer = CreateReadbackBuffer(sizeof(uint64_t) * m_timestampCount);

    m_commandList->ResolveQueryData(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, m_timestampCount, timestampReadbackBuffer.Get(), 0);
    ExecuteCommandListAndWait();

    void* pData = nullptr;
    D3D12_RANGE readRange = { 0, sizeof(uint64_t) * m_timestampCount };
    timestampReadbackBuffer->Map(0, &readRange, &pData);

    std::vector<uint64_t> timestamps;
    const uint64_t* pTimestamps = reinterpret_cast<uint64_t*>(pData);
    timestamps.insert(timestamps.end(), pTimestamps, pTimestamps + m_timestampCount);

    m_timestampHeadIndex = 0;
    m_timestampCount = 0;

    return timestamps;
}

std::vector<double> Device::ResolveTimingSamples()
{
    std::vector<uint64_t> timestamps = ResolveTimestamps();
    if (timestamps.empty())
    {
        return {};
    }

    uint64_t frequency;
    THROW_IF_FAILED(m_queue->GetTimestampFrequency(&frequency));

    std::vector<double> samples(timestamps.size() / 2);

    for (uint32_t i = 0; i < samples.size(); ++i) 
    {
        uint64_t timestampDelta = (timestamps[2 * i + 1] - timestamps[2 * i]) * 1000;
        samples[i] = double(timestampDelta) / frequency / m_dispatchRepeat;
    }

    return samples;
}

#ifndef DXCOMPILER_NONE
void Device::EnsureDxcInterfaces()
{
#if defined(_GAMING_XBOX) || defined(_AMD64_)
    if (!m_dxcCompiler)
    {
        // Lazily create DXC compiler and helpers.
        THROW_IF_FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils)));
        THROW_IF_FAILED(m_dxcUtils->CreateDefaultIncludeHandler(&m_dxcIncludeHandler));
        THROW_IF_FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler)));
    }
#endif
}

IDxcUtils* Device::GetDxcUtils()
{
    EnsureDxcInterfaces();
    return m_dxcUtils.Get();
}

IDxcIncludeHandler* Device::GetDxcIncludeHandler()
{
    EnsureDxcInterfaces();
    return m_dxcIncludeHandler.Get();
}

IDxcCompiler3* Device::GetDxcCompiler()
{
    EnsureDxcInterfaces();
    return m_dxcCompiler.Get();
}
#endif // !DXCOMPILER_NONE

void Device::ClearShaderCaches()
{
    struct
    {
        D3D12_SHADER_CACHE_KIND_FLAGS kind;
        const char* name;
    } caches[] =
    {
        { D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER, "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER" },
        { D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS, "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS" },
        { D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED, "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED" },
        { D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED, "D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED" },
    };

    for (auto cache : caches)
    {
        auto hr = m_d3d->ShaderCacheControl(cache.kind, D3D12_SHADER_CACHE_CONTROL_FLAG_CLEAR);
        if (FAILED(hr))
        {
            m_logger->LogInfo(fmt::format("Clearing {} failed. Do you have developer mode enabled?", cache.name).c_str());
        }
        else
        {
            m_logger->LogInfo(fmt::format("Clearing {} succeeded.", cache.name).c_str());
        }
    }
}

/*static*/ uint32_t Device::GetSizeInBytes(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
        case DML_TENSOR_DATA_TYPE_INT8:
        case DML_TENSOR_DATA_TYPE_UINT8:
            return 1;

        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_INT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
            return 2;

        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_INT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
            return 4;
    
        case DML_TENSOR_DATA_TYPE_FLOAT64:
        case DML_TENSOR_DATA_TYPE_INT64:
        case DML_TENSOR_DATA_TYPE_UINT64:
            return 8;

        default:
            throw std::invalid_argument("Unknown data type");
    }
}

/*static*/ DXGI_FORMAT Device::GetDxgiFormatFromDmlTensorDataType(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: return DXGI_FORMAT_R16_FLOAT;
    case DML_TENSOR_DATA_TYPE_FLOAT32: return DXGI_FORMAT_R32_FLOAT;
    // case DML_TENSOR_DATA_TYPE_FLOAT64: no DXGI type exists
    case DML_TENSOR_DATA_TYPE_UINT8: return DXGI_FORMAT_R8_UINT;
    case DML_TENSOR_DATA_TYPE_UINT16: return DXGI_FORMAT_R16_UINT;
    case DML_TENSOR_DATA_TYPE_UINT32: return DXGI_FORMAT_R32_UINT;
    // case DML_TENSOR_DATA_TYPE_UINT64: no DXGI type exists
    case DML_TENSOR_DATA_TYPE_INT8: return DXGI_FORMAT_R8_SINT;
    case DML_TENSOR_DATA_TYPE_INT16: return DXGI_FORMAT_R16_SINT;
    case DML_TENSOR_DATA_TYPE_INT32: return DXGI_FORMAT_R32_SINT;
    // case DML_TENSOR_DATA_TYPE_INT64: no DXGI type exists
    default: throw std::invalid_argument(fmt::format("No DXGI_FORMAT exists for given data type."));
    }
}

void Device::DummyPresent()
{
#if defined(INCLUDE_DXGI)
    if (m_dummySwapChain)
    {
        m_dummySwapChain->Present(0, 0);
    }
#endif
}