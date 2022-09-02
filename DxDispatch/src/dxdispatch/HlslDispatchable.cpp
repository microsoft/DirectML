#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "CommandLineArgs.h"
#include "Dispatchable.h"
#include "HlslDispatchable.h"

using Microsoft::WRL::ComPtr;

HlslDispatchable::HlslDispatchable(std::shared_ptr<Device> device, const Model::HlslDispatchableDesc& desc, const CommandLineArgs& args) 
    : m_device(device), m_desc(desc), m_forceDisablePrecompiledShadersOnXbox(args.ForceDisablePrecompiledShadersOnXbox())
{
}

HlslDispatchable::BufferViewType GetViewType(const D3D12_SHADER_INPUT_BIND_DESC& desc)
{
    if ((desc.Dimension != D3D_SRV_DIMENSION_BUFFER) && 
        (desc.Type != D3D_SIT_CBUFFER) &&
        (desc.Type != D3D_SIT_TBUFFER))
    {
        throw std::invalid_argument("Only buffers are supported");
    }

    switch (desc.Type)
    {
    case D3D_SIT_TEXTURE: // Buffer
    case D3D_SIT_UAV_RWTYPED: // RWBuffer
    case D3D_SIT_TBUFFER: // tbuffer
        return HlslDispatchable::BufferViewType::Typed;

    case D3D_SIT_CBUFFER: // cbuffer
    case D3D_SIT_STRUCTURED: // StructuredBuffer
    case D3D_SIT_UAV_RWSTRUCTURED: // RWStructuredBuffer
    case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER: // (Append|Consume)StructuredBuffer
        return HlslDispatchable::BufferViewType::Structured;

    case D3D_SIT_BYTEADDRESS: // ByteAddresBuffer
    case D3D_SIT_UAV_RWBYTEADDRESS: // RWByteAddressBuffer
        return HlslDispatchable::BufferViewType::Raw;

    default: throw std::invalid_argument("Shader input type is not supported");
    }
}

D3D12_DESCRIPTOR_RANGE_TYPE GetDescriptorRangeType(const D3D12_SHADER_INPUT_BIND_DESC& desc)
{
    if ((desc.Dimension != D3D_SRV_DIMENSION_BUFFER) && 
        (desc.Type != D3D_SIT_CBUFFER) &&
        (desc.Type != D3D_SIT_TBUFFER))
    {
        throw std::invalid_argument("Only buffers are supported");
    }

    switch (desc.Type)
    {
    case D3D_SIT_CBUFFER: // cbuffer
        return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    case D3D_SIT_TEXTURE: // Buffer
    case D3D_SIT_STRUCTURED: // StructuredBuffer
    case D3D_SIT_BYTEADDRESS: // ByteAddresBuffer
    case D3D_SIT_TBUFFER: // tbuffer
        return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    
    case D3D_SIT_UAV_RWTYPED: // RWBuffer
    case D3D_SIT_UAV_RWSTRUCTURED: // RWStructuredBuffer
    case D3D_SIT_UAV_RWBYTEADDRESS: // RWByteAddressBuffer
    case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER: // (Append|Consume)StructuredBuffer
        return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;

    default: throw std::invalid_argument("Shader input type is not supported");
    }
}

using BindingData = std::tuple<
    std::vector<D3D12_DESCRIPTOR_RANGE1>, 
    std::unordered_map<std::string, HlslDispatchable::BindPoint>>;

// Reflects descriptor ranges and binding points from the HLSL source.
BindingData ReflectBindingData(gsl::span<D3D12_SHADER_INPUT_BIND_DESC> shaderInputDescs)
{
    std::vector<D3D12_DESCRIPTOR_RANGE1> descriptorRanges;
    std::unordered_map<std::string, HlslDispatchable::BindPoint> bindPoints;

    D3D12_DESCRIPTOR_RANGE1 currentRange = {};
    currentRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    uint32_t currentOffsetInDescriptors = 0;

    for (size_t resourceIndex = 0; resourceIndex < shaderInputDescs.size(); resourceIndex++)
    {
        const auto& shaderInputDesc = shaderInputDescs[resourceIndex];
        auto viewType = GetViewType(shaderInputDesc);
        auto rangeType = GetDescriptorRangeType(shaderInputDesc);
        auto numDescriptors = shaderInputDesc.BindCount;

        bindPoints[shaderInputDesc.Name] = { 
            viewType, 
            rangeType, 
            currentOffsetInDescriptors,
            (viewType == HlslDispatchable::BufferViewType::Structured ? shaderInputDesc.NumSamples : 0)
        };

        if (rangeType == currentRange.RangeType && shaderInputDesc.Space == currentRange.RegisterSpace)
        {
            currentRange.NumDescriptors += numDescriptors;
        }
        else
        {
            if (currentRange.NumDescriptors > 0)
            {
                descriptorRanges.push_back(currentRange);
            }

            currentRange.RangeType = rangeType;
            currentRange.NumDescriptors = numDescriptors;
            currentRange.RegisterSpace = shaderInputDesc.Space;
        }

        currentOffsetInDescriptors += numDescriptors;
    }

    if (currentRange.NumDescriptors > 0)
    {
        descriptorRanges.push_back(currentRange);
    }

    return std::make_tuple(descriptorRanges, bindPoints);
}

void HlslDispatchable::CreateRootSignatureAndBindingMap()
{
    D3D12_SHADER_DESC shaderDesc = {};
    THROW_IF_FAILED(m_shaderReflection->GetDesc(&shaderDesc));
    
    std::vector<D3D12_SHADER_INPUT_BIND_DESC> shaderInputDescs(shaderDesc.BoundResources);
    for (uint32_t resourceIndex = 0; resourceIndex < shaderDesc.BoundResources; resourceIndex++)
    {
        THROW_IF_FAILED(m_shaderReflection->GetResourceBindingDesc(resourceIndex, &shaderInputDescs[resourceIndex]));
    }

    std::vector<D3D12_ROOT_PARAMETER1> rootParameters;
    auto [descriptorRanges, bindPoints] = ReflectBindingData(shaderInputDescs);
    m_bindPoints = bindPoints;

    if (!descriptorRanges.empty())
    {
        D3D12_ROOT_PARAMETER1 rootParameter = {};
        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(descriptorRanges.size());
        rootParameter.DescriptorTable.pDescriptorRanges = descriptorRanges.data();
        rootParameters.push_back(rootParameter);
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rootSigDesc.Desc_1_1.NumParameters = static_cast<UINT>(rootParameters.size());
    rootSigDesc.Desc_1_1.pParameters = rootParameters.data();

    ComPtr<ID3DBlob> rootSignatureBlob;
    ComPtr<ID3DBlob> rootSignatureErrors;
#ifdef _GAMING_XBOX
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &rootSignatureBlob, &rootSignatureErrors);
#else
    HRESULT hr = m_device->D3DModule()->SerializeVersionedRootSignature(&rootSigDesc, &rootSignatureBlob, &rootSignatureErrors);
#endif
    if (FAILED(hr))
    {
        if (rootSignatureErrors)
        {
            LogError(static_cast<LPCSTR>(rootSignatureErrors->GetBufferPointer()));
        }
        THROW_HR(hr);
    }

    THROW_IF_FAILED(m_device->D3D()->CreateRootSignature(
        0, 
        rootSignatureBlob->GetBufferPointer(), 
        rootSignatureBlob->GetBufferSize(), 
        IID_GRAPHICS_PPV_ARGS(m_rootSignature.ReleaseAndGetAddressOf())));
}

void HlslDispatchable::CompileWithDxc()
{
    if (!m_device->GetDxcCompiler())
    {
        throw std::runtime_error("DXC is not available for this platform");
    }

    ComPtr<IDxcBlobEncoding> source;
    THROW_IF_FAILED(m_device->GetDxcUtils()->LoadFile(
        m_desc.sourcePath.c_str(), 
        nullptr, 
        &source));

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = source->GetBufferPointer();
    sourceBuffer.Size = source->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_ACP;

    std::vector<std::wstring> compilerArgs(m_desc.compilerArgs.size());
    for (size_t i = 0; i < m_desc.compilerArgs.size(); i++)
    {
        compilerArgs[i] = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(m_desc.compilerArgs[i]);
    }

#ifdef _GAMING_XBOX
    if (m_forceDisablePrecompiledShadersOnXbox)
    {
        compilerArgs.push_back(L"-D");
        compilerArgs.push_back(L"__XBOX_DISABLE_PRECOMPILE");
    }
#endif

    std::vector<LPCWSTR> lpcwstrArgs(compilerArgs.size());
    for (size_t i = 0; i < compilerArgs.size(); i++)
    {
        lpcwstrArgs[i] = compilerArgs[i].data();
    }

    ComPtr<IDxcResult> result;
    THROW_IF_FAILED(m_device->GetDxcCompiler()->Compile(
        &sourceBuffer, 
        lpcwstrArgs.data(), 
        static_cast<UINT32>(lpcwstrArgs.size()), 
        m_device->GetDxcIncludeHandler(), 
        IID_PPV_ARGS(&result)));

    ComPtr<IDxcBlobUtf8> errors;
    THROW_IF_FAILED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr));
    if (errors != nullptr && errors->GetStringLength() != 0)
    {
        std::string errorsString{ errors->GetStringPointer() };
        LogError(fmt::format("DXC failed to compile with errors: {}", errorsString));
    }

    HRESULT compileStatus = S_OK;
    THROW_IF_FAILED(result->GetStatus(&compileStatus));
    if (FAILED(compileStatus))
    {
        throw std::invalid_argument("Failed to compile.");
    }

    ComPtr<IDxcBlob> shaderBlob;
    THROW_IF_FAILED(result->GetOutput(
        DXC_OUT_OBJECT, 
        IID_PPV_ARGS(&shaderBlob), 
        nullptr));

    ComPtr<IDxcBlob> reflectionBlob;
    THROW_IF_FAILED(result->GetOutput(
        DXC_OUT_REFLECTION, 
        IID_PPV_ARGS(&reflectionBlob), 
        nullptr));

    ComPtr<IDxcBlob> pdbBlob;
    ComPtr<IDxcBlobUtf16> pdbName;
    if (SUCCEEDED(result->GetOutput(
        DXC_OUT_PDB, 
        IID_PPV_ARGS(&pdbBlob), 
        &pdbName)))
    {
        // TODO: store this in a temp directory?
        FILE* fp = NULL;
        _wfopen_s(&fp, pdbName->GetStringPointer(), L"wb");
        fwrite(pdbBlob->GetBufferPointer(), pdbBlob->GetBufferSize(), 1, fp);
        fclose(fp);
    }

    DxcBuffer reflectionBuffer;
    reflectionBuffer.Ptr = reflectionBlob->GetBufferPointer();
    reflectionBuffer.Size = reflectionBlob->GetBufferSize();
    reflectionBuffer.Encoding = DXC_CP_ACP;

    THROW_IF_FAILED(m_device->GetDxcUtils()->CreateReflection(
        &reflectionBuffer, 
        IID_PPV_ARGS(m_shaderReflection.ReleaseAndGetAddressOf())));

    CreateRootSignatureAndBindingMap();

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
    psoDesc.CS.BytecodeLength = shaderBlob->GetBufferSize();
    THROW_IF_FAILED(m_device->D3D()->CreateComputePipelineState(
        &psoDesc,
        IID_GRAPHICS_PPV_ARGS(m_pipelineState.ReleaseAndGetAddressOf())));

    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = static_cast<uint32_t>(m_bindPoints.size());
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(
        &descriptorHeapDesc, 
        IID_GRAPHICS_PPV_ARGS(m_descriptorHeap.ReleaseAndGetAddressOf())));
}

void HlslDispatchable::Initialize()
{
    if (m_desc.compiler == Model::HlslDispatchableDesc::Compiler::DXC)
    {
        CompileWithDxc();
    }
    else
    {
        throw std::invalid_argument("FXC isn't supported yet");
    }
}

void HlslDispatchable::Bind(const Bindings& bindings)
{
    uint32_t descriptorIncrementSize = m_device->D3D()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (auto& binding : bindings)
    {
        auto& targetName = binding.first;
        auto& sources = binding.second;
        assert(sources.size() == 1); // TODO: support multiple
        auto& source = sources[0];

        assert(source.resource != nullptr);
        assert(source.resourceDesc != nullptr);

        if (!std::holds_alternative<Model::BufferDesc>(source.resourceDesc->value))
        {
            throw std::invalid_argument("HLSL operators currently only support buffer bindings");
        }
        auto& sourceBufferDesc = std::get<Model::BufferDesc>(source.resourceDesc->value);

        auto& bindPointIterator = m_bindPoints.find(targetName);
        if (bindPointIterator == m_bindPoints.end())
        {
            throw std::invalid_argument(fmt::format("Attempting to bind shader input '{}', which does not exist (or was optimized away) in the shader.", targetName));
        }
        auto& bindPoint = bindPointIterator->second;

        CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle{
            m_descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 
            static_cast<int>(bindPoint.offsetInDescriptorsFromTableStart), 
            descriptorIncrementSize
        };

        auto FillViewDesc = [&](auto& viewDesc)
        {
            viewDesc.Buffer.StructureByteStride = bindPoint.structureByteStride;
            viewDesc.Buffer.NumElements = source.elementCount;
            viewDesc.Buffer.FirstElement = source.elementOffset;

            if (bindPoint.viewType == BufferViewType::Typed)
            {
                if (source.format)
                {
                    viewDesc.Format = *source.format;
                }
                else
                {
                    // If the binding doesn't specify, assume the data type used to initialize the buffer.
                    viewDesc.Format = Device::GetDxgiFormatFromDmlTensorDataType(sourceBufferDesc.initialValuesDataType);
                }
            }
            else if (bindPoint.viewType == BufferViewType::Structured)
            {
                if (source.format && *source.format != DXGI_FORMAT_UNKNOWN)
                {
                    throw std::invalid_argument(fmt::format("'{}' is a structured buffer, so the format must be omitted or UNKNOWN.", targetName));
                }
                viewDesc.Format = DXGI_FORMAT_UNKNOWN;
            }
            else if (bindPoint.viewType == BufferViewType::Raw)
            {
                if (source.format && *source.format != DXGI_FORMAT_R32_TYPELESS)
                {
                    throw std::invalid_argument(fmt::format("'{}' is a raw buffer, so the format must be omitted or R32_TYPELESS.", targetName));
                }

                if (sourceBufferDesc.sizeInBytes % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT != 0)
                {
                    throw std::invalid_argument(fmt::format(
                        "Attempting to bind '{}' as a raw buffer, but its size ({} bytes) is not aligned to {} bytes", 
                        source.resourceDesc->name,
                        sourceBufferDesc.sizeInBytes,
                        D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT));
                }

                viewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
                if constexpr (std::is_same_v<decltype(viewDesc), D3D12_UNORDERED_ACCESS_VIEW_DESC&>)
                {
                    viewDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                }
                if constexpr (std::is_same_v<decltype(viewDesc), D3D12_SHADER_RESOURCE_VIEW_DESC&>)
                {
                    viewDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                }
            }
        };

        if (bindPoint.descriptorType == D3D12_DESCRIPTOR_RANGE_TYPE_UAV)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            FillViewDesc(uavDesc);
            uavDesc.Buffer.CounterOffsetInBytes = source.counterOffsetBytes;
            m_device->D3D()->CreateUnorderedAccessView(source.resource, source.counterResource, &uavDesc, cpuHandle);
        }
        else if (bindPoint.descriptorType == D3D12_DESCRIPTOR_RANGE_TYPE_SRV)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            FillViewDesc(srvDesc);
            m_device->D3D()->CreateShaderResourceView(source.resource, &srvDesc, cpuHandle);
        }
        else if (bindPoint.descriptorType == D3D12_DESCRIPTOR_RANGE_TYPE_CBV)
        {
            if (sourceBufferDesc.sizeInBytes % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT != 0)
            {
                throw std::invalid_argument(fmt::format(
                    "Attempting to bind '{}' as a constant buffer, but its size ({} bytes) is not aligned to {} bytes", 
                    source.resourceDesc->name,
                    sourceBufferDesc.sizeInBytes,
                    D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT));
            }

            D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
            cbvDesc.BufferLocation = source.resource->GetGPUVirtualAddress();
            cbvDesc.SizeInBytes = sourceBufferDesc.sizeInBytes;
            m_device->D3D()->CreateConstantBufferView(&cbvDesc, cpuHandle);
        }
        else
        {
            throw std::invalid_argument("Unexpected binding type");
        }
    }

    m_device->GetCommandList()->SetComputeRootSignature(m_rootSignature.Get());
    m_device->GetCommandList()->SetPipelineState(m_pipelineState.Get());
    ID3D12DescriptorHeap* descriptorHeaps[] = { m_descriptorHeap.Get() };
    m_device->GetCommandList()->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
    m_device->GetCommandList()->SetComputeRootDescriptorTable(0, m_descriptorHeap->GetGPUDescriptorHandleForHeapStart());
}

void HlslDispatchable::Dispatch(const Model::DispatchCommand& args)
{
    m_device->GetCommandList()->Dispatch(args.threadGroupCount[0], args.threadGroupCount[1], args.threadGroupCount[2]);
}