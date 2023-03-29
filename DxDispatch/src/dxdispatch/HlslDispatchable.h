#pragma once

#include "CommandLineArgs.h"

class HlslDispatchable : public Dispatchable
{
public:
    HlslDispatchable(std::shared_ptr<Device> device, const Model::HlslDispatchableDesc& desc, const CommandLineArgs& args);

    void Initialize() final;
    void Bind(const Bindings& bindings, uint32_t iteration) final;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration) final;
    void Wait() final;

    enum class BufferViewType
    {
        Typed,      // (RW)Buffer
        Structured, // (RW|Append|Consume)StructuredBuffer
        Raw         // (RW)ByteAddresBuffer
    };

    struct BindPoint
    {
        BufferViewType viewType;
        D3D12_DESCRIPTOR_RANGE_TYPE descriptorType;
        uint32_t offsetInDescriptorsFromTableStart;
        uint32_t structureByteStride;
    };

private:
    void CompileWithDxc();
    void CreateRootSignatureAndBindingMap();

private:
    std::shared_ptr<Device> m_device;
    Model::HlslDispatchableDesc m_desc;
    bool m_forceDisablePrecompiledShadersOnXbox;
    Microsoft::WRL::ComPtr<ID3D12ShaderReflection> m_shaderReflection;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
    std::unordered_map<std::string, BindPoint> m_bindPoints;
};