#pragma once

struct Dispatchable
{
    struct BindingSource
    {
        ID3D12Resource* resource;
        const Model::ResourceDesc* resourceDesc;
        uint64_t elementCount;
        uint64_t elementSizeInBytes;
        uint64_t elementOffset;
        std::optional<DXGI_FORMAT> format;
        ID3D12Resource* counterResource;
        uint64_t counterOffsetBytes;
        std::vector<int64_t> shape;
    };

    struct DeferredBinding
    {
        std::string name;
        uint64_t elementCount;
        uint64_t elementSizeInBytes;
        DML_TENSOR_DATA_TYPE type;
        std::vector<int64_t> shape;
        std::vector<std::byte> cpuValues;
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    };

    using DeferredBindings = std::unordered_map<std::string, DeferredBinding>;

    // Maps bind points (target names) to a source resources.
    using Bindings = std::unordered_map<std::string, std::vector<BindingSource>>;

    virtual ~Dispatchable() = default;

    virtual void Initialize() = 0;
    virtual void Bind(const Bindings& bindings, uint32_t iteration) = 0;
    virtual void Dispatch(const Model::DispatchCommand& args, uint32_t iteration) = 0;

    virtual bool SupportsDeferredBinding(){ return false; }

    virtual void Wait()
    {
        // Only one Wait needs to be implemented, depending on SupportsDeferredBinding()
        THROW_HR(E_NOTIMPL);
    }

    virtual void Wait(DeferredBindings& deferredBinings)
    {
        // Only one Wait needs to be implemented, depending on SupportsDeferredBinding()
        THROW_HR(E_NOTIMPL);
    }

};