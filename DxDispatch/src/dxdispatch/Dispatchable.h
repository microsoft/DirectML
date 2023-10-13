#pragma once

struct Dispatchable
{
    struct BindingSource
    {
        ID3D12Resource* resource;
        Model::ResourceDesc* resourceDesc;
        uint64_t elementCount;
        uint64_t elementSizeInBytes;
        uint64_t elementOffset;
        std::optional<DXGI_FORMAT> format;
        ID3D12Resource* counterResource;
        uint64_t counterOffsetBytes;
        std::vector<int64_t> shape;
        bool useDeferredBinding;
    };

    // Maps bind points (target names) to a source resources.
    using Bindings = std::unordered_map<std::string, std::vector<BindingSource>>;

    virtual ~Dispatchable() {};

    virtual void Initialize() = 0;
    virtual void Bind(Bindings& bindings, uint32_t iteration) = 0;
    virtual void Dispatch(const Model::DispatchCommand& args, uint32_t iteration) = 0;
    virtual void Wait() = 0;
};