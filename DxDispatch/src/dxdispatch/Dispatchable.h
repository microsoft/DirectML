#pragma once

struct Dispatchable
{
    struct BindingSource
    {
        ID3D12Resource* resource = nullptr;
        Model::ResourceDesc* resourceDesc = nullptr;
        uint64_t elementCount = 0;
        uint64_t elementSizeInBytes = 0;
        uint64_t elementOffset = 0;
        std::optional<DXGI_FORMAT> format;
        ID3D12Resource* counterResource = nullptr;
        uint64_t counterOffsetBytes = 0;
        std::vector<int64_t> shape;
        bool deferredBinding = false;
    };

    // Maps bind points (target names) to a source resources.
    using Bindings = std::unordered_map<std::string, std::vector<BindingSource>>;

    virtual ~Dispatchable() {};

    virtual void Initialize() = 0;
    virtual void Bind(Bindings& bindings, uint32_t iteration) = 0;
    virtual void Dispatch(const Model::DispatchCommand& args, uint32_t iteration) = 0;
    virtual void Wait() = 0;
};