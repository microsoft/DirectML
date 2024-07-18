#pragma once
#include "DirectMLHelpers\DmlSerializedGraphDesc.h"

class DmlSerializedGraphDispatchable : public Dispatchable
{
public:
    struct BindPoint
    {
        std::string name;
        uint32_t resourceCount;
        bool required;
    };

    struct BindPoints
    {
        std::vector<BindPoint> inputs;
        std::vector<BindPoint> outputs;
    };
    DmlSerializedGraphDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::DmlSerializedGraphDispatchableDesc& desc);

    void Initialize() final ;
    void Bind(const Bindings& bindings, uint32_t iteration) final ;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings) final;

private:

    void BuildGraph();
    BindPoints GetBindPoints(const DmlSerializedGraphDesc& serializedDesc);
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> ExtractConstantDataTypes(const DmlSerializedGraphDesc& serializedDesc);
    Bindings GenerateInitialBindingsFromGraph(const DmlSerializedGraphDesc& graphDesc);
    void CreateResourceFromConstantNode(const DmlSerializedGraphNode& node);
    std::vector<std::byte> LoadFileContents(const std::filesystem::path& filepath);
    static uint32_t GetElementSize(DML_TENSOR_DATA_TYPE dataType);


private:
    std::string m_name;
    std::shared_ptr<Device> m_device;
    BindPoints bindPoints;
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> m_constantDataTypes;
    Bindings m_bindings;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_resources;


    const Model::DmlSerializedGraphDispatchableDesc& m_desc;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_graphCompiled;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentBuffer;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
};