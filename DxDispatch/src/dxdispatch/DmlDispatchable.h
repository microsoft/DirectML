#pragma once
#include "DirectMLHelpers\DmlSerializedGraphDesc.h"


class DmlDispatchable : public Dispatchable
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
    DmlDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::DmlDispatchableDesc& desc, 
        const Dispatchable::Bindings& dmlInitBindings,
        IDxDispatchLogger* logger);

    DmlDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::DmlSerializedGraphDispatchableDesc& desc);

    void Initialize() final;
    void Bind(const Bindings& bindings, uint32_t iteration) final;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBinings) final;

private:
    std::string m_name;
    std::shared_ptr<Device> m_device;
    std::variant<Model::DmlDispatchableDesc, Model::DmlSerializedGraphDispatchableDesc> m_desc;
    Dispatchable::Bindings m_dmlInitBindings;
    Dispatchable::Bindings m_serializedGraphBindings;
    bool m_isSerializedGraph = false;
    Microsoft::WRL::ComPtr<IDMLOperator> m_operator;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_compiledOperator;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentBuffer;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;

    BindPoints m_bindPoints;
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> m_constantDataTypes;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_resources;
    
    void CreateOperator();
    void CompileOperator();
    void BuildGraph();
    BindPoints GetBindPoints() const;
    
    BindPoints GetSerializedBindPoints(const DmlSerializedGraphDesc& serializedDesc);
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> ExtractConstantDataTypes(const DmlSerializedGraphDesc& serializedDesc);
    Bindings GenerateInitialBindingsFromGraph(const DmlSerializedGraphDesc& graphDesc);
    void CreateResourceFromConstantNode(const DmlSerializedGraphNode& node);
    std::vector<std::byte> LoadFileContents(const std::filesystem::path& filepath);
    static uint32_t GetElementSize(DML_TENSOR_DATA_TYPE dataType);



    Microsoft::WRL::ComPtr<IDxDispatchLogger> m_logger;
};