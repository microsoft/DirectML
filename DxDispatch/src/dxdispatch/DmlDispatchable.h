#pragma once
#include "DirectMLHelpers\DmlSerializedGraphDesc.h"


class DmlDispatchable : public Dispatchable
{
public:

    DmlDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::DmlDispatchableDesc& desc, 
        const Dispatchable::Bindings& initBindings,
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
    Dispatchable::Bindings m_initBindings;
    bool m_isSerializedGraph = false;
    Microsoft::WRL::ComPtr<IDMLOperator> m_operator;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_compiledOperator;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentBuffer;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
    Microsoft::WRL::ComPtr<IDxDispatchLogger> m_logger;
    Model::DmlDispatchableDesc::BindPoints m_bindPoints;
    std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> m_constantDataTypes;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_resources;


    void BuildAndCompileGraph();
    //BindPoints GetBindPoints() const;
    //void CreateResourceFromConstantNode(const DmlSerializedGraphNode& node);

    
    // Model::DmlDispatchableDesc::BindPoints GetSerializedBindPoints(const DmlSerializedGraphDesc& serializedDesc);
    // std::unordered_map<std::string, DML_TENSOR_DATA_TYPE> ExtractConstantDataTypes(const DmlSerializedGraphDesc& serializedDesc);
    // Bindings GenerateInitialBindingsFromGraph(const DmlSerializedGraphDesc& graphDesc);
    // std::vector<std::byte> LoadFileContents(const std::filesystem::path& filepath);
    // static uint32_t GetElementSize(DML_TENSOR_DATA_TYPE dataType);



};