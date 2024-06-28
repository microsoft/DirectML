#pragma once

class DmlSerializedGraphDispatchable : public Dispatchable
{
public:
    DmlSerializedGraphDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::DmlSerializedGraphDispatchableDesc& desc, 
        const Dispatchable::Bindings& initBindings);

    //~DmlSerializedGraphDispatchable();
    void Initialize() final ;
    void Bind(const Bindings& bindings, uint32_t iteration) final ;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings) final;
    //add deferred bindimgs to above
    //void Wait() final;

private:
    void BuildGraph();
private:
    std::string m_name;
    std::shared_ptr<Device> m_device;
    const Model::DmlSerializedGraphDispatchableDesc& m_desc;
    const Dispatchable::Bindings& m_initBindings;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_graphCompiled;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentBuffer;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
};
