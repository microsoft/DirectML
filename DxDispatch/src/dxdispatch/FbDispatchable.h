
#pragma once

class BindingManager;

class FbDispatchable : public Dispatchable
{
public:
    FbDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::FbDispatchableDesc& desc, 
        const Model::Bindings& initBindings
    );
    //~FbDispatchable();
    void Initialize() final ;
    void Bind(const Bindings& bindings, uint32_t iteration) final ;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings) final;

    //add deferred bindimgs to above
    //void Wait() final;

private:
    std::string m_name;
    std::shared_ptr<Device> m_device;
    const Model::FbDispatchableDesc& m_desc;
    std::unique_ptr<BindingManager> m_bindingManager;
    const Model::Bindings& m_initBindings;
    Microsoft::WRL::ComPtr<IDMLOperator> m_operator;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_operatorCompiled;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentBuffer;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
};
