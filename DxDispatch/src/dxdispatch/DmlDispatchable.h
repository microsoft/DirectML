#pragma once

class DmlDispatchable : public Dispatchable
{
public:
    DmlDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::DmlDispatchableDesc& desc, 
        const Dispatchable::Bindings& initBindings);

    void Initialize() final;
    void Bind(const Bindings& bindings, uint32_t iteration) final;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration, uint32_t repeat) final;
    void Wait() final;

private:
    std::string m_name;
    std::shared_ptr<Device> m_device;
    const Model::DmlDispatchableDesc& m_desc;
    Dispatchable::Bindings m_initBindings;
    Microsoft::WRL::ComPtr<IDMLOperator> m_operator;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_operatorCompiled;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentBuffer;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
};