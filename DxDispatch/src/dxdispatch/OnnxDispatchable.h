#pragma once

class OnnxDispatchable : public Dispatchable
{
public:
    OnnxDispatchable(
        std::string_view name, 
        std::shared_ptr<Device> device, 
        const Model::OnnxDispatchableDesc& desc, 
        const Dispatchable::Bindings& initBindings);

    void Initialize() final;
    void Bind(const Bindings& bindings) final;
    void Dispatch(const Model::DispatchCommand& args) final;

private:
    std::string m_name;
    std::shared_ptr<Device> m_device;
    const Model::OnnxDispatchableDesc& m_desc;
    Dispatchable::Bindings m_initBindings;
};