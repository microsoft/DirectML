#pragma once

#include <onnxruntime_cxx_api.h>

class OnnxDispatchable : public Dispatchable
{
public:
    OnnxDispatchable(
        std::shared_ptr<Device> device, 
        const Model::OnnxDispatchableDesc& desc);

    void Initialize() final;
    void Bind(const Bindings& bindings) final;
    void Dispatch(const Model::DispatchCommand& args) final;

private:
    std::shared_ptr<Device> m_device;
    const Model::OnnxDispatchableDesc& m_desc;
    std::optional<Ort::Session> m_session;
    std::optional<Ort::IoBinding> m_bindings;
};