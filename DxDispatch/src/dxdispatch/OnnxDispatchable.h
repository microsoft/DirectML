#pragma once

#include <onnxruntime_cxx_api.h>
#include "dml_provider_factory.h"
#include "CommandLineArgs.h"

class OnnxDispatchable : public Dispatchable
{
public:
    OnnxDispatchable(
        std::shared_ptr<Device> device, 
        const Model::OnnxDispatchableDesc& desc,
        const CommandLineArgs& args);

    void Initialize() final;
    void Bind(const Bindings& bindings) final;
    void Dispatch(const Model::DispatchCommand& args) final;
    void Wait() final;

private:
    std::shared_ptr<Device> m_device;
    const Model::OnnxDispatchableDesc& m_desc;
    std::optional<Ort::Env> m_environment;
    std::optional<Ort::Session> m_session;
    const OrtDmlApi* m_ortDmlApi = nullptr;
    const CommandLineArgs& m_args;

    std::optional<Ort::IoBinding> m_ioBindings;
    std::vector<Ort::Value> m_tensors;
    std::vector<Microsoft::WRL::ComPtr<IUnknown>> m_tensorWrappers;
};