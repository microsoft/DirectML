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
        const CommandLineArgs& args,
        IDxDispatchLogger* logger);

    void Initialize() final;
    void Bind(const Bindings& bindings, uint32_t iteration) final;
    void Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& defferedBindings) final;

private:
    std::shared_ptr<Device> m_device;
    const Model::OnnxDispatchableDesc& m_desc;
    std::optional<Ort::Env> m_environment;
    std::optional<Ort::Session> m_session;
    const OrtDmlApi* m_ortDmlApi = nullptr;
    const CommandLineArgs& m_args;

    struct TensorBinding
    {
        std::string name;
        std::string resourceType;
        std::vector<int64_t> shape;
        ONNXTensorElementDataType dataType;
        bool isInput;
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        std::vector<std::byte> cpuProviderBuffer;
        Microsoft::WRL::ComPtr<IUnknown> wrapper;
        std::optional<Ort::Value> ortValue;
    };

    // ONNX dispatchables allow resources & bindings to be lazily instantiated. The merged bindings 
    // are the union of JSON bindings and bindings to lazily-allocated resources from the first Bind().
    std::vector<TensorBinding> m_mergedBindings;

    std::optional<Ort::IoBinding> m_ioBindings;
    Microsoft::WRL::ComPtr<IDxDispatchLogger> m_logger;
};