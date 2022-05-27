#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"

#include <onnxruntime_cxx_api.h>
#include "cpu_provider_factory.h"
#include "dml_provider_factory.h"

using Microsoft::WRL::ComPtr;

#define THROW_IF_NOT_OK(status) {auto localStatus = (status); if (localStatus) throw E_FAIL;}

OnnxDispatchable::OnnxDispatchable(
    std::shared_ptr<Device> device, 
    const Model::OnnxDispatchableDesc& desc
    ) : m_device(device), m_desc(desc)
{
}

void OnnxDispatchable::Initialize()
{
    // TODO: support AddFreeDimensionOverrideByName overrides in json model

    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    const OrtDmlApi* ortDmlApi;
    THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DxDispatch"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.
    
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // Note ORT_ENABLE_BASIC is useful for debugging.
    //ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1);

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProviderEx_DML(sessionOptions, m_device->DML(), m_device->GetCommandQueue()));

    m_session = Ort::Session(ortEnvironment, m_desc.sourcePath.wstring().c_str(), sessionOptions);
    m_bindings = Ort::IoBinding::IoBinding(*m_session);
}

void OnnxDispatchable::Bind(const Bindings& bindings)
{
    m_bindings->ClearBoundInputs();
    m_bindings->ClearBoundOutputs();

    Ort::MemoryInfo memoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    Ort::Allocator deviceAllocator(*m_session, memoryInformation);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    std::vector<std::vector<std::byte>> inputTensorValues; // Preserve the values since the CPU tensor just lightly wraps them.
    std::vector<std::vector<std::byte>> outputTensorValues;
    std::vector<ComPtr<IUnknown>> inputTensorWrappers; // Preserve lifetime of tensors in the Ort::Value, which doesn't seem to hold a reference.
    std::vector<ComPtr<IUnknown>> outputTensorWrappers;

    size_t const inputCount = m_session->GetInputCount();
    size_t const outputCount = m_session->GetOutputCount();

    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    const OrtDmlApi* ortDmlApi;
    THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

    // Loop though inputs and outputs.
    for (int bindingPass = 0; bindingPass < 2; ++bindingPass)
    {
        bool const isInputTensor = (bindingPass == 0);
        size_t const tensorCount = isInputTensor ? inputCount : outputCount;

        for (size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
        {
            OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION

            //BindValues(
            //    tensorIndex,
            //    isInputTensor,
            //    session,
            //    *ortDmlApi,
            //    ioBinding,
            //    memoryInformation,
            //    deviceAllocator,
            //    d3d12Device,
            //    commandQueue,
            //    commandAllocator,
            //    commandList,
            //    isInputTensor ? inputTensors : outputTensors,
            //    isInputTensor ? inputTensorValues : outputTensorValues,
            //    isInputTensor ? inputTensorWrappers : outputTensorWrappers
            //);
        }
    }
}

void OnnxDispatchable::Dispatch(const Model::DispatchCommand& args)
{
    //Ort::AllocatorWithDefaultOptions ortAllocator;
    //char* inputName = session.GetInputName(0, ortAllocator);
    //const std::array<const char*, 1> inputNames = { inputName };
    //char* outputName = session.GetOutputName(0, ortAllocator);
    //const std::array<const char*, 1> outputNames = { outputName };

    //Ort::RunOptions runOptions;
    //m_session->Run(
    //    runOptions,
    //    inputNames.data(),
    //    &inputTensor,
    //    1,
    //    outputNames.data(),
    //    &outputTensor,
    //    1);
}