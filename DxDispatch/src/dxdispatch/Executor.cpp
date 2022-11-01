#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "BucketAllocator.h"
#include "Model.h"
#include "Dispatchable.h"
#include "DmlDispatchable.h"
#ifndef DXCOMPILER_NONE
#include "HlslDispatchable.h"
#endif
#ifndef ONNXRUNTIME_NONE
#include "OnnxDispatchable.h"
#endif
#include "CommandLineArgs.h"
#include "Executor.h"
#include <half.hpp>

using Microsoft::WRL::ComPtr;

struct Timer
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    Timer() { start = std::chrono::steady_clock::now(); }
    Timer& Start() { start = std::chrono::steady_clock::now(); return *this; }
    Timer& End() { end = std::chrono::steady_clock::now(); return *this; }
    double DurationInMilliseconds() { return std::chrono::duration<double>(end - start).count() * 1000; }
};

Executor::Executor(Model& model, std::shared_ptr<Device> device, const CommandLineArgs& args) : m_model(model), m_device(device), m_commandLineArgs(args)
{
    // Initialize buffer resources.
    {
        PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255, 255, 0), "Initialize resources");
        for (auto& desc : model.GetResourceDescs())
        {
            // Only buffers are supported right now.
            assert(std::holds_alternative<Model::BufferDesc>(desc.value));
            auto& bufferDesc = std::get<Model::BufferDesc>(desc.value);
            auto wName = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(desc.name);
            m_resources[desc.name] = std::move(device->Upload(bufferDesc.sizeInBytes, bufferDesc.initialValues, wName));
        }
    }
    device->DispatchAndWait();

    // Create dispatchables.
    for (auto& desc : model.GetDispatchableDescs())
    {
        try
        {
            if (std::holds_alternative<Model::HlslDispatchableDesc>(desc.value))
            {
#ifdef DXCOMPILER_NONE
                throw std::invalid_argument("HLSL dispatchables require DXCompiler");
#else
                m_dispatchables[desc.name] = std::make_unique<HlslDispatchable>(device, std::get<Model::HlslDispatchableDesc>(desc.value), args);
#endif
            }
            else if (std::holds_alternative<Model::OnnxDispatchableDesc>(desc.value))
            {
#ifdef ONNXRUNTIME_NONE
                throw std::invalid_argument("ONNX dispatchables require ONNX Runtime");
#else
                m_dispatchables[desc.name] = std::make_unique<OnnxDispatchable>(device, std::get<Model::OnnxDispatchableDesc>(desc.value), args);
#endif
            }
            else
            {
                auto& dmlDispatchableDesc = std::get<Model::DmlDispatchableDesc>(desc.value);

                Dispatchable::Bindings initBindings;
                try
                {
                    initBindings = ResolveBindings(dmlDispatchableDesc.initBindings);
                }
                catch (const std::exception& e)
                {
                    LogError(fmt::format("Failed to resolve bindings: {}", e.what()));
                    return;
                }

                m_dispatchables[desc.name] = std::make_unique<DmlDispatchable>(desc.name, device, dmlDispatchableDesc, initBindings);
            }
        }
        catch(const std::exception& e)
        {
            throw std::invalid_argument(fmt::format("ERROR creating dispatchable '{}': {}", desc.name, e.what()));
        }
    }

    // Compile/initialize dispatchables.
    {
        PIXBeginEvent(m_device->GetCommandQueue(), PIX_COLOR(255, 255, 0), "Initialize dispatchables");
        for (auto& dispatchable : m_dispatchables)
        {
            try
            {
                PIXBeginEvent(PIX_COLOR(128,255,0), L"Init");
                dispatchable.second->Initialize();
                PIXEndEvent();
            }
            catch (const std::exception& e)
            {
                throw std::invalid_argument(fmt::format("ERROR while initializing '{}': {}", dispatchable.first, e.what()));
            }
        }
        PIXEndEvent(m_device->GetCommandQueue());
    }
}

void Executor::Run()
{
    for (auto& command : m_model.GetCommands())
    {
        std::visit(*this, command);
    }
}

void Executor::operator()(const Model::DispatchCommand& command)
{
    auto& dispatchable = m_dispatchables[command.dispatchableName];

    // DML and HLSL dispatchables write into the DxDispatch device command list; ONNX dispatchables use
    // a command list owned by the DML execution provider in onnxruntime.dll.
    const bool dispatchableUsesDeviceCommandList = dispatchable->RecordsDispatchIntoCommandList();

    Timer timer;
    std::vector<double> dispatchDurationsCPU;
    double totalDuration = 0.0;

    Dispatchable::Bindings bindings;
    try
    {
        bindings = ResolveBindings(command.bindings);
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to resolve bindings: {}", e.what()));
        return;
    }

    const uint32_t timestampCount = m_device->timestampCount;

    auto timestampReadbackBuffer = m_device->CreateReadbackBuffer(sizeof(uint64_t) * timestampCount);

    std::vector<uint64_t> timestamps;

    // Dispatch
    PIXBeginEvent(PIX_COLOR(128, 255, 0), L"Dispatch Loop");
    try
    {
        THROW_IF_FAILED(m_device->GetPixCaptureHelper().BeginCapturableWork(command.dispatchableName));

        for (uint32_t iteration = 0; iteration < m_commandLineArgs.DispatchIterations(); iteration++)
        {
            timer.Start();

            try
            {
                dispatchable->Bind(bindings);
            }
            catch (const std::exception& e)
            {
                LogError(fmt::format("ERROR while binding resources: {}\n", e.what()));
                return;
            }

            PIXBeginEvent(m_device->GetCommandQueue(), PIX_COLOR(255, 255, 0), "Dispatch '%s'", command.dispatchableName.c_str());

            uint32_t timestampIndex = (iteration * 2) % timestampCount;

            if (dispatchableUsesDeviceCommandList)
            {
                m_device->GetCommandList()->EndQuery(m_device->GetTimestampHeap(), D3D12_QUERY_TYPE_TIMESTAMP, timestampIndex);
                dispatchable->Dispatch(command);
                m_device->GetCommandList()->EndQuery(m_device->GetTimestampHeap(), D3D12_QUERY_TYPE_TIMESTAMP, timestampIndex + 1);
                m_device->DispatchAndWait();
            }
            else
            {
#ifndef ONNXRUNTIME_NONE
                OnnxDispatchable* onnx = dynamic_cast<OnnxDispatchable*>(dispatchable.get());

                if (onnx)
                {
                    m_device->GetCommandList()->EndQuery(m_device->GetTimestampHeap(), D3D12_QUERY_TYPE_TIMESTAMP, timestampIndex);
                    m_device->DispatchDontWait();
                    onnx->Dispatch(command);

                    m_device->GetCommandList()->EndQuery(m_device->GetTimestampHeap(), D3D12_QUERY_TYPE_TIMESTAMP, timestampIndex + 1);
                    m_device->DispatchDontWait();
                    onnx->Wait();
                }
#endif
            }

            PIXEndEvent(m_device->GetCommandQueue());

            double duration = timer.End().DurationInMilliseconds();
            dispatchDurationsCPU.push_back(duration);

            totalDuration += duration;
            if (m_commandLineArgs.TimeToRunInMilliseconds() &&
                totalDuration > m_commandLineArgs.TimeToRunInMilliseconds().value())
            {
                break;
            }
        }

        THROW_IF_FAILED(m_device->GetPixCaptureHelper().EndCapturableWork());
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to execute dispatchable: {}", e.what()));
        return;
    }
    PIXEndEvent();

    m_device->GetCommandList()->ResolveQueryData(m_device->GetTimestampHeap(), D3D12_QUERY_TYPE_TIMESTAMP, 0, timestampCount, timestampReadbackBuffer.Get(), 0);
    m_device->DispatchAndWait();

    void* pData = nullptr;
    D3D12_RANGE readRange = { 0, sizeof(uint64_t) * timestampCount };
    timestampReadbackBuffer->Map(0, &readRange, &pData);

    const uint64_t* pTimestamps = reinterpret_cast<uint64_t*>(pData);
    timestamps.insert(timestamps.end(), pTimestamps, pTimestamps + timestampCount);


    uint32_t iterations = dispatchDurationsCPU.size();
    // Skip the first dispatch (assuming multiple dispatches) since it warms up the pipeline.
    int skipped = (iterations > 1) ? 1 : 0;
    double totalTimeCPU = std::accumulate(dispatchDurationsCPU.begin() + skipped, dispatchDurationsCPU.end(), 0.0);
    double avgTimeCPU = totalTimeCPU / (dispatchDurationsCPU.size() - skipped);

    std::sort(dispatchDurationsCPU.begin(), dispatchDurationsCPU.end());
    double medianTimeCPU = dispatchDurationsCPU[iterations / 2];
    double minTimeCPU = dispatchDurationsCPU[0];
    double maxTimeCPU = dispatchDurationsCPU[iterations - 1];


    uint64_t frequency;
    m_device->GetCommandQueue()->GetTimestampFrequency(&frequency);

    uint32_t samples = std::min(iterations, (uint32_t)timestamps.size() / 2);
    std::vector<double> dispatchDurationsGPU(samples);

    for (uint32_t i = 0; i < samples; ++i) {
        uint64_t timestampDelta = (timestamps[2 * i + 1] - timestamps[2 * i]) * 1000;

        dispatchDurationsGPU[i] = double(timestampDelta) / frequency;
    }

    if (iterations > timestamps.size() / 2) {
        skipped = 0;
    }

    double totalTimeGPU = std::accumulate(dispatchDurationsGPU.begin() + skipped, dispatchDurationsGPU.end(), 0.0);
    double avgTimeGPU = totalTimeGPU / (dispatchDurationsGPU.size() - skipped);

    std::sort(dispatchDurationsGPU.begin(), dispatchDurationsGPU.end());
    double medianTimeGPU = dispatchDurationsGPU[samples / 2];
    double minTimeGPU = dispatchDurationsGPU[0];
    double maxTimeGPU = dispatchDurationsGPU[samples - 1];

    
    if (m_commandLineArgs.VerboseTimings()) {
        LogInfo(fmt::format("Dispatch '{}': {} iterations\nCPU Timings: {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max",
            command.dispatchableName, iterations, avgTimeCPU, minTimeCPU, medianTimeCPU, maxTimeCPU));
        LogInfo(fmt::format("GPU Timings: {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max\n",
            avgTimeGPU, minTimeGPU, medianTimeGPU, maxTimeGPU));
    }
    else if (iterations == 1) {
        LogInfo(fmt::format("Dispatch '{}': {} iteration, {:.4f} ms (CPU), {:.4f} ms (GPU)",
            command.dispatchableName, iterations, medianTimeCPU, medianTimeGPU));
    }
    else {
        LogInfo(fmt::format("Dispatch '{}': {} iterations, {:.4f} ms median (CPU), {:.4f} ms median (GPU)",
            command.dispatchableName, iterations, medianTimeCPU, medianTimeGPU));
    }
}

template <typename T>
struct BufferDataView
{
    gsl::span<const std::byte> byteValues;
    const Model::BufferDesc& desc;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const BufferDataView<T>& view)
{
    uint32_t elementCount = view.desc.initialValues.size() / Device::GetSizeInBytes(view.desc.initialValuesDataType);
    auto values = reinterpret_cast<const T*>(view.byteValues.data());
    for (uint32_t elementIndex = 0; elementIndex < elementCount; elementIndex++)
    {
        os << values[elementIndex];
        if (elementIndex < elementCount - 1)
        {
            os << ", ";
        }
    }
    return os;
}

std::string ToString(gsl::span<const std::byte> byteValues, const Model::BufferDesc& desc)
{
    std::stringstream ss;
    switch (desc.initialValuesDataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: ss << BufferDataView<half_float::half>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_FLOAT32: ss << BufferDataView<float>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_FLOAT64: ss << BufferDataView<double>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT8: ss << BufferDataView<uint8_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT16: ss << BufferDataView<uint16_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT32: ss << BufferDataView<uint32_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT64: ss << BufferDataView<uint64_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT8: ss << BufferDataView<int8_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT16: ss << BufferDataView<int16_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT32: ss << BufferDataView<int32_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT64: ss << BufferDataView<int64_t>{byteValues, desc}; break;
    default: throw std::invalid_argument("Unexpected DML_TENSOR_DATA_TYPE");
    }
    return ss.str();
}

void Executor::operator()(const Model::PrintCommand& command)
{
    PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "Print: %s", command.resourceName.c_str());

    try
    {
        auto resource = m_resources[command.resourceName];
        auto outputValues = m_device->Download(resource.Get());
        auto& resourceDesc = m_model.GetResource(command.resourceName);
        auto& bufferDesc = std::get<Model::BufferDesc>(resourceDesc.value);
        LogInfo(fmt::format("Resource '{}': {}", command.resourceName, ToString(outputValues, bufferDesc)));
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to print resource: {}", e.what()));
    }
}

Dispatchable::Bindings Executor::ResolveBindings(const Model::Bindings& modelBindings)
{
    Dispatchable::Bindings bindings;

    for (auto& modelBinding : modelBindings)
    {
        std::vector<Dispatchable::BindingSource> sourceResources;

        for (auto& modelSource : modelBinding.second)
        {
            // Validated when the model is constructed.
            assert(m_resources.find(modelSource.name) != m_resources.end());
            auto& resourceDesc = m_model.GetResource(modelSource.name);

            Dispatchable::BindingSource source = {};
            source.elementSizeInBytes = modelSource.elementSizeInBytes;
            source.elementCount = modelSource.elementCount;
            source.elementOffset = modelSource.elementOffset;
            source.format = modelSource.format;
            source.resource = m_resources[modelSource.name].Get();
            source.resourceDesc = &resourceDesc;

            if (std::holds_alternative<Model::BufferDesc>(resourceDesc.value))
            {
                auto& modelBufferDesc = std::get<Model::BufferDesc>(resourceDesc.value);
                if (source.elementSizeInBytes == 0 && modelBufferDesc.initialValuesDataType != DML_TENSOR_DATA_TYPE_UNKNOWN)
                {
                    // If the binding doesn't specify, assume the data type size used to initialize the buffer.
                    source.elementSizeInBytes = Device::GetSizeInBytes(modelBufferDesc.initialValuesDataType);
                }

                if (source.elementCount == 0 && source.elementSizeInBytes != 0)
                {
                    // If the binding doesn't specify, assume the number of elements used to initialize the buffer.
                    source.elementCount = modelBufferDesc.initialValues.size() / source.elementSizeInBytes;
                }
            }

            if (modelSource.counterName)
            {
                // Validated when the model is constructed.
                assert(m_resources.find(*modelSource.counterName) != m_resources.end());
                source.counterResource = m_resources[*modelSource.counterName].Get();
                source.counterOffsetBytes = modelSource.counterOffsetBytes;
            }

            sourceResources.push_back(source);
        }

        bindings[modelBinding.first] = sourceResources;
    }

    return bindings;
}