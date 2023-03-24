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
#include "StdSupport.h"
#include "NpyReaderWriter.h"
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

struct Timings
{
    std::vector<double> samples;

    struct Stats
    {
        double average;
        double median;
        double min;
        double max;
    };

    Stats ComputeStats(size_t warmupSampleCount)
    {
        Stats stats = {};
        if (samples.empty())
        {
            return stats;
        }

        // The first samples may be "warmup" runs that should be discarded to avoid skewing the results.
        size_t discardedSamples = std::min(warmupSampleCount, samples.size() - 1);

        double totalTime = std::accumulate(samples.begin() + discardedSamples, samples.end(), 0.0);
        stats.average = totalTime / (samples.size() - discardedSamples);

        std::sort(samples.begin(), samples.end());
        stats.median = samples[samples.size() / 2];
        stats.min = samples[0];
        stats.max = samples[samples.size() - 1];

        return stats;
    }
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
    device->ExecuteCommandListAndWait();

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
        Timer timer;

        PIXBeginEvent(m_device->GetCommandQueue(), PIX_COLOR(255, 255, 0), "Initialize dispatchables");
        for (auto& dispatchable : m_dispatchables)
        {
            try
            {
                timer.Start();
                PIXBeginEvent(PIX_COLOR(128,255,0), L"Init");
                dispatchable.second->Initialize();
                PIXEndEvent();
                timer.End();

                if (m_commandLineArgs.VerboseTimings())
                {
                    LogInfo(fmt::format("Initialize '{}': {:.4f} ms", dispatchable.first, timer.DurationInMilliseconds()));
                }
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

    Timings cpuTimings;
    Timings gpuTimings;
    Timings bindTimings;

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

    // Dispatch
    uint32_t iterationsCompleted = 0;
    PIXBeginEvent(PIX_COLOR(128, 255, 0), L"Dispatch Loop");
    try
    {
        Timer loopTimer;
        Timer iterationTimer;

        for (; iterationsCompleted < m_commandLineArgs.DispatchIterations(); iterationsCompleted++)
        {
            // Bind
            iterationTimer.Start();
            PIXBeginEvent(PIX_COLOR(128, 255, 0), L"Bind");
            try
            {
                dispatchable->Bind(bindings);
            }
            catch (const std::exception& e)
            {
                LogError(fmt::format("ERROR while binding resources: {}\n", e.what()));
                return;
            }
            PIXEndEvent();
            bindTimings.samples.push_back(iterationTimer.End().DurationInMilliseconds());

            // Dispatch
            iterationTimer.Start();
            dispatchable->Dispatch(command);
            dispatchable->Wait();
            cpuTimings.samples.push_back(iterationTimer.End().DurationInMilliseconds());

            if (m_commandLineArgs.TimeToRunInMilliseconds() &&
                loopTimer.End().DurationInMilliseconds() > m_commandLineArgs.TimeToRunInMilliseconds().value())
            {
                break;
            }
        }
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to execute dispatchable: {}", e.what()));
        return;
    }
    PIXEndEvent();

    auto cpuStats = cpuTimings.ComputeStats(1);
    auto bindStats = bindTimings.ComputeStats(1);

    // GPU timings are capped at a fixed size. If iterations > samples then the 
    // first warmup sample was overwritten (no need to discard any samples).
    gpuTimings.samples = m_device->ResolveTimingSamples();
    auto gpuStats = gpuTimings.ComputeStats(iterationsCompleted > gpuTimings.samples.size() ? 0 : 1);

    if (iterationsCompleted > 0)
    {
        if (m_commandLineArgs.VerboseTimings())
        {
            LogInfo(fmt::format("Dispatch '{}': {} iterations", 
                command.dispatchableName, iterationsCompleted
            ));

            LogInfo(fmt::format("CPU Timings: {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                cpuStats.average, cpuStats.min, cpuStats.median, cpuStats.max
            ));

            LogInfo(fmt::format("GPU Timings: {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                gpuStats.average, gpuStats.min, gpuStats.median, gpuStats.max
            ));

            LogInfo(fmt::format("Bind Timings: {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                bindStats.average, bindStats.min, bindStats.median, bindStats.max
            ));
        }
        else
        {
            LogInfo(fmt::format("Dispatch '{}': {} iterations, {:.4f} ms median (CPU), {:.4f} ms median (GPU), {:.4f} ms median (Bind)", 
                command.dispatchableName, 
                iterationsCompleted,
                cpuStats.median,
                gpuStats.median,
                bindStats.median
            ));
        }
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

void Executor::operator()(const Model::WriteFileCommand& command)
{
    PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "WriteFile: %s", command.resourceName.c_str());

    try
    {
        auto resource = m_resources[command.resourceName];
        auto fileData = m_device->Download(resource.Get());
        auto& resourceDesc = m_model.GetResource(command.resourceName);
        auto& bufferDesc = std::get<Model::BufferDesc>(resourceDesc.value);

        std::ofstream file(command.targetPath.c_str(), std::ifstream::trunc | std::ifstream::binary);
        if (!file.is_open())
        {
            throw std::ios::failure("Could not open file");
        }

        // If NumPy array, serialize data into .npy file.
        if (IsNpyFilenameExtension(command.targetPath))
        {
            // If no dimensions were given, then treat as a 1D array.
            std::vector<uint32_t> dimensions(command.dimensions);
            if (dimensions.empty())
            {
                uint32_t elementCount = bufferDesc.sizeInBytes / Device::GetSizeInBytes(bufferDesc.initialValuesDataType);
                dimensions.push_back(elementCount);
            }

            std::vector<std::byte> npyFileData;
            WriteNpy(fileData, bufferDesc.initialValuesDataType, dimensions, /*out*/ npyFileData);
            std::swap(fileData, npyFileData);
        }

        file.write(reinterpret_cast<const char*>(fileData.data()), fileData.size());
        LogInfo(fmt::format("Resource '{}' written to '{}'", command.resourceName, command.targetPath));
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to write resource to file '{}': {}", command.targetPath, e.what()));
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