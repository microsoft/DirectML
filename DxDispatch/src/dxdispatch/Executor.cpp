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
#include "OnnxDispatchable.h"
#include "CommandLineArgs.h"
#include "Executor.h"
#include <half.hpp>

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
        // PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "Initialize resources");
        for (auto& desc : model.GetResourceDescs())
        {
            // Only buffers are supported right now.
            assert(std::holds_alternative<Model::BufferDesc>(desc.value));
            auto& bufferDesc = std::get<Model::BufferDesc>(desc.value);
            auto wName = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(desc.name);
            m_resources[desc.name] = std::move(device->Upload(bufferDesc.sizeInBytes, bufferDesc.initialValues, wName));
        }
        device->DispatchAndWait();
        device->PrintDebugLayerMessages();
    }

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
                m_dispatchables[desc.name] = std::make_unique<OnnxDispatchable>(device, std::get<Model::OnnxDispatchableDesc>(desc.value));
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
                    m_device->PrintDebugLayerMessages();
                    return;
                }

                m_dispatchables[desc.name] = std::make_unique<DmlDispatchable>(desc.name, device, dmlDispatchableDesc, initBindings);
            }
        }
        catch(const std::exception& e)
        {
            device->PrintDebugLayerMessages();
            throw std::invalid_argument(fmt::format("ERROR creating dispatchable '{}': {}", desc.name, e.what()));
        }
    }

    // Compile/initialize dispatchables.
    {
        PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "Initialize dispatchables");
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
                m_device->PrintDebugLayerMessages();
                throw std::invalid_argument(fmt::format("ERROR while initializing '{}': {}", dispatchable.first, e.what()));
            }
        }
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
    uint32_t executeIterations = m_commandLineArgs.BenchmarkingEnabled() ? 4 : 1;
    std::vector<double> executeAverages;

    for (uint32_t executeIteration = 0; executeIteration < executeIterations; executeIteration++)
    {
        PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "Dispatch '%s'", command.dispatchableName.c_str());

        auto& dispatchable = m_dispatchables[command.dispatchableName];
        
        Dispatchable::Bindings bindings;
        try
        {
            bindings = ResolveBindings(command.bindings);
        }
        catch (const std::exception& e)
        {
            LogError(fmt::format("Failed to resolve bindings: {}", e.what()));
            m_device->PrintDebugLayerMessages();
            return;
        }

        PIXBeginEvent(PIX_COLOR(128,255,0), L"Dispatch");

        Timer timer;
        std::vector<double> dispatchDurations;
        uint32_t dispatchIterations = m_commandLineArgs.BenchmarkingEnabled() ? 
            m_commandLineArgs.BenchmarkingDispatchRepeat() : 
            1;

        try
        {
            dispatchable->Bind(bindings);
        }
        catch (const std::exception& e)
        {
            LogError(fmt::format("ERROR while binding resources: {}\n", e.what()));
            m_device->PrintDebugLayerMessages();
            return;
        }

        try
        {
            const bool recordsDispatchIntoCommandList = dispatchable->RecordsDispatchIntoCommandList();

            timer.Start();
            for (uint32_t iteration = 0; iteration < dispatchIterations; iteration++)
            {
                dispatchable->Dispatch(command);
                if (recordsDispatchIntoCommandList && iteration != dispatchIterations - 1)
                {
                    auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
                    m_device->GetCommandList()->ResourceBarrier(1, &uavBarrier);
                }
            }
            if (recordsDispatchIntoCommandList)
            {
                m_device->DispatchAndWait();
            }
            dispatchDurations.push_back(timer.End().DurationInMilliseconds() / dispatchIterations);
        }
        catch (const std::exception& e)
        {
            LogError(fmt::format("Failed to execute dispatchable: {}", e.what()));
            m_device->PrintDebugLayerMessages();
            return;
        }

        PIXEndEvent();

        if (m_commandLineArgs.BenchmarkingEnabled())
        {
            double avgDispatch = 0;
            for (auto& dur : dispatchDurations) 
            { 
                avgDispatch += dur; 
            } 
            avgDispatch /= dispatchDurations.size();

            executeAverages.push_back(avgDispatch);
        }
    }

    if (m_commandLineArgs.BenchmarkingEnabled())
    {
        // Assuming multiple times, skip the first since it warms up the pipeline.
        int skipped = executeAverages.empty() ? 0 : 1;
        double avgTime = std::accumulate(
            executeAverages.begin() + skipped, 
            executeAverages.end(), 
            0.0) / (executeAverages.size() - skipped);

        LogInfo(fmt::format("Dispatch '{}': {} ms", command.dispatchableName, avgTime));
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
        m_device->PrintDebugLayerMessages();
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