#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "JsonParsers.h"
#ifndef ONNXRUNTIME_NONE
#include "OnnxParsers.h"
#endif
#include "Executor.h"
#include "CommandLineArgs.h"
#include "ModuleInfo.h"

using Microsoft::WRL::ComPtr;

int main(int argc, char** argv)
{
    CommandLineArgs args;
    try
    {
        args = CommandLineArgs(argc, argv);
    }
    catch (std::exception& e)
    {
        LogError(fmt::format("Failed to parse command-line arguments: {}", e.what()));
        return 1;
    }

    // Needs to be constructed *before* D3D12 device. A warning is printed if DXCore.dll is loaded first,
    // even though the D3D12Device isn't created yet, so we create the capture helper first to avoid this
    // message.
    auto pixCaptureHelper = std::make_shared<PixCaptureHelper>(args.GetPixCaptureType());
    auto dxCoreModule = std::make_shared<DxCoreModule>();
    auto d3dModule = std::make_shared<D3d12Module>();
    auto dmlModule = std::make_shared<DmlModule>();

    if (args.PrintHelp())
    {
        LogInfo(args.HelpText());
        return 0;
    }

    if (args.ShowDependencies())
    {
#if defined(_WIN32) && !defined(_GAMING_XBOX)
        // D3D12.dll lazily loads D3D12Core.dll. Calling any exported function forces D3D12Core.dll to load
        // so its version can be printed, and GetDebugInterface is inexpensive.
        Microsoft::WRL::ComPtr<ID3D12Debug> debug;
        d3dModule->GetDebugInterface(IID_PPV_ARGS(&debug));
#endif

        PrintDependencies();
        return 0;
    }

    if (args.ShowAdapters())
    {
        for (auto& adapter : Adapter::GetAll(dxCoreModule))
        {
            LogInfo(adapter.GetDetailedDescription() + "\n");
        }

        return 0;
    }

    std::shared_ptr<Device> device;
    try
    {
        Adapter adapter = Adapter::Select(dxCoreModule, args.AdapterSubstring());
        device = std::make_shared<Device>(
            adapter.GetAdapter(), 
            args.DebugLayersEnabled(), 
            args.CommandListType(),
            pixCaptureHelper,
            d3dModule,
            dmlModule
        );
        LogInfo(fmt::format("Running on '{}'", adapter.GetDescription()));
    }
    catch (std::exception& e)
    {
        LogError(fmt::format("Failed to create a device: {}", e.what()));
        return 1;
    }

    Model model;
    try
    {
        if (args.ModelPath().extension() == ".json")
        {
            model = JsonParsers::ParseModel(args.ModelPath());
        }
        else if (args.ModelPath().extension() == ".onnx")
        {
#ifdef ONNXRUNTIME_NONE
            throw std::invalid_argument("ONNX dispatchables require ONNX Runtime");
#else
            model = OnnxParsers::ParseModel(
                device->DML(), 
                device->GetCommandQueue(),
                args.ModelPath(), 
                args.GetOnnxFreeDimensionNameOverrides(),
                args.GetOnnxFreeDimensionDenotationOverrides()
            );
#endif
        }
        else
        {
            throw std::invalid_argument("Expected a .json or .onnx file");
        }
    }
    catch (std::exception& e)
    {
        LogError(fmt::format("Failed to parse the model: {}", e.what()));
        return 1;
    }

    try
    {
        Executor executor{model, device, args};
        executor.Run();
    }
    catch (std::exception& e)
    {
        LogError(fmt::format("Failed to execute the model: {}", e.what()));
        return 1;
    }

    // Ensure remaining D3D references are released before the D3D module is released.
    pixCaptureHelper = nullptr;

    return 0;
}