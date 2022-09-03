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

using Microsoft::WRL::ComPtr;

#if !defined(_GAMING_XBOX) && defined(WIN32)
// Needed for DX12 agility SDK. Xbox uses DX12.x from the GDK.
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = DIRECT3D_AGILITY_SDK_VERSION; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = DIRECT3D_AGILITY_SDK_PATH; }
#endif

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

    if (args.PrintHelp())
    {
        LogInfo(args.HelpText());
        return 0;
    }

    // Needs to be constructed *before* D3D12 device. A warning is printed if DXCore.dll is loaded first,
    // even though the D3D12Device isn't created yet, so we create the capture helper first to avoid this
    // message.
    auto pixCaptureHelper = std::make_unique<PixCaptureHelper>(args.GetPixCaptureType());

    std::shared_ptr<DxCoreModule> dxCoreModule = std::make_shared<DxCoreModule>();

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
            std::move(pixCaptureHelper)
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
                args.GetOnnxFreeDimensionOverrides()
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

    return 0;
}