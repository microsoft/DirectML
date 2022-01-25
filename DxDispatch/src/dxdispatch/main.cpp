#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "JsonParsers.h"
#include "Executor.h"
#include "CommandLineArgs.h"

using Microsoft::WRL::ComPtr;

// Needed for DX12 agility SDK.
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = DIRECT3D_AGILITY_SDK_VERSION; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = DIRECT3D_AGILITY_SDK_PATH; }

int main(int argc, char** argv)
{
    CommandLineArgs args;
    try
    {
        args = CommandLineArgs(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "ERROR: failed to parse command-line arguments." << std::endl << e.what() << std::endl;
        return 1;
    }

    if (args.PrintHelp())
    {
        std::cout << args.HelpText() << std::endl;
        return 0;
    }

    if (args.ShowAdapters())
    {
        for (auto& adapter : Adapter::GetAll())
        {
            std::cout << adapter.GetDetailedDescription() << "\n\n";
        }

        return 0;
    }

    Model model;
    try
    {
        model = JsonParsers::ParseModel(args.ModelPath());
    }
    catch (std::exception& e)
    {
        std::cerr << "ERROR: failed to parse the JSON model." << std::endl << e.what() << std::endl;
        return 1;
    }

    std::shared_ptr<Device> device;
    try
    {
        Adapter adapter = Adapter::Select(args.AdapterSubstring());
        device = std::make_shared<Device>(
            adapter.GetDXCoreAdapter(), 
            args.DebugLayersEnabled(), 
            args.CommandListType());
        std::cout << "Running on '" << adapter.GetDescription() << "'\n";
    }
    catch (std::exception& e)
    {
        std::cerr << "ERROR: failed to create a device:" << std::endl << e.what() << std::endl;
        return 1;
    }

    try
    {
        Executor executor{model, device, args};
        executor.Run();
    }
    catch (std::exception& e)
    {
        std::cerr << "ERROR: failed to execute the model." << std::endl << e.what() << std::endl;
        return 1;
    }

    return 0;
}