#include <dxcore.h>
#include <wrl/client.h>
#include <wil/result.h>

std::tuple<Microsoft::WRL::ComPtr<IDXCoreAdapter>, D3D_FEATURE_LEVEL> SelectAdapter(
    std::string_view adapterNameFilter = "")
{
    using Microsoft::WRL::ComPtr;

    ComPtr<IDXCoreAdapterFactory> adapterFactory;
    THROW_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(adapterFactory.GetAddressOf())));

    // First try getting all GENERIC_ML devices, which is the broadest set of adapters 
    // and includes both GPUs and NPUs; however, running this sample on an older build of 
    // Windows may not have drivers that report GENERIC_ML.
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_1_0_GENERIC;
    ComPtr<IDXCoreAdapterList> adapterList;
    THROW_IF_FAILED(adapterFactory->CreateAdapterList(
        1,
        &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML,
        adapterList.GetAddressOf()
    ));

    // Fall back to CORE_COMPUTE if GENERIC_ML devices are not available. This is a more restricted
    // set of adapters and may filter out some NPUs.
    if (adapterList->GetAdapterCount() == 0)
    {
        featureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
        THROW_IF_FAILED(adapterFactory->CreateAdapterList(
            1, 
            &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, 
            adapterList.GetAddressOf()
        ));
    }

    if (adapterList->GetAdapterCount() == 0)
    {
        throw std::runtime_error("No compatible adapters found.");
    }

    // Sort the adapters by preference, with hardware and high-performance adapters first.
    DXCoreAdapterPreference preferences[] = 
    {
        DXCoreAdapterPreference::Hardware,
        DXCoreAdapterPreference::HighPerformance
    };

    THROW_IF_FAILED(adapterList->Sort(_countof(preferences), preferences));

    std::vector<ComPtr<IDXCoreAdapter>> adapters;
    std::vector<std::string> adapterDescriptions;
    std::optional<uint32_t> firstAdapterMatchingNameFilter;

    for (uint32_t i = 0; i < adapterList->GetAdapterCount(); i++)
    {
        ComPtr<IDXCoreAdapter> adapter;
        THROW_IF_FAILED(adapterList->GetAdapter(i, adapter.GetAddressOf()));

        size_t descriptionSize;
        THROW_IF_FAILED(adapter->GetPropertySize(
            DXCoreAdapterProperty::DriverDescription, 
            &descriptionSize
        ));

        std::string adapterDescription(descriptionSize - 1, '\0');
        THROW_IF_FAILED(adapter->GetProperty(
            DXCoreAdapterProperty::DriverDescription, 
            descriptionSize, 
            adapterDescription.data()
        ));

        adapters.push_back(adapter);
        adapterDescriptions.push_back(adapterDescription);

        if (!firstAdapterMatchingNameFilter &&
            adapterDescription.find(adapterNameFilter) != std::string::npos)
        {
            firstAdapterMatchingNameFilter = i;
            std::cout << "Adapter[" << i << "]: " << adapterDescription << " (SELECTED)\n";
        }
        else
        {
            std::cout << "Adapter[" << i << "]: " << adapterDescription << "\n";
        }
    }

    if (!firstAdapterMatchingNameFilter)
    {
        throw std::invalid_argument("No adapters match the provided name filter.");
    }

    return { adapters[*firstAdapterMatchingNameFilter], featureLevel };
}

std::tuple<Microsoft::WRL::ComPtr<IDMLDevice>, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> CreateDmlDeviceAndCommandQueue(std::string_view adapterNameFilter = "")
{
    using Microsoft::WRL::ComPtr;
    
    auto [adapter, featureLevel] = SelectAdapter(adapterNameFilter);

    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), featureLevel, IID_PPV_ARGS(&d3d12Device)));

    ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)));

    D3D12_COMMAND_QUEUE_DESC queueDesc = 
    {
        .Type = D3D12_COMMAND_LIST_TYPE_COMPUTE,
        .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0
    };

    ComPtr<ID3D12CommandQueue> commandQueue;
    THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

    return { dmlDevice, commandQueue };
}