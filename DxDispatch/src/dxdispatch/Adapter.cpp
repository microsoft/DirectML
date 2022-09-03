#include "pch.h"
#include "Adapter.h"

using Microsoft::WRL::ComPtr;

Adapter::Adapter(IAdapter* adapter, std::shared_ptr<DxCoreModule> dxCoreModule) 
    : m_adapter(adapter), m_dxCoreModule(std::move(dxCoreModule))
{
#ifdef _GAMING_XBOX
    assert(adapter == nullptr);
    m_description = "Xbox";
    m_driverVersion = "Unknown";
    m_driverVersionRaw.value = 0;
    m_isHardware = true;
    m_isDetachable = false;
    m_isIntegrated = false;
#else
    TryGetProperty(DXCoreAdapterProperty::DriverDescription, m_description);
    TryGetProperty(DXCoreAdapterProperty::IsHardware, m_isHardware);
    TryGetProperty(DXCoreAdapterProperty::IsDetachable, m_isDetachable);
    TryGetProperty(DXCoreAdapterProperty::IsIntegrated, m_isIntegrated);
    TryGetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, m_dedicatedAdapterMemory);
    TryGetProperty(DXCoreAdapterProperty::DedicatedSystemMemory, m_dedicatedSystemMemory);
    TryGetProperty(DXCoreAdapterProperty::SharedSystemMemory, m_sharedSystemMemory);
    TryGetProperty(DXCoreAdapterProperty::SharedSystemMemory, m_sharedSystemMemory);
    TryGetProperty(DXCoreAdapterProperty::HardwareID, m_hardwareId);
    if (TryGetProperty(DXCoreAdapterProperty::DriverVersion, m_driverVersionRaw.value))
    {
        std::ostringstream oss;
        oss << m_driverVersionRaw.parts.a << ".";
        oss << m_driverVersionRaw.parts.b << ".";
        oss << m_driverVersionRaw.parts.c << ".";
        oss << m_driverVersionRaw.parts.d;
        m_driverVersion = oss.str();
    }
#endif
}

std::string Adapter::GetDetailedDescription() const
{
    auto FormatBytes = [](uint64_t sizeInBytes)
    {
        if (sizeInBytes > 1073741824) { return fmt::format("{:.2f} GB", sizeInBytes / 1073741824.0); }
        if (sizeInBytes > 1048576) { return fmt::format("{:.2f} MB", sizeInBytes / 1048576.0); }
        if (sizeInBytes > 1024) { return fmt::format("{:.2f} KB", sizeInBytes / 1024.0); }
        return fmt::format("{} bytes", sizeInBytes);
    };

    return fmt::format(
        R"({}
-Version: {}
-Hardware: {}
-Integrated: {}
-Dedicated Adapter Memory: {}
-Dedicated System Memory: {}
-Shared System Memory: {})", 
        m_description, 
        m_driverVersion,
        m_isHardware,
        m_isIntegrated,
        FormatBytes(m_dedicatedAdapterMemory),
        FormatBytes(m_dedicatedSystemMemory),
        FormatBytes(m_sharedSystemMemory)
        );
}

std::vector<Adapter> Adapter::GetAll(std::shared_ptr<DxCoreModule> module)
{
    std::vector<Adapter> adapters;

#ifdef _GAMING_XBOX
    adapters.emplace_back(nullptr);
#else
    ComPtr<IDXCoreAdapterFactory> adapterFactory;
    THROW_IF_FAILED(module->CreateAdapterFactory(IID_PPV_ARGS(adapterFactory.GetAddressOf())));

    ComPtr<IDXCoreAdapterList> adapterList;
    GUID attributes[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
    THROW_IF_FAILED(adapterFactory->CreateAdapterList(
        _countof(attributes),
        attributes,
        adapterList.GetAddressOf()));

    DXCoreAdapterPreference preferences[] = {
        DXCoreAdapterPreference::Hardware,
        DXCoreAdapterPreference::HighPerformance
    };
    THROW_IF_FAILED(adapterList->Sort(_countof(preferences), preferences));

    adapters.reserve(adapterList->GetAdapterCount());

    for (uint32_t i = 0; i < adapterList->GetAdapterCount(); i++)
    {
        ComPtr<IDXCoreAdapter> dxcoreAdapter;
        THROW_IF_FAILED(adapterList->GetAdapter(i, dxcoreAdapter.ReleaseAndGetAddressOf()));
        adapters.emplace_back(dxcoreAdapter.Get(), module);
    }
#endif

    return adapters;
}

Adapter Adapter::Select(std::shared_ptr<DxCoreModule> module, std::string_view adapterSubstring)
{
    auto adapters = Adapter::GetAll(module);

    for (auto& adapter : adapters)
    {
        if (strstr(adapter.GetDescription().data(), adapterSubstring.data()))
        {
            return adapter;
        }
    }

    throw std::invalid_argument(fmt::format("No adapter found that contains the substring '{}'.", adapterSubstring));
}