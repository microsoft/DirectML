#pragma once

#include "DxModules.h"

class Adapter
{
public:
    Adapter() = default;
    Adapter(IAdapter* adapter, std::shared_ptr<DxCoreModule>);

    IAdapter* GetAdapter() { return m_adapter.Get(); }
    std::string_view GetDescription() const { return m_description; }
    std::string GetDetailedDescription() const;

    static Adapter Select(std::shared_ptr<DxCoreModule> module, std::string_view adapterSubstring = {});
    static std::vector<Adapter> GetAll(std::shared_ptr<DxCoreModule> module);

private:
#ifndef _GAMING_XBOX
    template <typename T>
    bool TryGetProperty(DXCoreAdapterProperty prop, T& outputValue)
    {
        if (m_adapter->IsPropertySupported(prop))
        {
            THROW_IF_FAILED(m_adapter->GetProperty(prop, &outputValue));
            return true;
        }
        return false;
    }

    template <>
    bool TryGetProperty(DXCoreAdapterProperty prop, std::string& outputValue)
    {
        if (m_adapter->IsPropertySupported(prop))
        {
            size_t propSize;
            THROW_IF_FAILED(m_adapter->GetPropertySize(prop, &propSize));

            outputValue.resize(propSize);
            THROW_IF_FAILED(m_adapter->GetProperty(prop, propSize, outputValue.data()));

            // Trim any trailing nul characters.
            while (!outputValue.empty() && outputValue.back() == '\0')
            {
                outputValue.pop_back();
            }

            return true;
        }
        return false;
    }
#endif

private:
    std::shared_ptr<DxCoreModule> m_dxCoreModule;
    Microsoft::WRL::ComPtr<IAdapter> m_adapter;
    std::string m_description;
    std::string m_driverVersion;
    bool m_isHardware = false;
    bool m_isDetachable = false;
    bool m_isIntegrated = false;

    struct
    {
        union
        {
            struct
            {
                uint16_t d;
                uint16_t c;
                uint16_t b;
                uint16_t a;
            } parts;
            uint64_t value;
        };
    } m_driverVersionRaw;

#ifndef _GAMING_XBOX
    DXCoreHardwareID m_hardwareId = {};
#endif

    // Bytes of dedicated adapter memory (not shared with CPU).
    uint64_t m_dedicatedAdapterMemory = 0;

    // Bytes of system memory reserved for this adapter (not shared with CPU).
    uint64_t m_dedicatedSystemMemory = 0;

    // Maximum bytes of system memory that may be consumed by the adapter during operation.
    uint64_t m_sharedSystemMemory = 0;

    bool m_isSupported_D3D12_GRAPHICS = false;
    bool m_isSupported_CORE_COMPUTE = false;
    bool m_isSupported_GENERIC_ML = false;
};