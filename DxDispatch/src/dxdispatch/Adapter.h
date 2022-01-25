#pragma once

class Adapter
{
public:
    Adapter() = default;
    Adapter(IDXCoreAdapter* adapter);

    IDXCoreAdapter* GetDXCoreAdapter() { return m_adapter.Get(); }
    std::string_view GetDescription() const { return m_description; }
    std::string GetDetailedDescription() const;

    static Adapter Select(std::string_view adapterSubstring = {});
    static std::vector<Adapter> GetAll();

private:
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
            
            return true;
        }
        return false;
    }

private:
    Microsoft::WRL::ComPtr<IDXCoreAdapter> m_adapter;
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

    DXCoreHardwareID m_hardwareId = {};

    // Bytes of dedicated adapter memory (not shared with CPU).
    uint64_t m_dedicatedAdapterMemory = 0;

    // Bytes of system memory reserved for this adapter (not shared with CPU).
    uint64_t m_dedicatedSystemMemory = 0;

    // Maximum bytes of system memory that may be consumed by the adapter during operation.
    uint64_t m_sharedSystemMemory = 0;
};