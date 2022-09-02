#pragma once

class D3d12Module
{
public:
    D3d12Module(std::wstring_view moduleName);

    HRESULT CreateDevice(
        IUnknown* adapter,
        D3D_FEATURE_LEVEL minimumFeatureLevel,
        REFIID riid,
        void** device
    );

    HRESULT GetDebugInterface(
        REFIID riid,
        void** debug
    );

    HRESULT SerializeVersionedRootSignature(
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* rootSignature,
        ID3DBlob** blob,
        ID3DBlob** errorBlob
    );

private:
    wil::unique_hmodule m_module;
    decltype(&D3D12CreateDevice) m_d3d12CreateDevice = nullptr;
    decltype(&D3D12GetDebugInterface) m_d3d12GetDebugInterface = nullptr;
    decltype(&D3D12SerializeVersionedRootSignature) m_d3d12SerializeVersionedRootSignature = nullptr;
};