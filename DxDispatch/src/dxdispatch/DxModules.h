#pragma once

// Wraps d3d12.dll / libd3d12.so
class D3d12Module
{
public:
#if defined(_GAMING_XBOX)
    D3d12Module(std::wstring_view moduleName = nullptr);
#elif defined(WIN32)
    D3d12Module(std::wstring_view moduleName = L"d3d12.dll");
#else
    D3d12Module(std::wstring_view moduleName = L"libd3d12.so");
#endif

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

// Wraps dxcore.dll / libdxcore.so
class DxCoreModule
{
public:
#if defined(_GAMING_XBOX)
    DxCoreModule(std::wstring_view moduleName = nullptr);
#elif defined(WIN32)
    DxCoreModule(std::wstring_view moduleName = L"dxcore.dll");
#else
    DxCoreModule(std::wstring_view moduleName = L"libdxcore.so");
#endif

    HRESULT CreateAdapterFactory(REFIID riid, void** factory);

private:
    // DXCoreCreateAdapterFactory has a C++ overload so we must be explicit in
    // the function signature.
    using DXCoreCreateAdapterFactoryFn = HRESULT __stdcall(REFIID, void**);

    wil::unique_hmodule m_module;
    DXCoreCreateAdapterFactoryFn* m_dxCoreCreateAdapterFactory = nullptr;
};