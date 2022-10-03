#pragma once

class Module
{
public:
    explicit Module(const char* moduleName);
    Module(const Module&) = delete;
    Module(Module&& other);
    Module& operator=(Module&& other);
    ~Module();

    void* GetSymbol(const char* name);

    template <typename T>
    void InitSymbol(T* functionPtrPtr, const char* name)
    {
        *functionPtrPtr = reinterpret_cast<T>(GetSymbol(name));
    }

    template <typename T, typename... Args>
    auto InvokeSymbol(T functionPtr, Args&&... args)
    {
        if (!m_module || !functionPtr)
        {
            return E_FAIL;
        }
        return functionPtr(std::forward<Args>(args)...);
    }

protected:
#ifdef WIN32
    wil::unique_hmodule m_module;
#else
    void* m_module = nullptr;
#endif
};

// Wraps d3d12.dll / libd3d12.so. Not used for Xbox.
class D3d12Module : public Module
{
public:
#if defined(_GAMING_XBOX)
    D3d12Module(const char* moduleName = nullptr);
#elif defined(WIN32)
    D3d12Module(const char* moduleName = "d3d12.dll");
#else
    D3d12Module(const char* moduleName = "libd3d12.so");
#endif

#ifndef _GAMING_XBOX
    inline HRESULT CreateDevice(IUnknown* adapter, D3D_FEATURE_LEVEL minimumFeatureLevel, REFIID riid, void** device)
    {
        return InvokeSymbol(m_d3d12CreateDevice, adapter, minimumFeatureLevel, riid, device);
    }
#endif

    inline HRESULT GetDebugInterface(REFIID riid, void** debug)
    {
        return InvokeSymbol(m_d3d12GetDebugInterface, riid, debug);
    }

    inline HRESULT SerializeVersionedRootSignature(
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* rootSignature,
        ID3DBlob** blob,
        ID3DBlob** errorBlob
        )
    {
        return InvokeSymbol(m_d3d12SerializeVersionedRootSignature, rootSignature, blob, errorBlob);
    }

private:
    decltype(&D3D12CreateDevice) m_d3d12CreateDevice = nullptr;
    decltype(&D3D12GetDebugInterface) m_d3d12GetDebugInterface = nullptr;
    decltype(&D3D12SerializeVersionedRootSignature) m_d3d12SerializeVersionedRootSignature = nullptr;
};

// Wraps dxcore.dll / libdxcore.so. Not used for Xbox.
class DxCoreModule : public Module
{
public:
#if defined(_GAMING_XBOX)
    DxCoreModule(const char* moduleName = nullptr);
#elif defined(WIN32)
    DxCoreModule(const char* moduleName = "dxcore.dll");
#else
    DxCoreModule(const char* moduleName = "libdxcore.so");
#endif

    inline HRESULT CreateAdapterFactory(REFIID riid, void** factory)
    {
        return InvokeSymbol(m_dxCoreCreateAdapterFactory, riid, factory);
    }

private:
    // DXCoreCreateAdapterFactory has a C++ overload so we must be explicit in
    // the function signature.
    using DXCoreCreateAdapterFactoryFn = HRESULT __stdcall(REFIID, void**);
    DXCoreCreateAdapterFactoryFn* m_dxCoreCreateAdapterFactory = nullptr;
};

// Wraps directml.dll / libdirectml.so.
class DmlModule : public Module
{
public:
#if defined(WIN32)
    DmlModule(const char* moduleName = "directml.dll");
#else
    DmlModule(const char* moduleName = "libdirectml.so");
#endif

    inline HRESULT CreateDevice1(ID3D12Device* d3d12Device, DML_CREATE_DEVICE_FLAGS flags, DML_FEATURE_LEVEL minimumFeatureLevel, REFIID riid, void** device)
    {
        return InvokeSymbol(m_dmlCreateDevice1, d3d12Device, flags, minimumFeatureLevel, riid, device);
    }

private:
    decltype(&DMLCreateDevice1) m_dmlCreateDevice1 = nullptr;
};