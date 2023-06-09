#include "pch.h"
#include "DxModules.h"

#ifndef WIN32
#include <dlfcn.h>
#endif

#if !defined(_GAMING_XBOX) && defined(WIN32)
extern "C" { __declspec(dllexport) extern UINT D3D12SDKVersion = DIRECT3D_AGILITY_SDK_VERSION; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8"./D3D12/"; }

void SetDisableAgilitySDK(bool disableAgilitySDK)
{
    if (disableAgilitySDK)
    {
        D3D12SDKVersion = 0;
    }
}
#else
void SetDisableAgilitySDK(bool disableAgilitySDK) { }
#endif

Module::Module(const char* moduleName)
{
    if (!moduleName)
    {
        return;
    }

#ifdef WIN32
    m_module.reset(LoadLibraryA(moduleName));
#else
    m_module = dlopen(moduleName, RTLD_LAZY);
#endif
}

Module::Module(Module&& other)
{
    m_module = std::move(other.m_module);
    other.m_module = nullptr;
}

Module& Module::operator=(Module&& other)
{
    m_module = std::move(other.m_module);
    other.m_module = nullptr;
    return *this;
}

Module::~Module()
{
    if (m_module)
    {
#ifndef WIN32
        dlclose(m_module);
#endif
        m_module = nullptr;
    }
}

void* Module::GetSymbol(const char* name)
{
#ifdef WIN32
    return GetProcAddress(m_module.get(), name);
#else
    return dlsym(m_module, name);
#endif
}

D3d12Module::D3d12Module(const char* moduleName) : Module(moduleName)
{
    if (m_module)
    {
        InitSymbol(&m_d3d12CreateDevice, "D3D12CreateDevice");
        InitSymbol(&m_d3d12GetDebugInterface, "D3D12GetDebugInterface");
        InitSymbol(&m_d3d12SerializeVersionedRootSignature, "D3D12SerializeVersionedRootSignature");
    }
}

DxCoreModule::DxCoreModule(const char* moduleName) : Module(moduleName)
{
    if (m_module)
    {
        InitSymbol(&m_dxCoreCreateAdapterFactory, "DXCoreCreateAdapterFactory");
    }
}

DmlModule::DmlModule(const char* moduleName) : Module(moduleName)
{
    if (m_module)
    {
        InitSymbol(&m_dmlCreateDevice1, "DMLCreateDevice1");
    }
}