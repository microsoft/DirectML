#include "pch.h"
#include "D3D12Module.h"

D3d12Module::D3d12Module(std::wstring_view moduleName)
{
   m_module.reset(LoadLibraryW(moduleName.data()));

   if (m_module)
   {
       m_d3d12CreateDevice = reinterpret_cast<decltype(&D3D12CreateDevice)>(
           GetProcAddress(m_module.get(), "D3D12CreateDevice")
           );

       m_d3d12GetDebugInterface = reinterpret_cast<decltype(&D3D12GetDebugInterface)>(
           GetProcAddress(m_module.get(), "D3D12GetDebugInterface")
           );

       m_d3d12SerializeVersionedRootSignature = reinterpret_cast<decltype(&D3D12SerializeVersionedRootSignature)>(
           GetProcAddress(m_module.get(), "D3D12SerializeVersionedRootSignature")
           );
   }
}

HRESULT D3d12Module::CreateDevice(
   IUnknown* adapter,
   D3D_FEATURE_LEVEL minimumFeatureLevel,
   REFIID riid,
   void** device
   )
{
   RETURN_HR_IF_NULL(E_FAIL, m_module.get());
   RETURN_HR_IF_NULL(E_FAIL, m_d3d12CreateDevice);
   return m_d3d12CreateDevice(adapter, minimumFeatureLevel, riid, device);
}

HRESULT D3d12Module::GetDebugInterface(
   REFIID riid,
   void** debug
   )
{
   RETURN_HR_IF_NULL(E_FAIL, m_module.get());
   RETURN_HR_IF_NULL(E_FAIL, m_d3d12GetDebugInterface);
   return m_d3d12GetDebugInterface(riid, debug);
}

HRESULT D3d12Module::SerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* rootSignature,
    ID3DBlob** blob,
    ID3DBlob** errorBlob
    )
{
    RETURN_HR_IF_NULL(E_FAIL, m_module.get());
    RETURN_HR_IF_NULL(E_FAIL, m_d3d12SerializeVersionedRootSignature);
    return m_d3d12SerializeVersionedRootSignature(rootSignature, blob, errorBlob);
}