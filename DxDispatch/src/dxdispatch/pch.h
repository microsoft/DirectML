#pragma once

#define NOMINMAX
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <sstream>
#include <variant>
#include <codecvt>
#include <locale>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <functional>
#include <numeric>
#include <thread>
#include <mutex>

#ifndef _WIN32
#include <wsl/winadapter.h>
#include "directml_guids.h"
#else
#include <Windows.h>
#include <wrl\implements.h>

namespace Microsoft::WRL
{
    // Helper wrapper over Microsoft::WRL::RuntimeClass. This is already implemented in 
    // common/inc/linux/wrl/linux_impl.h.
    template <typename... TInterfaces>
    using Base = Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
        TInterfaces...
    >;
}
#endif

#include <wil/result.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fmt/format.h>

#ifdef _GAMING_XBOX_SCARLETT // Xbox Series X/S - Use GDK
#include <d3d12_xs.h>
#include <d3dx12_xs.h>
#include <d3d12shader_xs.h>
#include <dxcapi_xs.h>
#include <pix3.h>
using IAdapter = IDXGIAdapter;
#elif _GAMING_XBOX_XBOXONE // XboxOne - Use GDK
#include <d3d12_x.h>
#include <d3dx12_x.h>
#include <d3d12shader_x.h>
#include <dxcapi_x.h>
#include <pix3.h>
using IAdapter = IDXGIAdapter;
#else // Desktop/PC - Use DirectX-Headers
#include <directx/d3d12.h>
#include <directx/d3dx12.h>
#include <directx/dxcore.h>
#ifndef DXCOMPILER_NONE
#include <directx/d3d12shader.h>
#endif
#include <dxcapi.h>
#include <WinPixEventRuntime/pix3.h>

#define IGraphicsUnknown IUnknown
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
using IAdapter = IDXCoreAdapter;
#endif

#include <DirectML.h>
#include "DirectMLX.h"

#include "DxDispatchInterface.h"
#include "Logging.h"