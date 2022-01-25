#pragma once

#define NOMINMAX

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

#include <wrl/client.h>
#include <wil/result.h>
#include <Windows.h>
#include <d3d12.h>
#include <dxcore.h>

#include <gsl/gsl>

#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h>
#include <directx/d3dx12.h>
#include "DirectMLX.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <fmt/format.h>