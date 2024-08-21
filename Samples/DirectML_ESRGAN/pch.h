#define UNICODE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP

#include <Windows.h>
#include <DirectML.h>
#include <d3d12.h>
#include <dxcore.h>
#include <wrl/client.h>
#include <wil/result.h>
#include <wincodec.h>

#include <optional>
#include <iostream>
#include <filesystem>
#include <span>
#include <string>

#include "half.hpp"
#include "cxxopts.hpp"
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"