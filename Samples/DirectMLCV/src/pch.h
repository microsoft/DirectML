#pragma once

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
#include <wrl/client.h>
#include <wil/result.h>
#include <wil/resource.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include "d3dx12.h"

#include <optional>
#include <string>