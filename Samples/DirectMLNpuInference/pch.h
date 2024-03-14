// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define NOMINMAX

#include <wil/result.h>
#include <wil/resource.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <optional>
#include <numeric>
#include <chrono>

#include "d3dx12.h" // The D3D12 Helper Library that you downloaded.
#include <dxgi1_4.h>
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS

#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
#include <DirectMLX.h>
