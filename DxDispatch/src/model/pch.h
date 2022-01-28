#pragma once

#define NOMINMAX

#ifndef _WIN32
#include <wsl/winadapter.h>
#include "directml_guids.h"
#endif

#include <unordered_map>
#include <vector>
#include <iostream>
#include <filesystem>
#include <variant>
#include <fstream>
#include <deque>
#include <optional>
#include <set>
#include <string_view>
#include <wrl/client.h>
#include <wil/result.h>
#include <gsl/gsl>
#include <DirectML.h>
#include "DirectMLX.h"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fmt/format.h>
#include <half.hpp>

template<class... Ts> struct overload : Ts... { using Ts::operator()...; };
template<class... Ts> overload(Ts...) -> overload<Ts...>;