//--------------------------------------------------------------------------------------
// LoadWeights.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#pragma once

#include <map>

typedef std::vector<float> WeightsType;
typedef std::map<std::string, WeightsType> WeightMapType;

bool LoadWeights(const std::string& fpath, WeightMapType& weightMap);