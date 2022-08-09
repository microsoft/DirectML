#pragma once

#include "PixCaptureHelper.h"

class CommandLineArgs
{
public:
    CommandLineArgs() = default;
    CommandLineArgs(int argc, char** argv);

    bool ShowAdapters() const { return m_showAdapters; }
    bool PrintHelp() const { return m_printHelp; }
    bool DebugLayersEnabled() const { return m_debugLayersEnabled; }
    bool ForceDisablePrecompiledShadersOnXbox() const { return m_forceDisablePrecompiledShadersOnXbox; }
    const std::string& AdapterSubstring() const { return m_adapterSubstring; }
    const std::filesystem::path& ModelPath() const { return m_modelPath; }
    const std::string& HelpText() const { return m_helpText; }
    uint32_t DispatchIterations() const { return m_dispatchIterations; }
    std::optional<uint32_t> TimeToRunInMilliseconds() const { return m_timeToRunInMilliseconds; }
    D3D12_COMMAND_LIST_TYPE CommandListType() const { return m_commandListType; }
    PixCaptureType GetPixCaptureType() const { return m_pixCaptureType; }
    gsl::span<const std::pair<std::string, uint32_t>> GetOnnxFreeDimensionOverrides() const { return m_freeDimensionOverrides; }

private:
    bool m_showAdapters = false;
    bool m_printHelp = false;
    bool m_debugLayersEnabled = false;
    bool m_forceDisablePrecompiledShadersOnXbox = true;
    std::string m_adapterSubstring = "";
    std::filesystem::path m_modelPath;
    std::string m_helpText;
    uint32_t m_dispatchIterations = 1;
    std::optional<uint32_t> m_timeToRunInMilliseconds = {};
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    PixCaptureType m_pixCaptureType = PixCaptureType::Manual;

    // [onnx models] Overrides for free dimensions. First value in pair the name, second value is the dimension size.
    std::vector<std::pair<std::string, uint32_t>> m_freeDimensionOverrides;
};