#pragma once

#include "PixCaptureHelper.h"

class CommandLineArgs
{
public:
    CommandLineArgs() = default;
    CommandLineArgs(int argc, char** argv);

    bool ShowAdapters() const { return m_showAdapters; }
    bool ShowDependencies() const { return m_showDependencies; }
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
    const std::string& PixCaptureName() const { return m_pixCaptureName; }
    gsl::span<const std::pair<std::string, uint32_t>> GetOnnxFreeDimensionNameOverrides() const { return m_freeDimensionNameOverrides; }
    gsl::span<const std::pair<std::string, uint32_t>> GetOnnxFreeDimensionDenotationOverrides() const { return m_freeDimensionDenotationOverrides; }

private:
    bool m_showAdapters = false;
    bool m_showDependencies = false;
    bool m_printHelp = false;
    bool m_debugLayersEnabled = false;
    bool m_forceDisablePrecompiledShadersOnXbox = true;
    std::string m_adapterSubstring = "";
    std::filesystem::path m_modelPath;
    std::string m_pixCaptureName;
    std::string m_helpText;
    uint32_t m_dispatchIterations = 1;
    std::optional<uint32_t> m_timeToRunInMilliseconds = {};

    // Tools like PIX generally work better when work is recorded into a graphics queue, so it's set as the default here.
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    PixCaptureType m_pixCaptureType = PixCaptureType::Manual;

    // [onnx models] Overrides for free dimensions by name. 
    // The first value in the pair is the name, and the second value is the dimension size.
    std::vector<std::pair<std::string, uint32_t>> m_freeDimensionNameOverrides;

    // [onnx models] Overrides for free dimensions by denotation. 
    // The first value in the pair is the denotation, and the second value is the dimension size.
    std::vector<std::pair<std::string, uint32_t>> m_freeDimensionDenotationOverrides;
};