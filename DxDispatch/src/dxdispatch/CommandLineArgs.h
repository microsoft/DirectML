#pragma once

class CommandLineArgs
{
public:
    CommandLineArgs() = default;
    CommandLineArgs(int argc, char** argv);

    bool ShowAdapters() const { return m_showAdapters; }
    bool PrintHelp() const { return m_printHelp; }
    bool DebugLayersEnabled() const { return m_debugLayersEnabled; }
    bool BenchmarkingEnabled() const { return m_benchmarkingEnabled; }
    bool ForceDisablePrecompiledShadersOnXbox() const { return m_forceDisablePrecompiledShadersOnXbox; }
    const std::string& AdapterSubstring() const { return m_adapterSubstring; }
    const std::filesystem::path& ModelPath() const { return m_modelPath; }
    const std::string& HelpText() const { return m_helpText; }
    uint32_t BenchmarkingDispatchRepeat() const { return m_benchmarkingDispatchRepeat; }
    D3D12_COMMAND_LIST_TYPE CommandListType() const { return m_commandListType; }

private:
    bool m_showAdapters = false;
    bool m_printHelp = false;
    bool m_debugLayersEnabled = false;
    bool m_benchmarkingEnabled = false;
    bool m_forceDisablePrecompiledShadersOnXbox = true;
    std::string m_adapterSubstring = "";
    std::filesystem::path m_modelPath;
    std::string m_helpText;
    uint32_t m_benchmarkingDispatchRepeat = 128;
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
};