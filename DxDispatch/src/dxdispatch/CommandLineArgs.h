#pragma once

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
    D3D12_COMMAND_LIST_TYPE CommandListType() const { return m_commandListType; }

private:
    bool m_showAdapters = false;
    bool m_printHelp = false;
    bool m_debugLayersEnabled = false;
    bool m_forceDisablePrecompiledShadersOnXbox = true;
    std::string m_adapterSubstring = "";
    std::filesystem::path m_modelPath;
    std::string m_helpText;
    uint32_t m_dispatchIterations = 1;
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
};