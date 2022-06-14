#include "pch.h"
#include "CommandLineArgs.h"
#include <cxxopts.hpp>
#include "config.h"

CommandLineArgs::CommandLineArgs(int argc, char** argv)
{
        auto banner = fmt::format(R"({} version {}
  DirectML     : {}
  D3D12        : {}
  DXCompiler   : {}
  PIX          : {}
  ONNX Runtime : {}
)",
        c_projectName, 
        c_projectVersion,
        c_directmlConfig,
        c_d3d12Config,
        c_dxcompilerConfig,
        c_pixConfig,
        c_ortConfig);
    
    cxxopts::Options options(c_projectName, banner);
    options.add_options()
        (
            "m,model", 
            "Path to JSON/ONNX model file", 
            cxxopts::value<decltype(m_modelPath)>()
        )
        (
            "d,debug", 
            "Enable D3D and DML debug layers", 
            cxxopts::value<bool>()
        )
        (
            "i,dispatch_iterations", 
            "The number of times to repeat each dispatch", 
            cxxopts::value<uint32_t>()->default_value("1")
        )
        (
            "h,help", 
            "Print command-line usage help", 
            cxxopts::value<bool>()
        )
        (
            "a,adapter", 
            "Substring to match a desired DirectX adapter", 
            cxxopts::value<std::string>()->default_value(m_adapterSubstring)
        )
        (
            "s,show_adapters", 
            "Show all available DirectX adapters", 
            cxxopts::value<bool>()
        )
        (
            "q,direct_queue", 
            "Use a direct queue/lists (default is a compute queue/lists)", 
            cxxopts::value<bool>()
        )
        // DxDispatch generates root signatures that are guaranteed to match HLSL source, which eliminates
        // having to write it inline in the HLSL file. DXC for Xbox precompiles shaders for Xbox (by default), 
        // but precompilation requires the root signature to be in the HLSL source itself; to allow use of the
        // DxDispatch-generated root signature we have to disable precompilation by defining `__XBOX_DISABLE_PRECOMPILE`
        // for each DXC invocation. The `xbox_allow_precompile` switch allows users to opt out of this behavior
        // and instead manually write root signatures in the HLSL source.
        (
            "xbox_allow_precompile",
            "Disables automatically defining __XBOX_DISABLE_PRECOMPILE when compiling shaders for Xbox",
            cxxopts::value<bool>()
        )
        (
            "c,pix_capture_type",
            "Type of PIX captures to take: gpu, timing, or manual.",
            cxxopts::value<std::string>()->default_value("manual")
        )
        (
            "f,onnx_free_dim_override",
            "List of free dimension overrides by name (ONNX models only). Can be repeated. Example: -f foo:3 -f bar:5",
            cxxopts::value<std::vector<std::string>>()->default_value({})
        )
        ;
    
    options.positional_help("<PATH_TO_MODEL>");

    options.parse_positional({"model"});
    auto result = options.parse(argc, argv);

    m_printHelp = result.arguments().empty() || (result.count("help") && result["help"].as<decltype(m_printHelp)>());

    if (result.count("debug")) 
    { 
        m_debugLayersEnabled = result["debug"].as<bool>(); 
    }
    
    if (result.count("dispatch_iterations"))
    {
        m_dispatchIterations = result["dispatch_iterations"].as<uint32_t>();
    }

    if (result.count("model")) 
    { 
        m_modelPath = result["model"].as<decltype(m_modelPath)>(); 
    }

    if (result.count("adapter")) 
    { 
        m_adapterSubstring = result["adapter"].as<decltype(m_adapterSubstring)>(); 
    }

    if (result.count("show_adapters")) 
    { 
        m_showAdapters = result["show_adapters"].as<bool>(); 
    }

    if (result.count("direct_queue") && result["direct_queue"].as<bool>()) 
    { 
        m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    }

    if (result.count("xbox_allow_precompile") && result["xbox_allow_precompile"].as<bool>())
    {
        m_forceDisablePrecompiledShadersOnXbox = false;
    }

    if (result.count("pix_capture_type")) 
    { 
        auto pixCaptureTypeStr = result["pix_capture_type"].as<decltype(m_adapterSubstring)>(); 
        if (pixCaptureTypeStr == "gpu")
        {
            m_pixCaptureType = PixCaptureType::ProgrammaticGpu;
        }
        else if (pixCaptureTypeStr == "timing")
        {
            m_pixCaptureType = PixCaptureType::ProgrammaticTiming;
#ifndef _GAMING_XBOX
            throw std::invalid_argument("Programmatic timing captures are not supported");
#endif
        }
        else if (pixCaptureTypeStr == "manual")
        {
            m_pixCaptureType = PixCaptureType::Manual;
        }
        else
        {
            throw std::invalid_argument("Unexpected value for pix_capture_type. Must be 'gpu', 'timing', or 'manual'");
        }
    }

    if (result.count("onnx_free_dim_override"))
    {
        auto freeDimOverrides = result["onnx_free_dim_override"].as<std::vector<std::string>>(); 
        for (auto& value : freeDimOverrides)
        {
            auto splitPos = value.find(":");
            if (splitPos == std::string::npos)
            {
                throw std::invalid_argument("Expected ':' separating free dimension name and its value");
            }
            auto overrideName = value.substr(0, splitPos);
            auto overrideValue = value.substr(splitPos + 1);
            m_freeDimensionOverrides.emplace_back(overrideName, std::stoul(overrideValue));
        }
    }

    m_helpText = options.help();
}