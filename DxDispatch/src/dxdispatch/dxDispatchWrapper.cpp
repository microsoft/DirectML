#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "JsonParsers.h"
#include "Executor.h"
#include "CommandLineArgs.h"
#include "ModuleInfo.h"
#include "dxDispatchWrapper.h"

using namespace Microsoft::WRL;

class ModelWrapper
{
public:
    ModelWrapper(Model&& model):
        m_model(std::move(model))
    {}

    ~ModelWrapper() = default;
    Model &Value()
    {
        return m_model;
    }
private:
    Model m_model;
};

HRESULT DxDispatch::CreateDxDispatchFromJsonString(
    _In_            int argc,
    _In_            char** argv,
    _In_opt_        LPCSTR jsonConfig,
    _In_opt_        IUnknown *adapterUnk,
    _In_opt_        IDxDispatchLogger *customLogger,
    _COM_Outptr_    IDxDispatch **dxDispatch)
{
    ComPtr<DxDispatch> dxDispatchImpl;
    ComPtr<IAdapter> adapter;

    if(adapterUnk)
    {
        RETURN_IF_FAILED(adapterUnk->QueryInterface(IID_PPV_ARGS(&adapter)));
    }
#ifdef WIN32
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<DxDispatch>(&dxDispatchImpl, argc, argv, jsonConfig, adapter.Get(), customLogger));
#else
     dxDispatchImpl = Make<DxDispatch>();
     dxDispatchImpl->RuntimeClassInitialize(argc, argv, jsonConfig, adapter.Get(), customLogger);
#endif
    return dxDispatchImpl->QueryInterface(IID_PPV_ARGS(dxDispatch));
}

DxDispatch::DxDispatch()
{
#ifdef WIN32
    AddDllRef();
#endif
}

HRESULT DxDispatch::RuntimeClassInitialize(
    _In_            int argc,
    _In_            char** argv,
    _In_opt_        LPCSTR jsonConfig,
    _In_opt_        IAdapter* adapter,
    _In_opt_        IDxDispatchLogger* customLogger) try
{
    auto lock = std::scoped_lock(m_lock);
    if (m_logger) // Should only initialize once
    {
        m_logger->LogError(fmt::format("{} can only be initialized once", __FUNCTION__).c_str());
        return E_UNEXPECTED;
    }
    if (customLogger)
    {
        m_logger = customLogger;
    }
    else
    {
        m_logger = Microsoft::WRL::Make<DxDispatchConsoleLogger>();
        RETURN_HR_IF_NULL(E_OUTOFMEMORY, m_logger);
    }

    m_options = std::make_shared<CommandLineArgs>(argc, (char**)argv);

    SetDisableAgilitySDK(m_options->DisableAgilitySDK());

    // Needs to be constructed *before* D3D12 device. A warning is printed if DXCore.dll is loaded first,
    // even though the D3D12Device isn't created yet, so we create the capture helper first to avoid this
    // message.
    m_pixCaptureHelper = std::make_shared<PixCaptureHelper>(m_options->GetPixCaptureType(), m_options->PixCaptureName());
    m_dxCoreModule = std::make_shared<DxCoreModule>();
    m_d3dModule = std::make_shared<D3d12Module>();
    m_dmlModule = std::make_shared<DmlModule>();

    if (m_options->PrintHelp())
    {
        m_logger->LogInfo(m_options->HelpText().c_str());
        return S_FALSE;
    }

    if (m_options->ShowDependencies())
    {
#if defined(_WIN32) && !defined(_GAMING_XBOX)
        // D3D12.dll lazily loads D3D12Core.dll. Calling any exported function forces D3D12Core.dll to load
        // so its version can be printed, and GetDebugInterface is inexpensive.
        Microsoft::WRL::ComPtr<ID3D12Debug> debug;
        m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&debug));
#endif

        PrintDependencies();
    }

    if (m_options->ShowAdapters())
    {
        for (auto& adapter : Adapter::GetAll(m_dxCoreModule))
        {
            m_logger->LogInfo(std::string(adapter.GetDetailedDescription() + "\n").c_str());
        }
    }
    auto model = m_options->ModelPath();
    if (!model.has_value() &&
        nullptr == jsonConfig)
    {
        return S_FALSE;
    }

    std::optional<Adapter> dxDispatchAdapter;
    if (nullptr == adapter)
    {
        dxDispatchAdapter = Adapter::Select(
            m_dxCoreModule,
            m_options->AdapterSubstring());
    }
    else
    {
        dxDispatchAdapter = Adapter(adapter, m_dxCoreModule);
    }
    m_options->SetAdapter(dxDispatchAdapter->GetAdapter());
    m_device = std::make_shared<Device>(
        dxDispatchAdapter->GetAdapter(),
        D3D_FEATURE_LEVEL_1_0_CORE,
        m_options->DebugLayersEnabled(),
        m_options->CommandListType(),
        m_options->DispatchRepeat(),
        m_options->GetUavBarrierAfterDispatch(),
        m_options->GetAliasingBarrierAfterDispatch(),
        m_pixCaptureHelper,
        m_d3dModule,
        m_dmlModule,
        m_logger.Get()
    );
    m_logger->LogInfo(fmt::format("Running on '{}'", dxDispatchAdapter->GetDescription()).c_str());

    if (m_options->ClearShaderCaches())
    {
        m_device->ClearShaderCaches();
    }

    auto inputPath = m_options->InputPath();
    auto outputPath = m_options->InputPath();

    if (!inputPath.has_value())
    {
        if (model.has_value())
        {
            inputPath = model.value().parent_path();
        }
        else
        {
            inputPath = std::filesystem::current_path();
        }
    }
    if (!outputPath.has_value())
    {
        outputPath = std::filesystem::current_path();
    }

    if (jsonConfig)
    {
        std::string_view fileContent(jsonConfig);

        rapidjson::Document doc;

        constexpr rapidjson::ParseFlag parseFlags = rapidjson::ParseFlag(
            rapidjson::kParseFullPrecisionFlag |
            rapidjson::kParseCommentsFlag |
            rapidjson::kParseTrailingCommasFlag |
            rapidjson::kParseStopWhenDoneFlag);

        std::vector<char> input{ jsonConfig, jsonConfig + strlen(jsonConfig) };
        doc.ParseInsitu<parseFlags>(&input[0]);
        m_modelWrapper = std::unique_ptr<ModelWrapper>(new ModelWrapper(
            JsonParsers::ParseModel(
                doc,
                fileContent,
                inputPath.value(),
                outputPath.value())));
    }
    else if (model.value().extension() == ".json")
    {
        m_modelWrapper = std::unique_ptr<ModelWrapper>(new ModelWrapper(JsonParsers::ParseModel(
            model.value(),
            inputPath.value(),
            outputPath.value())));
    }
    else if (model.value().extension() == ".onnx")
    {
#ifdef ONNXRUNTIME_NONE
        throw std::invalid_argument("ONNX dispatchables require ONNX Runtime");
#else
        auto name = model.value().filename().string();
        m_modelWrapper = std::unique_ptr<ModelWrapper>(
            new ModelWrapper(
                Model(
                    {}, // resource
                    { {name, Model::OnnxDispatchableDesc{model.value()}} },  // dispatchables
                    { {"dispatch", name, Model::DispatchCommand{name, {}, {}} } }, // commands
                    BucketAllocator{})
            )
        );
#endif
    }
    else
    {
        m_logger->LogError("Expected a .json or .onnx file");
        return E_NOTIMPL;
    }
    return S_OK;
} CATCH_RETURN();

HRESULT DxDispatch::RunAll() try
{
    auto lock = std::scoped_lock(m_lock);
    if (nullptr == m_modelWrapper) // Should only initialize once
    {
        m_logger->LogError(fmt::format("{} called before initialize", __FUNCTION__).c_str());
        return E_UNEXPECTED;
    }

    RETURN_IF_FAILED(m_pixCaptureHelper->BeginCapturableWork());
    m_executor = std::make_unique<Executor>(m_modelWrapper->Value(), m_device, *m_options, m_logger.Get());
    m_executor->Run();
    RETURN_IF_FAILED(m_pixCaptureHelper->EndCapturableWork());
    return S_OK;
    
} CATCH_RETURN();

UINT32 DxDispatch::GetCommandCount()
{
    auto lock = std::scoped_lock(m_lock);
    if (nullptr == m_modelWrapper) // Should only initialize once
    {
        m_logger->LogError(fmt::format("{} called before initialize", __FUNCTION__).c_str());
        return E_UNEXPECTED;
    }
    if (m_executor == nullptr)
    {
        m_executor = std::make_unique<Executor>(m_modelWrapper->Value(), m_device, *m_options, m_logger.Get());
    }
    return m_executor->GetCommandCount();
}

HRESULT DxDispatch::RunCommand(
            UINT32 index) try
{
    auto lock = std::scoped_lock(m_lock);
    if (nullptr == m_modelWrapper) // Should only initialize once
    {
        m_logger->LogError(fmt::format("{} called before initialize", __FUNCTION__).c_str());
        return E_UNEXPECTED;
    }
    if (m_executor == nullptr)
    {
        m_executor = std::make_unique<Executor>(m_modelWrapper->Value(), m_device, *m_options, m_logger.Get());
    }
    return m_executor->RunCommand(index);
}  CATCH_RETURN();

HRESULT DxDispatch::GetObject(
            REFGUID objectId,
            REFIID riid,
            _COM_Outptr_  void **ppvObject) try
{
    if (DxDispatch_DmlDevice == objectId)
    {
        return m_device->DML()->QueryInterface(riid, ppvObject);
    }
    else if (DxDispatch_DxDevice == objectId)
    {
        return m_device->D3D()->QueryInterface(riid, ppvObject);
    }
    else
    {
        return E_NOINTERFACE;
    }    
} CATCH_RETURN();

DxDispatch::~DxDispatch()
{
    // Ensure remaining D3D references are released before the D3D module is released,
    // regardless of normal exit or exception.
    m_modelWrapper.reset();
    m_options.reset();
    m_pixCaptureHelper.reset();
    m_device.reset();
#ifdef WIN32
    ReleaseDllRef();
#endif
}

