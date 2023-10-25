#pragma once

class Device;
class DmlModule;
class D3d12Module;
class DxCoreModule;
class PixCaptureHelper;
class CommandLineArgs;
class ModelWrapper;
class Executor;

#ifdef WIN32

extern ULONG AddDllRef();
extern ULONG ReleaseDllRef();

#else

WINADAPTER_IID(IDxDispatchLogger, 0xE05E128D, 0x9A97, 0x4AEE, 0x85, 0xD8, 0x17, 0x25, 0xC9, 0x2E, 0x41, 0x72);
WINADAPTER_IID(IDxDispatch,       0x1D5837DF, 0x8496, 0x42A6, 0xAA, 0x5B, 0xAA, 0x0D, 0xD1, 0x27, 0xC3, 0xB4);

#endif

class DxDispatch : public Microsoft::WRL::Base<IDxDispatch>
{
public:
    static HRESULT STDMETHODCALLTYPE CreateDxDispatchFromJsonString(
        _In_            int argc,
        _In_            char** argv,
        _In_opt_        LPCSTR jsonConfig,
        _In_opt_        IUnknown *pAdapter,
        _In_opt_        IDxDispatchLogger *pCustomLogger,
        _COM_Outptr_    IDxDispatch **dxDispatch );

    DxDispatch();
    HRESULT RuntimeClassInitialize(
        _In_            int argc,
        _In_            char** argv,
        _In_opt_        LPCSTR jsonConfig,
        _In_opt_        IAdapter* adapter,
        _In_opt_        IDxDispatchLogger* customLogger);

    // IDxDispatch
    HRESULT STDMETHODCALLTYPE  RunAll(
            ) final;

    UINT32 STDMETHODCALLTYPE GetCommandCount(
                ) final;

    HRESULT STDMETHODCALLTYPE  RunCommand(
        UINT32 index) final;

    HRESULT STDMETHODCALLTYPE GetObject(
        REFGUID objectId,
        REFIID riid,
        _COM_Outptr_  void **ppvObject) final;

protected:

    virtual ~DxDispatch();

    std::mutex                                  m_lock;
    UINT32                                      m_currentIndex = 0;
    UINT32                                      m_commandCount = 0;

    Microsoft::WRL::ComPtr<IDxDispatchLogger>   m_logger;
    std::unique_ptr<ModelWrapper>               m_modelWrapper;
    std::shared_ptr<Device>                     m_device;
    std::shared_ptr<DmlModule>                  m_dmlModule;
    std::shared_ptr<D3d12Module>                m_d3dModule;
    std::shared_ptr<DxCoreModule>               m_dxCoreModule;
    std::shared_ptr<PixCaptureHelper>           m_pixCaptureHelper;
    std::shared_ptr<CommandLineArgs>            m_options;
    std::shared_ptr<Executor>                   m_executor;
};