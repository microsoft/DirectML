#pragma once

class Device;
class DmlModule;
class D3d12Module;
class DxCoreModule;
class PixCaptureHelper;
class CommandLineArgs;
class ModelWrapper;


extern ULONG AddDllRef();

extern ULONG ReleaseDllRef();

extern BOOL CanUnload();

class DxDispatch final : public IDxDispatch
{
public:
    static HRESULT STDMETHODCALLTYPE CreateDxDispatchFromJsonString(
        _In_            int argc,
        _In_            char** argv,
        _In_opt_        LPCSTR jsonConfig,
        _In_opt_        IUnknown *pAdapter,
        _In_opt_        IDxDispatchLogger *pCustomLogger,
        _COM_Outptr_    IDxDispatch **dxDispatch );

    // IUnknown
    HRESULT STDMETHODCALLTYPE QueryInterface( 
        REFIID riid,
        _COM_Outptr_ void __RPC_FAR *__RPC_FAR *ppvObject) final;

    ULONG STDMETHODCALLTYPE AddRef( void) final;

    ULONG STDMETHODCALLTYPE Release( void) final;

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

private:
    DxDispatch();

    HRESULT Intialize(
        _In_            int argc,
        _In_            char** argv,
        _In_opt_        LPCSTR jsonConfig,
        _In_opt_        IAdapter *adapter,
        _In_opt_        IDxDispatchLogger *customLogger);

    virtual ~DxDispatch();

    std::mutex                                  m_lock;
    UINT32                                      m_currentIndex = 0;
    UINT32                                      m_commandCount = 0;
    volatile ULONG                              m_refCount = 0;

    Microsoft::WRL::ComPtr<IDxDispatchLogger>   m_logger;
    std::unique_ptr<ModelWrapper>               m_modelWrapper;
    std::shared_ptr<Device>                     m_device;
    std::shared_ptr<DmlModule>                  m_dmlModule;
    std::shared_ptr<D3d12Module>                m_d3dModule;
    std::shared_ptr<DxCoreModule>               m_dxCoreModule;
    std::shared_ptr<PixCaptureHelper>           m_pixCaptureHelper;
    std::shared_ptr<CommandLineArgs>            m_options;
};