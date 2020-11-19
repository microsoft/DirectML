//--------------------------------------------------------------------------------------
// DirectMLSuperResolution.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#pragma once

#include "DeviceResources.h"
#include "StepTimer.h"
#include "LoadWeights.h"
#include "MediaEnginePlayer.h"

// Force the default NCHW (batch/channels/height/width) tensor format, instead of determining
// this based on the GPU vendor. Setting this may help run on older Nvidia hardware.
#define FORCE_NCHW 0

// Use video frames as input to the DirectML model, instead of a static texture.
#define USE_VIDEO 1

// Let DirectML manage the data in the weight tensors. This can be faster on some hardware.
#define DML_MANAGED_WEIGHTS 1

class SmoothedFPS
{
public:
    SmoothedFPS(float secondsInterval = 1.f)
    {
        Initialize(secondsInterval);
    }

    void Initialize(float secondsInterval = 1.f)
    {
        m_secondsInterval = secondsInterval;
        m_timeAccumulator = 0.0f;
        m_frameAccumulator = 0;
        m_smoothedFPS = 0.0f;
    }

    void Tick(float DeltaTime)
    {
        m_timeAccumulator += DeltaTime;
        ++m_frameAccumulator;

        if (m_timeAccumulator >= m_secondsInterval)
        {
            m_smoothedFPS = (float)m_frameAccumulator / m_timeAccumulator;
            m_timeAccumulator = 0.0f;
            m_frameAccumulator = 0;
        }
    }

    float GetFPS() const { return m_smoothedFPS; }

private:
    float m_smoothedFPS;
    float m_timeAccumulator;
    uint32_t m_frameAccumulator;
    float m_secondsInterval;
};

enum class TensorLayout
{
    Default,
    NHWC
};

// A basic sample implementation that creates a D3D12 device and
// provides a render loop.
class Sample final : public DX::IDeviceNotify
{
public:

    Sample() noexcept(false);
    ~Sample();

    // Initialization and management
    void Initialize(HWND window, int width, int height);
    
    // Basic render loop
    void Tick();

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // Messages
    void OnActivated();
    void OnDeactivated();
    void OnSuspending();
    void OnResuming();
    void OnWindowMoved();
    void OnWindowSizeChanged(int width, int height);

    // Properties
    void GetDefaultSize( int& width, int& height ) const;

private:

    void Update(DX::StepTimer const& timer);
    void Render();
    void UpdateZoomVertexBuffer();

    void Clear();

    void CreateDeviceDependentResources();
    void CreateTextureResources();
    void CreateDirectMLResources();
    void InitializeDirectMLResources();
    void CreateUIResources();

    void GetStrides(
        _In_reads_(4) const uint32_t* sizes,
        TensorLayout layout,
        _Out_writes_(4) uint32_t* stridesOut
    );
    void CreateUpsampleLayer(
        _In_reads_(4) const uint32_t* inputSizes,
        _Inout_updates_(1) uint64_t* inputBufferRequiredSize,
        _Inout_updates_(1) uint64_t* outputBufferRequiredSize,
        _Out_writes_(4) uint32_t* outputSizesOut,
        _Out_writes_(1) IDMLCompiledOperator** compiledOpOut);
    void CreateConvolutionLayer(
        _In_reads_(4) const uint32_t* inputSizes,
        _In_reads_(4) const uint32_t* filterSizes,
        bool useBiasAndActivation,
        _Inout_updates_(1) uint64_t* inputBufferRequiredSize,
        _Inout_updates_(1) uint64_t* outputBufferRequiredSize,
        _Out_writes_(4) uint32_t* outputSizesOut,
        _Out_writes_(1) IDMLCompiledOperator** compiledOpOut);

    void CreateWeightTensors(
        WeightMapType& weights,
        const char* convLayerName,
        const char* scaleLayerName,
        const char* shiftLayerName,
        dml::Span<const uint32_t> filterSizes,
        DirectX::ResourceUploadBatch& uploadBatch,
        _Out_writes_(1) ID3D12Resource** filterWeightResourceOut,
        _Out_writes_opt_(1) ID3D12Resource** biasWeightResourceOut);
    void CreateWeightResource(
        _In_reads_(4) const uint32_t* tensorSizes,
        _Out_writes_(1) ID3D12Resource** d3dResourceOut);
    void CreateWindowSizeDependentResources();
    void BindTempResourceIfNeeded(
        DML_BINDING_PROPERTIES& bindingProps,
        _In_reads_(1) IDMLBindingTable* initBindingTable,
        _Out_writes_opt_(1) ID3D12Resource** tempResource);

    // DirectML method for setting up Tensors and creating operators
#if !(USE_DMLX)
    void CreateAdditionLayer(
        _In_reads_(4) const uint32_t* inputSizes,
        _Out_writes_(1) IDMLCompiledOperator** compiledOpOut);
#endif

    // Device resources
    std::unique_ptr<DX::DeviceResources>            m_deviceResources;

    // Rendering loop timer
    DX::StepTimer                                   m_timer;

    // Input devices
    std::unique_ptr<DirectX::GamePad>               m_gamePad;
    std::unique_ptr<DirectX::Keyboard>              m_keyboard;
    DirectX::GamePad::ButtonStateTracker            m_gamePadButtons;
    DirectX::Keyboard::KeyboardStateTracker         m_keyboardButtons;
    bool                                            m_ctrlConnected;

    // DirectXTK objects
    std::unique_ptr<DirectX::GraphicsMemory>        m_graphicsMemory;
    std::unique_ptr<DirectX::DescriptorHeap>        m_SRVDescriptorHeap;
    std::unique_ptr<DirectX::DescriptorHeap>        m_RTVDescriptorHeap;
    
    // UI
    SmoothedFPS                                     m_fps;
    std::unique_ptr<DirectX::BasicEffect>           m_lineEffect;
    std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>> m_lineBatch;
    std::unique_ptr<DirectX::DescriptorHeap>        m_fontDescriptorHeap;
    std::unique_ptr<DirectX::SpriteBatch>           m_spriteBatch;
    std::unique_ptr<DirectX::SpriteFont>            m_labelFont;
    std::unique_ptr<DirectX::SpriteFont>            m_labelFontBold;
    std::unique_ptr<DirectX::SpriteFont>            m_legendFont;
    std::unique_ptr<DirectX::SpriteFont>            m_ctrlFont;

    // Video player
    std::unique_ptr<MediaEnginePlayer>              m_player;
    HANDLE                                          m_sharedVideoTexture;

    // Direct3D 12 objects for rendering texture to screen
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_texRootSignatureNN;           // Nearest-neighbor texture upscale
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_texPipelineStateNN;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_texRootSignatureLinear;       // Bilinear texture upscale
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_texPipelineStateLinear;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_tensorRenderRootSignature;    // Render from DML tensor format to texture
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_tensorRenderPipelineState;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_texture;                      // Static input texture to render, if USE_VIDEO == 0
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_videoTexture;                 // Input video frame to render, if USE_VIDEO == 1
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_finalResultTexture;           // Upscaled 4K texture output
    uint32_t                                        m_origTextureHeight;
    uint32_t                                        m_origTextureWidth;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW                        m_vertexBufferView;
    D3D12_INDEX_BUFFER_VIEW                         m_indexBufferView;

    // Additional D3D12 objects for rendering zoomed picture-in-picture window
    DirectX::SharedGraphicsResource                 m_zoomedVertexHeap;
    D3D12_VERTEX_BUFFER_VIEW                        m_zoomedVertexBufferView;
    
    // Compute objects for converting texture to DML tensor format
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_computePSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_computeRootSignature;

    // DirectML objects
    Microsoft::WRL::ComPtr<IDMLDevice>              m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder>     m_dmlCommandRecorder;

    TensorLayout                                    m_tensorLayout;

    // Model layer sizes and indices
    static const size_t                             c_numUpsampleLayers = 2;
    static const size_t                             c_numConvLayers = 7;
    static const size_t                             c_numIntermediateBuffers = 2;

    enum OpTypes : uint32_t
    {
        e_opUpsample,
        e_opConv,
        e_opAdd,
        e_opCount
    };
    
    // Shared Resources
    std::unique_ptr<DirectX::DescriptorHeap>        m_dmlDescriptorHeap;
    
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelOutput;

    // DirectML Model Resources
#if !(USE_DMLX)
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelIntermediateResult[c_numIntermediateBuffers];

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelUpsamplePersistentResources[c_numUpsampleLayers];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelConvPersistentResources[c_numConvLayers];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelAddPersistentResource;

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelInitTemporaryResources[e_opCount];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelUpsampleTemporaryResources[c_numUpsampleLayers];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelConvTemporaryResources[c_numConvLayers];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelAddTemporaryResource;
#endif

    // DirectMLX Model Resources
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelConvFilterWeights[c_numConvLayers];
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelConvBiasWeights[c_numConvLayers];

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelPersistentResource;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelTemporaryResource;

    // DirectML operations
#if !(USE_DMLX)
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlUpsampleOps[c_numUpsampleLayers];
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlUpsampleBindings[c_numUpsampleLayers];
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlConvOps[c_numConvLayers];
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlConvBindings[c_numConvLayers];
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlAddResidualOp;
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlAddResidualBinding;
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_dmlOpInitializers[e_opCount];
#endif

    // DirectMLX operations
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlGraph;
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlBindingTable;
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_dmlOpInitializer;

    // Application state
    bool                                            m_useDml;
    bool                                            m_showPip;
    float                                           m_zoomX;
    float                                           m_zoomY;
    float                                           m_zoomWindowSize;
    bool                                            m_zoomUpdated;

    const float                                     c_minZoom = 0.005f;
    const float                                     c_maxZoom = 0.05f;
        
    // DirectX index enums
    enum SrvDescriptors : uint32_t
    {
        e_descTexture,
        e_descModelInput,
        e_descModelOutput,
        e_descFinalResultTextureSrv,
        e_srvDescCount
    };

    enum RtvDescriptors : uint32_t
    {
        e_descFinalResultTextureRtv,
        e_rtvDescCount
    };

    enum FontDescriptors : uint32_t
    {
        e_descLabelFont,
        e_descLabelFontBold,
        e_descLegendFont,
        e_descCtrlFont,
        e_fontDescCount,
    };

    enum ComputeRootParameters : uint32_t
    {
        e_crpIdxCB = 0,
        e_crpIdxSRV,
        e_crpIdxUAV,
        e_crpIdxCount
    };

    enum TensorRenderRootParameters : uint32_t
    {
        e_rrpIdxCB = 0,
        e_rrpIdxSRV,
        e_rrpIdxCount
    };
};