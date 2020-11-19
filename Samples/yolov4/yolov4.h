//--------------------------------------------------------------------------------------
// yolov4.h
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "DeviceResources.h"
#include "StepTimer.h"
#include "MediaEnginePlayer.h"
#include "WeightData.h"

namespace YoloV4Constants
{
    // The classes of objects that yolov4 can detect
    static const char* const c_classes[] =
    {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "potted plant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    };

    static const uint32_t c_numClasses = ARRAYSIZE(c_classes);

    // Input images are rescaled to 608x608 before being fed into the model
    static const uint32_t c_inputWidth = 608;
    static const uint32_t c_inputHeight = 608;

    // Discard any predictions which have a low score (i.e. predictions that the model isn't very confident about)
    static const float c_scoreThreshold = 0.25f;

    // Threshold for non-maximal suppression (NMS) which determines how much overlap between bounding boxes is needed
    // before they're eliminated
    static const float c_nmsThreshold = 0.213f;

    // YoloV4 produces bounding boxes on different scales (small, medium, large) and the outputs of the model need
    // to be scaled according to their appropriate constants.
    struct BBoxData
    {
        float xyScale;
        float stride;
        std::array<float, 6> anchors;

        static BBoxData Small()
        {
            BBoxData data;
            data.xyScale = 1.2f;
            data.stride = 8;
            data.anchors = { 12,16,   19,36,   40,28   };
            return data;
        }

        static BBoxData Medium()
        {
            BBoxData data;
            data.xyScale = 1.1f;
            data.stride = 16;
            data.anchors = { 36,75,   76,55,   72,146  };
            return data;
        }

        static BBoxData Large()
        {
            BBoxData data;
            data.xyScale = 1.05f;
            data.stride = 32;
            data.anchors = { 142,110, 192,243, 459,401 };
            return data;
        }
    };
};

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

struct Prediction
{
    // Bounding box coordinates
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    uint32_t predictedClass;
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

    void Clear();

    void CreateDeviceDependentResources();
    void CreateTextureResources();
    void CreateDirectMLResources();
    void InitializeDirectMLResources();
    void CreateUIResources();
    void CreateWindowSizeDependentResources();

    struct ModelOutput
    {
        // DEFAULT buffer containing the output contents
        Microsoft::WRL::ComPtr<ID3D12Resource>      output;

        // READBACK buffer for retrieving the output contents from the GPU
        Microsoft::WRL::ComPtr<ID3D12Resource>      readback;

        // Size, format, etc. of the output data
        dml::TensorDesc                             desc;
    };

    // Given a raw output of the model, retrieves the predictions (a bounding box, detected class, and score) of the
    // model.
    void GetModelPredictions(
        const ModelOutput& modelOutput,
        const YoloV4Constants::BBoxData& constants,
        std::vector<Prediction>* out);

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
    uint32_t                                        m_origTextureHeight;
    uint32_t                                        m_origTextureWidth;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW                        m_vertexBufferView;
    D3D12_INDEX_BUFFER_VIEW                         m_indexBufferView;
    
    // Compute objects for converting texture to DML tensor format
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_computePSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_computeRootSignature;

    // DirectML objects
    Microsoft::WRL::ComPtr<IDMLDevice>              m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder>     m_dmlCommandRecorder;

    // Shared Resources
    std::unique_ptr<DirectX::DescriptorHeap>        m_dmlDescriptorHeap;
    
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelInput;
    ModelOutput                                     m_modelSOutput;
    ModelOutput                                     m_modelMOutput;
    ModelOutput                                     m_modelLOutput;

    std::optional<WeightData>                       m_modelWeights;

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelPersistentResource;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelTemporaryResource;

    // DirectMLX operations
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlGraph;
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlBindingTable;
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_dmlOpInitializer;
        
    // DirectX index enums
    enum SrvDescriptors : uint32_t
    {
        e_descTexture,
        e_descModelInput,
        e_srvDescCount
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