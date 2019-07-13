//--------------------------------------------------------------------------------------
// DirectMLSuperResolution.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "DirectMLSuperResolution.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"
#include "Float16Compressor.h"

// Use video frames as input to the DirectML model, instead of a static texture.
#define USE_VIDEO 1

// Force the default NCHW (batch/channels/height/width) tensor format, instead of determining
// this based on the GPU vendor. Setting this may help run on older Nvidia hardware.
#define FORCE_NCHW 0

// Let DirectML manage the data in the weight tensors. This can be faster on some hardware.
#define DML_MANAGED_WEIGHTS 1

const wchar_t* c_videoPath = L"FH3_540p60.mp4";
const wchar_t* c_imagePath = L"Assets\\FH3_1_540p.png";

const float c_pipSize = 0.45f;   // Relative size of the picture-in-picture window

extern void ExitSample();

using namespace DirectX;

using Microsoft::WRL::ComPtr;

#pragma warning(disable : 4238)

namespace
{
    struct Vertex
    {
        XMFLOAT4 position;
        XMFLOAT2 texcoord;
    };

    struct ImageLayoutCB
    {
        UINT Height;
        UINT Width;
        bool UseNhwc;
    };

    std::vector<uint8_t> LoadBGRAImage(const wchar_t* filename, uint32_t& width, uint32_t& height)
    {
        ComPtr<IWICImagingFactory> wicFactory;
        DX::ThrowIfFailed(CoCreateInstance(CLSID_WICImagingFactory2, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wicFactory)));

        ComPtr<IWICBitmapDecoder> decoder;
        DX::ThrowIfFailed(wicFactory->CreateDecoderFromFilename(filename, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, decoder.GetAddressOf()));

        ComPtr<IWICBitmapFrameDecode> frame;
        DX::ThrowIfFailed(decoder->GetFrame(0, frame.GetAddressOf()));

        DX::ThrowIfFailed(frame->GetSize(&width, &height));

        WICPixelFormatGUID pixelFormat;
        DX::ThrowIfFailed(frame->GetPixelFormat(&pixelFormat));

        uint32_t rowPitch = width * sizeof(uint32_t);
        uint32_t imageSize = rowPitch * height;

        std::vector<uint8_t> image;
        image.resize(size_t(imageSize));

        if (memcmp(&pixelFormat, &GUID_WICPixelFormat32bppBGRA, sizeof(GUID)) == 0)
        {
            DX::ThrowIfFailed(frame->CopyPixels(nullptr, rowPitch, imageSize, reinterpret_cast<BYTE*>(image.data())));
        }
        else
        {
            ComPtr<IWICFormatConverter> formatConverter;
            DX::ThrowIfFailed(wicFactory->CreateFormatConverter(formatConverter.GetAddressOf()));

            BOOL canConvert = FALSE;
            DX::ThrowIfFailed(formatConverter->CanConvert(pixelFormat, GUID_WICPixelFormat32bppBGRA, &canConvert));
            if (!canConvert)
            {
                throw std::exception("CanConvert");
            }

            DX::ThrowIfFailed(formatConverter->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
                WICBitmapDitherTypeErrorDiffusion, nullptr, 0, WICBitmapPaletteTypeMedianCut));

            DX::ThrowIfFailed(formatConverter->CopyPixels(nullptr, rowPitch, imageSize, reinterpret_cast<BYTE*>(image.data())));
        }

        return image;
    }

    // Divide and round up
    static UINT DivUp(UINT a, UINT b)
    {
        return (a + b - 1) / b;
    }

    void GetStrides(
        _In_reads_(4) const uint32_t* sizes,
        TensorLayout layout,
        _Out_writes_(4) uint32_t* stridesOut
    )
    {
        switch (layout)
        {
        case TensorLayout::NHWC:
            stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
            stridesOut[1] = 1;
            stridesOut[2] = sizes[1] * sizes[3];
            stridesOut[3] = sizes[1];
            break;

        default:
            stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
            stridesOut[1] = sizes[2] * sizes[3];
            stridesOut[2] = sizes[3];
            stridesOut[3] = 1;
        }
    }

    /** Calculates the minimum number of bytes required to store a buffer tensor with the specified type, sizes, and
    strides. The formula can be expressed as the following:

    IndexOfLastElement = dot(Sizes - 1, Strides);
    MinimumImpliedSizeInBytes = roundup((IndexOfLastElement + 1) * ElementSizeInBytes, 4)

    In other words, the minimum size of a tensor is the index of the one-past-the-end element, multiplied by the
    element size (e.g. 2 bytes for a FLOAT16 tensor). Additionally DirectML requires that all buffers bound must have
    a total size which is DWORD-aligned, and hence the minimum implied size in bytes must be rounded up to the nearest
    4-byte boundary.
    */
    inline UINT64 DMLCalcBufferTensorSize(
        DML_TENSOR_DATA_TYPE dataType,
        UINT dimensionCount,
        _In_reads_(dimensionCount) const UINT* sizes,
        _In_reads_opt_(dimensionCount) const UINT* strides
    )
    {
        UINT elementSizeInBytes = 0;
        switch (dataType)
        {
        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
        case DML_TENSOR_DATA_TYPE_INT32:
            elementSizeInBytes = 4;
            break;

        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
        case DML_TENSOR_DATA_TYPE_INT16:
            elementSizeInBytes = 2;
            break;

        case DML_TENSOR_DATA_TYPE_UINT8:
        case DML_TENSOR_DATA_TYPE_INT8:
            elementSizeInBytes = 1;
            break;

        default:
            return 0; // Invalid data type
        }

        UINT64 minimumImpliedSizeInBytes = 0;
        if (!strides)
        {
            minimumImpliedSizeInBytes = sizes[0];
            for (UINT i = 1; i < dimensionCount; ++i)
            {
                minimumImpliedSizeInBytes *= sizes[i];
            }
            minimumImpliedSizeInBytes *= elementSizeInBytes;
        }
        else
        {
            UINT indexOfLastElement = 0;
            for (UINT i = 0; i < dimensionCount; ++i)
            {
                indexOfLastElement += (sizes[i] - 1) * strides[i];
            }

            minimumImpliedSizeInBytes = (indexOfLastElement + 1) * elementSizeInBytes;
        }

        // Round up to the nearest 4 bytes.
        minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ui64;

        return minimumImpliedSizeInBytes;
    }

    UINT GetDescriptorCount(size_t numOps, IDMLCompiledOperator** ops, IDMLOperatorInitializer* initializer)
    {
        auto bindingProps = initializer->GetBindingProperties();

        UINT requiredDescriptorCount = bindingProps.RequiredDescriptorCount;

        for (size_t i = 0; i < numOps; i++)
        {
            bindingProps = ops[i]->GetBindingProperties();
            requiredDescriptorCount = std::max(requiredDescriptorCount, bindingProps.RequiredDescriptorCount);
        }

        return requiredDescriptorCount;
    }
}

Sample::Sample()
    : m_ctrlConnected(false)
    , m_tensorLayout(TensorLayout::Default)
    , m_useDml(true)
    , m_showPip(true)
    , m_zoomWindowSize(0.05f)
    , m_zoomX(0.5f)
    , m_zoomY(0.5f)
    , m_zoomUpdated(false)
{
    // Use gamma-correct rendering.
    // Renders only 2D, so no need for a depth buffer.
    m_deviceResources = std::make_unique<DX::DeviceResources>(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
        3, D3D_FEATURE_LEVEL_11_0, DX::DeviceResources::c_AllowTearing);
    m_deviceResources->RegisterDeviceNotify(this);
}

Sample::~Sample()
{
    if (m_deviceResources)
    {
        m_deviceResources->WaitForGpu();
    }
}

// Initialize the Direct3D resources required to run.
void Sample::Initialize(HWND window, int width, int height)
{
    m_gamePad = std::make_unique<GamePad>();

    m_keyboard = std::make_unique<Keyboard>();
    
    m_deviceResources->SetWindow(window, width, height);

    m_deviceResources->CreateDeviceResources();  	
    CreateDeviceDependentResources();

    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();
}

#pragma region Frame Update
// Executes basic render loop.
void Sample::Tick()
{
    m_timer.Tick([&]()
    {
        Update(m_timer);
    });

    Render();
}

// Updates the world.
void Sample::Update(DX::StepTimer const& timer)
{
    PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

    float elapsedTime = float(timer.GetElapsedSeconds());

    m_fps.Tick(elapsedTime);

    auto pad = m_gamePad->GetState(0);
    if (pad.IsConnected())
    {
        m_ctrlConnected = true;

        m_gamePadButtons.Update(pad);

        if (pad.IsViewPressed())
        {
            ExitSample();
        }

        if (m_gamePadButtons.a == DirectX::GamePad::ButtonStateTracker::PRESSED)
        {
            m_useDml = !m_useDml;
        }

        if (m_gamePadButtons.y == DirectX::GamePad::ButtonStateTracker::PRESSED)
        {
            m_showPip = !m_showPip;
        }

        if (m_gamePadButtons.x == DirectX::GamePad::ButtonStateTracker::PRESSED && m_player.get() != nullptr)
        {
            if (m_player->IsPlaying())
            {
                m_player->Pause();
            }
            else
            {
                m_player->Play();
            }
        }

        const float TriggerR = pad.triggers.right;
        const float TriggerL = pad.triggers.left;
        const float ThumbLeftX = pad.thumbSticks.leftX;
        const float ThumbLeftY = pad.thumbSticks.leftY;

        if (m_showPip && (TriggerR != 0.0f || TriggerL != 0.0f || ThumbLeftX != 0.0f || ThumbLeftY != 0.0f))
        {
            m_zoomWindowSize += TriggerL * elapsedTime * 0.05f;
            m_zoomWindowSize -= TriggerR * elapsedTime * 0.05f;
            m_zoomX += ThumbLeftX * elapsedTime * 4.0f * m_zoomWindowSize;
            m_zoomY -= ThumbLeftY * elapsedTime * 4.0f * m_zoomWindowSize;
            m_zoomUpdated = true;
        }
    }
    else
    {
        m_ctrlConnected = false;
        m_gamePadButtons.Reset();
    }

    auto kb = m_keyboard->GetState();
    m_keyboardButtons.Update(kb);

    if (kb.Escape)
    {
        ExitSample();
    }

    if (m_keyboardButtons.IsKeyPressed(Keyboard::Space))
    {
        m_useDml = !m_useDml;
    }

    if (m_keyboardButtons.IsKeyPressed(Keyboard::Z))
    {
        m_showPip = !m_showPip;
    }

    if (m_keyboardButtons.IsKeyPressed(Keyboard::Enter) && m_player.get() != nullptr)
    {
        if (m_player->IsPlaying())
        {
            m_player->Pause();
        }
        else
        {
            m_player->Play();
        }
    }

    if (m_showPip && (kb.W || kb.S || kb.Up || kb.Down || kb.Left || kb.Right))
    {
        m_zoomWindowSize += (kb.S ? 0.1f : (kb.W ? -0.1f : 0.0f)) * elapsedTime;
        m_zoomX += (kb.Right ? 4.0f : (kb.Left ? -4.0f : 0.0f)) * elapsedTime * m_zoomWindowSize;
        m_zoomY += (kb.Down ? 4.0f : (kb.Up ? -4.0f : 0.0f)) * elapsedTime * m_zoomWindowSize;
        m_zoomUpdated = true;
    }

    PIXEndEvent();
}
#pragma endregion

#pragma region Frame Render
// Draws the scene.
void Sample::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

#if USE_VIDEO
    // Get the latest video frame
    RECT r = { 0, 0, static_cast<LONG>(m_origTextureWidth), static_cast<LONG>(m_origTextureHeight) };
    MFVideoNormalizedRect rect = { 0.0f, 0.0f, 1.0f, 1.0f };
    m_player->TransferFrame(m_sharedVideoTexture, rect, r);
#endif

    // Prepare the command list to render a new frame.
    m_deviceResources->Prepare();
    Clear();
    
    auto commandList = m_deviceResources->GetCommandList();

    // If requested, run the current frame texture through the DirectML model to upscale it.
    if (m_useDml)
    {
        // Convert image to tensor format (original texture -> model input)
        {
            PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Convert input image");

            ID3D12DescriptorHeap* pHeaps[] = { m_SRVDescriptorHeap->Heap() };
            commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

            commandList->SetComputeRootSignature(m_computeRootSignature.Get());

            ImageLayoutCB imageLayoutCB = {};
            imageLayoutCB.Height = m_origTextureHeight;
            imageLayoutCB.Width = m_origTextureWidth;
            imageLayoutCB.UseNhwc = (m_tensorLayout == TensorLayout::NHWC);

            commandList->SetComputeRoot32BitConstants(e_crpIdxCB, 3, &imageLayoutCB, 0);
            commandList->SetComputeRootDescriptorTable(e_crpIdxSRV, m_SRVDescriptorHeap->GetGpuHandle(e_descTexture));
            commandList->SetComputeRootDescriptorTable(e_crpIdxUAV, m_SRVDescriptorHeap->GetGpuHandle(e_descModelInput));

            commandList->SetPipelineState(m_computePSO.Get());
            commandList->Dispatch(DivUp(m_origTextureWidth, 32), DivUp(m_origTextureHeight, 16), 1);

            commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

            PIXEndEvent(commandList);
        }

        // Run the DirectML operations (model input -> model output)
        {
            PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"DML ops");

            ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
            commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

            // Create an upsampled (nearest neighbor) version of the image first
            m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlUpsampleOps[0].Get(), m_dmlUpsampleBindings[0].Get());
            // No UAV barrier is required here since we don't use the result right away.

            // Run the intermediate model steps: 3 convolutions (with premultiplied batch normalization
            // baked into the weights), an upsample, 3 convolutions w/ premultiplied batch norm, 1 final convolution.
            // This generates a residual image.
            for (int i = 0; i < c_numConvLayers; i++)
            {
                // Convolution
                m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlConvOps[i].Get(), m_dmlConvBindings[i].Get());
                commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

                if (i == 2)
                {
                    // Intermediate upsample
                    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlUpsampleOps[1].Get(), m_dmlUpsampleBindings[1].Get());
                    commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
                }
            }

            // Add the residual image to the original nearest-neighbor upscale
            m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlAddResidualOp.Get(), m_dmlAddResidualBinding.Get());
            // UAV barrier handled below
    
            PIXEndEvent(commandList);
        }
    }

    // Render either the DML result or a bilinear upscale to a texture
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render to texture");

        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_finalResultTexture.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET),
            CD3DX12_RESOURCE_BARRIER::Transition(m_modelOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::UAV(nullptr)    
        };

        commandList->ResourceBarrier(m_useDml ? _countof(barriers) : 1, barriers);

        auto rtv = m_RTVDescriptorHeap->GetCpuHandle(e_descFinalResultTextureRtv);
        commandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        // Use linear clear color for gamma-correct rendering.
        commandList->ClearRenderTargetView(rtv, Colors::Black, 0, nullptr);
            
        D3D12_VIEWPORT texViewport = {};
        D3D12_RECT texScissor = {};
        texViewport.Height = static_cast<FLOAT>(texScissor.bottom = m_origTextureHeight * 2);
        texViewport.Width = static_cast<FLOAT>(texScissor.right = m_origTextureWidth * 2);
            
        commandList->RSSetViewports(1, &texViewport);
        commandList->RSSetScissorRects(1, &texScissor);

        auto heap = m_SRVDescriptorHeap->Heap();

        // Convert output tensor back to image (model output -> final result texture)
        if (m_useDml)
        {
            commandList->SetGraphicsRootSignature(m_tensorRenderRootSignature.Get());
            commandList->SetPipelineState(m_tensorRenderPipelineState.Get());
            commandList->SetDescriptorHeaps(1, &heap);

            ImageLayoutCB imageLayoutCB = {};
            imageLayoutCB.Height = m_origTextureHeight * 2;
            imageLayoutCB.Width = m_origTextureWidth * 2;
            imageLayoutCB.UseNhwc = (m_tensorLayout == TensorLayout::NHWC);

            commandList->SetGraphicsRoot32BitConstants(e_rrpIdxCB, 3, &imageLayoutCB, 0);
            commandList->SetGraphicsRootDescriptorTable(e_rrpIdxSRV, m_SRVDescriptorHeap->GetGpuHandle(e_descModelOutput));
        }
        // Bilinear upscale of original image (original texture -> final result texture)
        else
        {
            commandList->SetGraphicsRootSignature(m_texRootSignatureLinear.Get());
            commandList->SetPipelineState(m_texPipelineStateLinear.Get());
            commandList->SetDescriptorHeaps(1, &heap);

            commandList->SetGraphicsRootDescriptorTable(0, m_SRVDescriptorHeap->GetGpuHandle(e_descTexture));
        }

        // Set necessary state.
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
        commandList->IASetIndexBuffer(&m_indexBufferView);

        // Draw quad.
        commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);
            
        PIXEndEvent(commandList);
    }
    
    // Render the result to the screen
    auto viewport = m_deviceResources->GetScreenViewport();
    auto scissorRect = m_deviceResources->GetScissorRect();

    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render to screen");

        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_finalResultTexture.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_modelOutput.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::UAV(nullptr)
        };

        commandList->ResourceBarrier(m_useDml ? _countof(barriers) : 1, barriers);
        commandList->OMSetRenderTargets(1, &m_deviceResources->GetRenderTargetView(), FALSE, nullptr);

        commandList->SetGraphicsRootSignature(m_texRootSignatureLinear.Get());
        commandList->SetPipelineState(m_texPipelineStateLinear.Get());

        auto heap = m_SRVDescriptorHeap->Heap();
        commandList->SetDescriptorHeaps(1, &heap);

        commandList->SetGraphicsRootDescriptorTable(0,
            m_SRVDescriptorHeap->GetGpuHandle(e_descFinalResultTextureSrv));

        // Set necessary state.
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetIndexBuffer(&m_indexBufferView);

        // Draw full screen texture
        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);
        commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);

        commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);
        
        PIXEndEvent(commandList);
    }

    // Draw zoomed picture-in-picture window
    if (m_showPip)
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render PIP");

        // Use nearest-neighbor interpolation so individual pixels are visible.
        commandList->SetGraphicsRootSignature(m_texRootSignatureNN.Get());
        commandList->SetPipelineState(m_texPipelineStateNN.Get());

        if (m_zoomUpdated)
        {
            UpdateZoomVertexBuffer();
            m_zoomUpdated = false;
        }

        auto pipViewport = viewport;
        auto pipScissor = scissorRect;

        pipViewport.Width = viewport.Width * c_pipSize;
        pipViewport.Height = viewport.Height * c_pipSize;
        pipScissor.right = static_cast<LONG>(scissorRect.right * c_pipSize);
        pipScissor.bottom = static_cast<LONG>(scissorRect.bottom * c_pipSize);

        commandList->SetGraphicsRootDescriptorTable(0,
            m_SRVDescriptorHeap->GetGpuHandle(e_descFinalResultTextureSrv));

        commandList->RSSetViewports(1, &pipViewport);
        commandList->RSSetScissorRects(1, &pipScissor);
        commandList->IASetVertexBuffers(0, 1, &m_zoomedVertexBufferView);

        commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);
        
        PIXEndEvent(commandList);
    }
    
    // Render the UI
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render UI");

        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);

        auto size = m_deviceResources->GetOutputSize();
        auto safe = SimpleMath::Viewport::ComputeTitleSafeArea(size.right, size.bottom);
        
        // Draw a border around the PIP so it stands out.
        if (m_showPip)
        {
            m_lineEffect->Apply(commandList);
            m_lineBatch->Begin(commandList);

            VertexPositionColor lowerLeft(SimpleMath::Vector3(0.f, size.bottom * c_pipSize, 0.f), ATG::Colors::White);
            VertexPositionColor upperRight(SimpleMath::Vector3(size.right * c_pipSize, 0.f, 0.f), ATG::Colors::White);
            VertexPositionColor lowerRight(SimpleMath::Vector3(size.right * c_pipSize, size.bottom * c_pipSize, 0.f), ATG::Colors::White);

            m_lineBatch->DrawLine(lowerLeft, lowerRight);
            m_lineBatch->DrawLine(upperRight, lowerRight);

            m_lineBatch->End();
        }

        // Draw the text HUD.
        ID3D12DescriptorHeap* fontHeaps[] = { m_fontDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(fontHeaps), fontHeaps);
                
        m_spriteBatch->Begin(commandList);

        float xCenter = static_cast<float>(safe.left + (safe.right - safe.left) / 2);

        const wchar_t* mainLegend = m_ctrlConnected ?
            L"[View] Exit   [Y] Toggle PIP   [A] Upscale Mode   [X] Play/Pause"
            : L"ESC - Exit     Z - Toggle PIP     SPACE - Upscale Mode     ENTER - Play/Pause";
        SimpleMath::Vector2 mainLegendSize = m_legendFont->MeasureString(mainLegend);
        auto mainLegendPos = SimpleMath::Vector2(xCenter - mainLegendSize.x / 2, static_cast<float>(safe.bottom) - m_legendFont->GetLineSpacing());

        // Render a drop shadow by drawing the text twice with a slight offset.
        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
            mainLegend, mainLegendPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.0f, 0.0f, 0.0f, 0.25f));
        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
            mainLegend, mainLegendPos, ATG::Colors::White);

        if (m_showPip)
        {
            const wchar_t* pipLegend = m_ctrlConnected ?
                L"[LThumb] Move Zoom Target\n[LT][RT] Zoom In/Out"
                : L"ARROWS - Move Zoom Target\nW - Zoom In\nS - Zoom Out";
            auto pipLegendPos = SimpleMath::Vector2(static_cast<float>(safe.left), 20.f + size.bottom * c_pipSize);

            DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
                pipLegend, pipLegendPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.0f, 0.0f, 0.0f, 0.25f));
            DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
                pipLegend, pipLegendPos, ATG::Colors::White);
        }

        const wchar_t* modeLabel = L"Upscale mode:";
        SimpleMath::Vector2 modeLabelSize = m_labelFontBold->MeasureString(modeLabel);
        auto modeLabelPos = SimpleMath::Vector2(safe.right - modeLabelSize.x, static_cast<float>(safe.top));

        m_labelFontBold->DrawString(m_spriteBatch.get(), modeLabel, modeLabelPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFontBold->DrawString(m_spriteBatch.get(), modeLabel, modeLabelPos, ATG::Colors::White);

        const wchar_t* modeType = m_useDml ? L"Super-resolution Neural Network" : L"Bilinear Filter";
        SimpleMath::Vector2 modeTypeSize = m_labelFont->MeasureString(modeType);
        auto modeTypePos = SimpleMath::Vector2(safe.right - modeTypeSize.x, static_cast<float>(safe.top) + m_labelFontBold->GetLineSpacing());

        m_labelFont->DrawString(m_spriteBatch.get(), modeType, modeTypePos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), modeType, modeTypePos, ATG::Colors::White);

        wchar_t fps[16];
        swprintf_s(fps, 16, L"%0.2f FPS", m_fps.GetFPS());
        SimpleMath::Vector2 fpsSize = m_labelFont->MeasureString(fps);
        auto fpsPos = SimpleMath::Vector2(safe.right - fpsSize.x, static_cast<float>(safe.top) + m_labelFont->GetLineSpacing() * 3.f);

        m_labelFont->DrawString(m_spriteBatch.get(), fps, fpsPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), fps, fpsPos, ATG::Colors::White);

        m_spriteBatch->End();

        PIXEndEvent(commandList);
    }

    // Show the new frame.
    PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");

    m_deviceResources->Present();

    PIXEndEvent(m_deviceResources->GetCommandQueue());

    m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());
}

void Sample::UpdateZoomVertexBuffer()
{
    m_zoomWindowSize = std::max(c_minZoom, std::min(m_zoomWindowSize, c_maxZoom));

    float minZoomX = m_zoomWindowSize;
    float minZoomY = m_zoomWindowSize;
    float maxZoomX = 1.0f - m_zoomWindowSize;
    float maxZoomY = 1.0f - m_zoomWindowSize;

    m_zoomX = std::max(minZoomX, std::min(m_zoomX, maxZoomX));
    m_zoomY = std::max(minZoomY, std::min(m_zoomY, maxZoomY));

    float zoomLeft = m_zoomX - m_zoomWindowSize;
    float zoomRight = m_zoomX + m_zoomWindowSize;
    float zoomTop = m_zoomY - m_zoomWindowSize;
    float zoomBottom = m_zoomY + m_zoomWindowSize;

    Vertex* vertexBuffer = reinterpret_cast<Vertex*>(m_zoomedVertexHeap.Memory());
    vertexBuffer[0].texcoord = { zoomLeft, zoomBottom };
    vertexBuffer[1].texcoord = { zoomRight, zoomBottom };
    vertexBuffer[2].texcoord = { zoomRight, zoomTop };
    vertexBuffer[3].texcoord = { zoomLeft, zoomTop };
}

// Helper method to clear the back buffers.
void Sample::Clear()
{
    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Clear");

    // Clear the views.
    auto rtvDescriptor = m_deviceResources->GetRenderTargetView();

    commandList->OMSetRenderTargets(1, &rtvDescriptor, FALSE, nullptr);
    // Use linear clear color for gamma-correct rendering.
    commandList->ClearRenderTargetView(rtvDescriptor, ATG::ColorsLinear::Background, 0, nullptr);

    // Set the viewport and scissor rect.
    auto viewport = m_deviceResources->GetScreenViewport();
    auto scissorRect = m_deviceResources->GetScissorRect();
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissorRect);

    PIXEndEvent(commandList);
}
#pragma endregion

#pragma region Message Handlers
// Message handlers
void Sample::OnActivated()
{
}

void Sample::OnDeactivated()
{
}

void Sample::OnSuspending()
{
}

void Sample::OnResuming()
{
    m_timer.ResetElapsedTime();
    m_gamePadButtons.Reset();
    m_keyboardButtons.Reset();
}

void Sample::OnWindowMoved()
{
    auto r = m_deviceResources->GetOutputSize();
    m_deviceResources->WindowSizeChanged(r.right, r.bottom);
}

void Sample::OnWindowSizeChanged(int width, int height)
{
    if (!m_deviceResources->WindowSizeChanged(width, height))
        return;

    CreateWindowSizeDependentResources();
}

// Properties
void Sample::GetDefaultSize(int& width, int& height) const
{
    width = 1920;
    height = 1080;
}
#pragma endregion

#pragma region Direct3D Resources
// These are the resources that depend on the device.
void Sample::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

    // Create descriptor heaps.
    {
        m_SRVDescriptorHeap = std::make_unique<DescriptorHeap>(device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            e_srvDescCount);

        m_RTVDescriptorHeap = std::make_unique<DescriptorHeap>(device,
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            e_rtvDescCount);

        m_fontDescriptorHeap = std::make_unique<DescriptorHeap>(device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            e_fontDescCount);
    }

    CreateTextureResources();
    CreateDirectMLResources();
    InitializeDirectMLResources();
    CreateUIResources();
}

void Sample::CreateTextureResources()
{
    auto device = m_deviceResources->GetD3DDevice();
        
    // Create root signatures with one sampler and one texture--one for nearest neighbor sampling,
    // and one for bilinear.
    {
        CD3DX12_DESCRIPTOR_RANGE descRange = {};
        descRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_ROOT_PARAMETER rp = {};
        rp.InitAsDescriptorTable(1, &descRange, D3D12_SHADER_VISIBILITY_PIXEL);

        // Nearest neighbor sampling
        D3D12_STATIC_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.MaxAnisotropy = 16;
        samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        samplerDesc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        samplerDesc.MinLOD = 0;
        samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
        samplerDesc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
        rootSignatureDesc.Init(1, &rp, 1, &samplerDesc,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS);

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                OutputDebugStringA(reinterpret_cast<const char*>(error->GetBufferPointer()));
            }
            throw DX::com_exception(hr);
        }

        DX::ThrowIfFailed(
            device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                IID_PPV_ARGS(m_texRootSignatureNN.ReleaseAndGetAddressOf())));

        // Bilinear sampling
        samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        rootSignatureDesc.Init(1, &rp, 1, &samplerDesc,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS);

        hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                OutputDebugStringA(reinterpret_cast<const char*>(error->GetBufferPointer()));
            }
            throw DX::com_exception(hr);
        }

        DX::ThrowIfFailed(
            device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                IID_PPV_ARGS(m_texRootSignatureLinear.ReleaseAndGetAddressOf())));
    }

    // Create the pipeline state for a basic textured quad render, which includes loading shaders.
    {
        auto vertexShaderBlob = DX::ReadData(L"VertexShader.cso");
        auto pixelShaderBlob = DX::ReadData(L"PixelShader.cso");

        static const D3D12_INPUT_ELEMENT_DESC s_inputElementDesc[2] =
        {
            { "SV_Position", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  0 },
            { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  0 },
        };

        // Describe and create the graphics pipeline state objects (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { s_inputElementDesc, _countof(s_inputElementDesc) };
        psoDesc.pRootSignature = m_texRootSignatureNN.Get();
        psoDesc.VS = { vertexShaderBlob.data(), vertexShaderBlob.size() };
        psoDesc.PS = { pixelShaderBlob.data(), pixelShaderBlob.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = m_deviceResources->GetDepthBufferFormat();
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = m_deviceResources->GetBackBufferFormat();
        psoDesc.SampleDesc.Count = 1;
        DX::ThrowIfFailed(
            device->CreateGraphicsPipelineState(&psoDesc,
                IID_PPV_ARGS(m_texPipelineStateNN.ReleaseAndGetAddressOf())));

        psoDesc.pRootSignature = m_texRootSignatureLinear.Get();
        DX::ThrowIfFailed(
            device->CreateGraphicsPipelineState(&psoDesc,
                IID_PPV_ARGS(m_texPipelineStateLinear.ReleaseAndGetAddressOf())));
    }

    // Create vertex buffer for full screen texture render.
    {
        static const Vertex s_vertexData[4] =
        {
            { { -1.f, -1.f, 1.f, 1.0f },{ 0.f, 1.f } },
            { { 1.f, -1.f, 1.f, 1.0f },{ 1.f, 1.f } },
            { { 1.f,  1.f, 1.f, 1.0f },{ 1.f, 0.f } },
            { { -1.f,  1.f, 1.f, 1.0f },{ 0.f, 0.f } },
        };

        // Note: using upload heaps to transfer static data like vert buffers is not 
        // recommended. Every time the GPU needs it, the upload heap will be marshalled 
        // over. Please read up on Default Heap usage. An upload heap is used here for 
        // code simplicity and because there are very few verts to actually transfer.
        DX::ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(sizeof(s_vertexData)),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(m_vertexBuffer.ReleaseAndGetAddressOf())));

        // Copy the quad data to the vertex buffer.
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
        DX::ThrowIfFailed(
            m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, s_vertexData, sizeof(s_vertexData));
        m_vertexBuffer->Unmap(0, nullptr);

        // Initialize the vertex buffer view.
        m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
        m_vertexBufferView.StrideInBytes = sizeof(Vertex);
        m_vertexBufferView.SizeInBytes = sizeof(s_vertexData);
    }

    // Create vertex buffer for zoomed picture-in-picture window.
    {
        // Dynamic vertex buffer since this will change
        m_zoomedVertexHeap = GraphicsMemory::Get().Allocate(sizeof(Vertex) * 4);

        float zoomLeft = m_zoomX - m_zoomWindowSize;
        float zoomRight = m_zoomX + m_zoomWindowSize;
        float zoomTop = m_zoomY - m_zoomWindowSize;
        float zoomBottom = m_zoomY + m_zoomWindowSize;
        
        static const Vertex s_zoomedVertexData[4] =
        {
            { { -1.f, -1.f, 1.f, 1.0f },{ zoomLeft, zoomBottom } },
            { { 1.f, -1.f, 1.f, 1.0f },{ zoomRight, zoomBottom } },
            { { 1.f,  1.f, 1.f, 1.0f },{ zoomRight, zoomTop } },
            { { -1.f,  1.f, 1.f, 1.0f },{ zoomLeft, zoomTop } },
        };

        memcpy(m_zoomedVertexHeap.Memory(), s_zoomedVertexData, sizeof(s_zoomedVertexData));

        // Initialize the vertex buffer view.
        m_zoomedVertexBufferView.BufferLocation = m_zoomedVertexHeap.GpuAddress();
        m_zoomedVertexBufferView.StrideInBytes = sizeof(Vertex);
        m_zoomedVertexBufferView.SizeInBytes = sizeof(s_zoomedVertexData);
    }

    // Create index buffer.
    {
        static const uint16_t s_indexData[6] =
        {
            3,1,0,
            2,1,3,
        };

        // See note above
        DX::ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(sizeof(s_indexData)),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(m_indexBuffer.ReleaseAndGetAddressOf())));

        // Copy the data to the index buffer.
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
        DX::ThrowIfFailed(
            m_indexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, s_indexData, sizeof(s_indexData));
        m_indexBuffer->Unmap(0, nullptr);

        // Initialize the index buffer view.
        m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
        m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;
        m_indexBufferView.SizeInBytes = sizeof(s_indexData);
    }

#if USE_VIDEO
    // Create video player.
    {
        wchar_t buff[MAX_PATH];
        DX::FindMediaFile(buff, MAX_PATH, c_videoPath);

        m_player = std::make_unique<MediaEnginePlayer>();
        m_player->Initialize(m_deviceResources->GetDXGIFactory(), device, DXGI_FORMAT_B8G8R8A8_UNORM);
        m_player->SetSource(buff);

        while (!m_player->IsInfoReady())
        {
            SwitchToThread();
        }

        m_player->GetNativeVideoSize(m_origTextureWidth, m_origTextureHeight);
        m_player->SetLoop(true);

        // Create texture to receive video frames.
        CD3DX12_RESOURCE_DESC desc(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            0,
            m_origTextureWidth,
            m_origTextureHeight,
            1,
            1,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            1,
            0,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);

        CD3DX12_HEAP_PROPERTIES defaultHeapProperties(D3D12_HEAP_TYPE_DEFAULT);

        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &defaultHeapProperties,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                nullptr,
                IID_PPV_ARGS(m_videoTexture.ReleaseAndGetAddressOf())));

        CreateShaderResourceView(device, m_videoTexture.Get(), m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));

        DX::ThrowIfFailed(
            device->CreateSharedHandle(
                m_videoTexture.Get(),
                nullptr,
                GENERIC_ALL,
                nullptr,
                &m_sharedVideoTexture));
    }
#else
    // Create static texture.
    {
        auto commandList = m_deviceResources->GetCommandList();
        commandList->Reset(m_deviceResources->GetCommandAllocator(), nullptr);

        ComPtr<ID3D12Resource> textureUploadHeap;
    
        D3D12_RESOURCE_DESC txtDesc = {};
        txtDesc.MipLevels = txtDesc.DepthOrArraySize = 1;
        txtDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        txtDesc.SampleDesc.Count = 1;
        txtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

        wchar_t buff[MAX_PATH];
        DX::FindMediaFile(buff, MAX_PATH, c_imagePath);

        UINT width, height;
        auto image = LoadBGRAImage(buff, width, height);
        txtDesc.Width = m_origTextureWidth = width;
        txtDesc.Height = m_origTextureHeight = height;

        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &txtDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(m_texture.ReleaseAndGetAddressOf())));

        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(m_texture.Get(), 0, 1);

        // Create the GPU upload buffer.
        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(textureUploadHeap.GetAddressOf())));

        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = image.data();
        textureData.RowPitch = static_cast<LONG_PTR>(txtDesc.Width * sizeof(uint32_t));
        textureData.SlicePitch = image.size();

        UpdateSubresources(commandList, m_texture.Get(), textureUploadHeap.Get(), 0, 0, 1, &textureData);
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ));

        // Describe and create a SRV for the texture.
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = txtDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(m_texture.Get(), &srvDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));
    
        DX::ThrowIfFailed(commandList->Close());
        m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

        // Wait until assets have been uploaded to the GPU.
        m_deviceResources->WaitForGpu();
    }
#endif
}

void Sample::CreateDirectMLResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Shader for converting texture to tensor
    {
        auto computeShaderBlob = DX::ReadData(L"ImageToTensor.cso");

        // Define root table layout
        CD3DX12_DESCRIPTOR_RANGE descRange[2];
        descRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0
        descRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0); // u0

        CD3DX12_ROOT_PARAMETER rootParameters[3];
        rootParameters[e_crpIdxCB].InitAsConstants(3, 0);
        rootParameters[e_crpIdxSRV].InitAsDescriptorTable(1, &descRange[0], D3D12_SHADER_VISIBILITY_ALL);
        rootParameters[e_crpIdxUAV].InitAsDescriptorTable(1, &descRange[1], D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC rootSignature(_countof(rootParameters), rootParameters);

        ComPtr<ID3DBlob> serializedSignature;
        DX::ThrowIfFailed(
            D3D12SerializeRootSignature(&rootSignature, D3D_ROOT_SIGNATURE_VERSION_1, serializedSignature.GetAddressOf(), nullptr));

        // Create the root signature
        DX::ThrowIfFailed(
            device->CreateRootSignature(
                0,
                serializedSignature->GetBufferPointer(),
                serializedSignature->GetBufferSize(),
                IID_PPV_ARGS(m_computeRootSignature.ReleaseAndGetAddressOf())));

        m_computeRootSignature->SetName(L"Compute RS");

        // Create compute pipeline state
        D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
        descComputePSO.pRootSignature = m_computeRootSignature.Get();
        descComputePSO.CS.pShaderBytecode = computeShaderBlob.data();
        descComputePSO.CS.BytecodeLength = computeShaderBlob.size();

        DX::ThrowIfFailed(
            device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(m_computePSO.ReleaseAndGetAddressOf())));
        m_computePSO->SetName(L"Compute PSO");
    }

    // Shader for rendering DML result tensor to texture
    // This can also be done with a compute shader, depending on the app's needs.
    {
        auto vsShaderBlob = DX::ReadData(L"TensorToImageVS.cso");
        auto psShaderBlob = DX::ReadData(L"TensorToImagePS.cso");

        static const D3D12_INPUT_ELEMENT_DESC s_inputElementDesc[1] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  0 },
        };

        // Define root table layout
        CD3DX12_DESCRIPTOR_RANGE descRange[1];
        descRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE); // t0

        CD3DX12_ROOT_PARAMETER rootParameters[2];
        rootParameters[e_rrpIdxCB].InitAsConstants(3, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[e_rrpIdxSRV].InitAsDescriptorTable(1, &descRange[0], D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_SIGNATURE_DESC rootSignature(_countof(rootParameters), rootParameters,
            0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

        ComPtr<ID3DBlob> serializedSignature;
        DX::ThrowIfFailed(
            D3D12SerializeRootSignature(&rootSignature, D3D_ROOT_SIGNATURE_VERSION_1, serializedSignature.GetAddressOf(), nullptr));

        // Create the root signature
        DX::ThrowIfFailed(
            device->CreateRootSignature(
                0,
                serializedSignature->GetBufferPointer(),
                serializedSignature->GetBufferSize(),
                IID_PPV_ARGS(m_tensorRenderRootSignature.ReleaseAndGetAddressOf())));

        m_tensorRenderRootSignature->SetName(L"Tensor Render RS");

        // Create pipeline state
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { s_inputElementDesc, _countof(s_inputElementDesc) };
        psoDesc.pRootSignature = m_tensorRenderRootSignature.Get();
        psoDesc.VS = { vsShaderBlob.data(), vsShaderBlob.size() };
        psoDesc.PS = { psShaderBlob.data(), psShaderBlob.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = m_deviceResources->GetDepthBufferFormat();
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;

        DX::ThrowIfFailed(
            device->CreateGraphicsPipelineState(&psoDesc,
                IID_PPV_ARGS(m_tensorRenderPipelineState.ReleaseAndGetAddressOf())));

        m_tensorRenderPipelineState->SetName(L"Tensor Render PSO");

        // Resource to hold the rendered texture
        D3D12_RESOURCE_DESC txtDesc = {};
        txtDesc.MipLevels = txtDesc.DepthOrArraySize = 1;
        txtDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        txtDesc.SampleDesc.Count = 1;
        txtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        txtDesc.Width = m_origTextureWidth * 2;
        txtDesc.Height = m_origTextureHeight * 2;
        txtDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
                
        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &txtDesc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                &CD3DX12_CLEAR_VALUE(DXGI_FORMAT_B8G8R8A8_UNORM, Colors::Black),
                IID_PPV_ARGS(m_finalResultTexture.ReleaseAndGetAddressOf())));

        // Create an RTV for rendering to the texture, and an SRV for rendering it back to the screen
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
        rtvDesc.Format = txtDesc.Format;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(m_finalResultTexture.Get(), &rtvDesc, m_RTVDescriptorHeap->GetCpuHandle(e_descFinalResultTextureRtv));

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = txtDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(m_finalResultTexture.Get(), &srvDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descFinalResultTextureSrv));
    }

    // DirectML device
    {
#if _DEBUG
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice)));
#else
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
#endif

#if FORCE_NCHW
        m_tensorLayout = TensorLayout::Default;
#else
        // Determine the best tensor layout based on the GPU vendor.
        // This is a fairly coarse-grained method, but recent Nvidia GPUs tend to use NHWC
        // layouts, while others use the default NCHW. 
        ComPtr<IDXGIAdapter1> adapter;
        DX::ThrowIfFailed(m_deviceResources->GetDXGIFactory()->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)));

        DXGI_ADAPTER_DESC adapterDesc;
        adapter->GetDesc(&adapterDesc);

        if (adapterDesc.VendorId == 0x10DE) // Nvidia
        {
            // This is faster on recent Nvidia hardware, but may be a problem on older hardware.
            // If necessary, set FORCE_NCHW to override this.
            m_tensorLayout = TensorLayout::NHWC;
        }
        else
        {
            m_tensorLayout = TensorLayout::Default;
        }
#endif

        DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
        DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
        DX::ThrowIfFailed(m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query), &fp16Query, sizeof(fp16Supported), &fp16Supported));

        if (!fp16Supported.IsSupported)
        {
            throw std::exception("FP16 data type support is required for this sample.");
        }

        DX::ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder)));
    }

    uint64_t modelInputBufferSize = 0;
    uint64_t modelOutputBufferSize = 0;
    uint64_t intermediateBufferMaxSize[] = { 0, 0 };

    // DirectML operator resources--implementation of the super-resolution model
    {
        // Create an upscaled (nearest neighbor) version of the image first
        uint32_t modelInputSizes[] = { 1, 3, m_origTextureHeight, m_origTextureWidth };
        uint32_t upscaledInputSizes[4];
        CreateUpsampleLayer(modelInputSizes, &modelInputBufferSize, &modelOutputBufferSize, upscaledInputSizes, &m_dmlUpsampleOps[0]);

        // Create the residual with three convolutions, an upsample, and four more convolutions
        WeightMapType weights;
        if (!LoadWeights("Assets\\weights.bin", weights))
        {
            throw std::exception("loadWeights");
        }

        ResourceUploadBatch weightUploadBatch(device);
        weightUploadBatch.Begin();
        
        uint32_t filterSizes[] = { 32, 3, 5, 5 };
        uint32_t intermediateInputSizes[2][4];
        CreateConvolutionLayer(modelInputSizes, filterSizes, true, &modelInputBufferSize,
            &intermediateBufferMaxSize[0], intermediateInputSizes[0], &m_dmlConvOps[0]);
        CreateWeightTensors(weights, "conv1/weights", "conv1/BatchNorm/scale", "conv1/BatchNorm/shift",
            filterSizes, weightUploadBatch, &m_modelConvFilterWeights[0], &m_modelConvBiasWeights[0]);

        // Which intermediate resource to use as input for the current operation. The other will be
        // used as output. Then the next op will swap the order.
        int inputIndex = 0;

        filterSizes[0] = 64;	// output filters
        filterSizes[1] = 32;	// input channels
        filterSizes[2] = 3;		// filter height
        filterSizes[3] = 3;		// filter width
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[1]);
        CreateWeightTensors(weights, "conv2/weights", "conv2/BatchNorm/scale", "conv2/BatchNorm/shift",
            filterSizes, weightUploadBatch, &m_modelConvFilterWeights[1], &m_modelConvBiasWeights[1]);
        inputIndex = 1 - inputIndex;
        
        filterSizes[1] = 64;
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[2]);
        CreateWeightTensors(weights, "conv3/weights", "conv3/BatchNorm/scale", "conv3/BatchNorm/shift", 
            filterSizes, weightUploadBatch, &m_modelConvFilterWeights[2], &m_modelConvBiasWeights[2]);
        inputIndex = 1 - inputIndex;

        CreateUpsampleLayer(intermediateInputSizes[inputIndex], &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlUpsampleOps[1]);
        inputIndex = 1 - inputIndex;

        filterSizes[0] = 32;
        filterSizes[2] = 5;
        filterSizes[3] = 5;
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[3]);
        CreateWeightTensors(weights, "conv_up1/conv/weights", "conv_up1/conv/BatchNorm/scale", "conv_up1/conv/BatchNorm/shift",
            filterSizes, weightUploadBatch, &m_modelConvFilterWeights[3], &m_modelConvBiasWeights[3]);
        inputIndex = 1 - inputIndex;

        filterSizes[1] = 32;
        filterSizes[2] = 3;
        filterSizes[3] = 3;
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[4]);
        CreateWeightTensors(weights, "conv4/weights", "conv4/BatchNorm/scale", "conv4/BatchNorm/shift", 
            filterSizes, weightUploadBatch, &m_modelConvFilterWeights[4], &m_modelConvBiasWeights[4]);
        inputIndex = 1 - inputIndex;
        
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, true, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[5]);
        CreateWeightTensors(weights, "conv5/weights", "conv5/BatchNorm/scale", "conv5/BatchNorm/shift", 
            filterSizes, weightUploadBatch, &m_modelConvFilterWeights[5], &m_modelConvBiasWeights[5]);
        inputIndex = 1 - inputIndex;

        filterSizes[0] = 3;
        CreateConvolutionLayer(intermediateInputSizes[inputIndex], filterSizes, false, &intermediateBufferMaxSize[inputIndex],
            &intermediateBufferMaxSize[1 - inputIndex], intermediateInputSizes[1 - inputIndex], &m_dmlConvOps[6]);
        CreateWeightTensors(weights, "conv6/weights", nullptr, nullptr, filterSizes, weightUploadBatch,
            &m_modelConvFilterWeights[6], nullptr);
        inputIndex = 1 - inputIndex;
    
        // Finally add the residual to the original upsampled image
        assert(memcmp(upscaledInputSizes, intermediateInputSizes[inputIndex], 4 * sizeof(uint16_t)) == 0);

        CreateAdditionLayer(upscaledInputSizes, &m_dmlAddResidualOp);

        weightUploadBatch.End(m_deviceResources->GetCommandQueue());
    }

    // Buffers for DML inputs and outputs
    {
        // Resource for input tensor
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(modelInputBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelInput)
        ));

        // Describe and create a UAV for the original input tensor.
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = static_cast<UINT>(modelInputBufferSize / sizeof(uint16_t));
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        device->CreateUnorderedAccessView(m_modelInput.Get(), nullptr, &uavDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descModelInput));

        // Model result tensor is 2x larger in both dimensions
        resourceDesc.Width = modelOutputBufferSize;
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelOutput)
        ));

        // Describe and create a SRV for the final result tensor.
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = static_cast<UINT>(modelOutputBufferSize / sizeof(uint16_t));
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        device->CreateShaderResourceView(m_modelOutput.Get(), &srvDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descModelOutput));

        // Create two resources for intermediate layer results. Each layer will ping-pong between these. They're each large
        // enough to hold the largest intermediate result required.
        for (int i = 0; i < 2; i++)
        {
            resourceDesc.Width = intermediateBufferMaxSize[i];
            DX::ThrowIfFailed(device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&m_modelIntermediateResult[i])
            ));
        }
    }
    
    // Wait until assets have been uploaded to the GPU.
    m_deviceResources->WaitForGpu();
}

void Sample::CreateUpsampleLayer(
    _In_reads_(4) const uint32_t* inputSizes,
    _Inout_updates_(1) uint64_t* inputBufferRequiredSize,
    _Inout_updates_(1) uint64_t* outputBufferRequiredSize,
    _Out_writes_(4) uint32_t* outputSizesOut,
    _Out_writes_(1) IDMLCompiledOperator** compiledOpOut)
{
    // Describe input and output tensors
    uint32_t inputStrides[4];
    GetStrides(inputSizes, m_tensorLayout, inputStrides);
    
    uint64_t inputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, inputSizes, inputStrides);
    // Because we can resuse resources for tensor storage, this tracks the resource size needed to hold the
    // largest possible tensor requested.
    *inputBufferRequiredSize = std::max(inputBufferSize, *inputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC inputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, inputSizes, inputStrides, inputBufferSize, 0 };
    DML_TENSOR_DESC inputDesc = { DML_TENSOR_TYPE_BUFFER, &inputBufferDesc };

    // Output size is double in height and width
    outputSizesOut[0] = inputSizes[0];
    outputSizesOut[1] = inputSizes[1];
    outputSizesOut[2] = inputSizes[2] * 2;
    outputSizesOut[3] = inputSizes[3] * 2;

    uint32_t outputStrides[4];
    GetStrides(outputSizesOut, m_tensorLayout, outputStrides);

    uint64_t outputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, outputSizesOut, outputStrides);
    *outputBufferRequiredSize = std::max(outputBufferSize, *outputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC outputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, outputSizesOut, outputStrides, outputBufferSize, 0 };
    DML_TENSOR_DESC outputDesc = { DML_TENSOR_TYPE_BUFFER, &outputBufferDesc };

    // Describe, create, and compile upsample operator
    DML_UPSAMPLE_2D_OPERATOR_DESC upsampleDesc = { &inputDesc, &outputDesc, {2, 2}, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR };
    DML_OPERATOR_DESC opDesc = { DML_OPERATOR_UPSAMPLE_2D, &upsampleDesc };

    ComPtr<IDMLOperator> op;
    DX::ThrowIfFailed(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(op.ReleaseAndGetAddressOf())));
    DX::ThrowIfFailed(m_dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(compiledOpOut)));
}

void Sample::CreateConvolutionLayer(
    _In_reads_(4) const uint32_t* inputSizes,
    _In_reads_(4) const uint32_t* filterSizes,
    bool useBiasAndActivation,
    _Inout_updates_(1) uint64_t* inputBufferRequiredSize,
    _Inout_updates_(1) uint64_t* outputBufferRequiredSize,
    _Out_writes_(4) uint32_t* outputSizesOut,
    _Out_writes_(1) IDMLCompiledOperator** compiledOpOut)
{
    // Describe input and output tensors    
    uint32_t inputStrides[4];
    GetStrides(inputSizes, m_tensorLayout, inputStrides);

    uint64_t inputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, inputSizes, inputStrides);
    *inputBufferRequiredSize = std::max(inputBufferSize, *inputBufferRequiredSize);

    DML_BUFFER_TENSOR_DESC inputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, inputSizes, inputStrides, inputBufferSize, 0 };
    DML_TENSOR_DESC inputDesc = { DML_TENSOR_TYPE_BUFFER, &inputBufferDesc };

    // The output shape has as many channels as there are convolution filters.
    outputSizesOut[0] = inputSizes[0];
    outputSizesOut[1] = filterSizes[0];
    outputSizesOut[2] = inputSizes[2];
    outputSizesOut[3] = inputSizes[3];

    uint32_t outputStrides[4];
    GetStrides(outputSizesOut, m_tensorLayout, outputStrides);

    uint64_t outputBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, outputSizesOut, outputStrides);
    *outputBufferRequiredSize = std::max(outputBufferSize, *outputBufferRequiredSize);    

    DML_BUFFER_TENSOR_DESC outputBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, outputSizesOut, outputStrides, outputBufferSize, 0 };
    DML_TENSOR_DESC outputDesc = { DML_TENSOR_TYPE_BUFFER, &outputBufferDesc };
    
    // Describe weight tensors
    uint32_t filterStrides[4];
    GetStrides(filterSizes, m_tensorLayout, filterStrides);
    uint64_t filterBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, filterSizes, filterStrides);

#if DML_MANAGED_WEIGHTS
    DML_BUFFER_TENSOR_DESC filterBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_OWNED_BY_DML, 4, filterSizes, filterStrides, filterBufferSize, 0 };
#else
    DML_BUFFER_TENSOR_DESC filterBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, filterSizes, filterStrides, filterBufferSize, 0 };
#endif
    DML_TENSOR_DESC filterDesc = { DML_TENSOR_TYPE_BUFFER, &filterBufferDesc };

    uint32_t biasSizes[] = { 1, filterSizes[0], 1, 1 };	// One bias per output channel    
    uint32_t biasStrides[4];
    GetStrides(biasSizes, m_tensorLayout, biasStrides);    
    uint64_t biasBufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, biasSizes, biasStrides);

#if DML_MANAGED_WEIGHTS
    DML_BUFFER_TENSOR_DESC biasBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_OWNED_BY_DML, 4, biasSizes, biasStrides, biasBufferSize, 0 };
#else
    DML_BUFFER_TENSOR_DESC biasBufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, biasSizes, biasStrides, biasBufferSize, 0 };
#endif
    DML_TENSOR_DESC biasDesc = { DML_TENSOR_TYPE_BUFFER, &biasBufferDesc };

    // Describe, create, and compile convolution operator

    // The output size of a convolution operation is given by:
    //  height = (inputHeight - filterHeight + 2*paddingHeight) / filterStride + 1
    //  width  = (inputWidth  - filterWidth  + 2*paddingWidth ) / filterStride + 1
    //
    // We want to preserve the height and width, so assuming stride is 1, we get:
    //  paddingHeight = (filterHeight - 1) / 2
    //  paddingWidth  = (filterWidth  - 1) / 2
    // If padding is fractional, we pad unevenly with ceil/floor.
    UINT paddingHeightTop = static_cast<UINT>(ceil((filterSizes[2] - 1) / 2.0f));
    UINT paddingHeightBottom = static_cast<UINT>(floor((filterSizes[2] - 1) / 2.0f));
    UINT paddingWidthLeft = static_cast<UINT>(ceil((filterSizes[3] - 1) / 2.0f));
    UINT paddingWidthRight = static_cast<UINT>(floor((filterSizes[3] - 1) / 2.0f));
    
    UINT strides[] = { 1, 1 };
    UINT dilations[] = { 1, 1 };
    UINT startPadding[] = { paddingHeightTop, paddingWidthLeft };
    UINT endPadding[] = { paddingHeightBottom, paddingWidthRight };
    UINT outputPadding[] = { 0, 0 };

    DML_ACTIVATION_RELU_OPERATOR_DESC fusedReluDesc = { 0 };
    DML_OPERATOR_DESC activationDesc = { DML_OPERATOR_ACTIVATION_RELU, &fusedReluDesc };

    DML_CONVOLUTION_OPERATOR_DESC convDesc = {
        &inputDesc,
        &filterDesc,
        useBiasAndActivation ? &biasDesc : nullptr,
        &outputDesc,
        DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        DML_CONVOLUTION_DIRECTION_FORWARD,
        2,
        strides,
        dilations,
        startPadding,
        endPadding,
        outputPadding,
        1,
        useBiasAndActivation ? &activationDesc : nullptr
    };
    DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CONVOLUTION, &convDesc };

    ComPtr<IDMLOperator> op;
    DX::ThrowIfFailed(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(op.ReleaseAndGetAddressOf())));
    DX::ThrowIfFailed(m_dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(compiledOpOut)));
}

void Sample::CreateAdditionLayer(
    _In_reads_(4) const uint32_t* inputSizes,
    _Out_writes_(1) IDMLCompiledOperator** compiledOpOut)
{
    // Describe input and output tensors
    uint32_t strides[4];
    GetStrides(inputSizes, m_tensorLayout, strides);
    uint64_t bufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, inputSizes, strides);

    DML_BUFFER_TENSOR_DESC bufferDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, inputSizes, strides, bufferSize, 0 };
    DML_TENSOR_DESC tensorDesc = { DML_TENSOR_TYPE_BUFFER, &bufferDesc };

    // Describe, create, and compile elementwise addition operator
    // Inputs and output are all the same size and use the same tensor desc.
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc = { &tensorDesc, &tensorDesc, &tensorDesc };
    DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc };

    ComPtr<IDMLOperator> op;
    DX::ThrowIfFailed(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(op.ReleaseAndGetAddressOf())));
    DX::ThrowIfFailed(m_dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(compiledOpOut)));
}

void Sample::CreateWeightTensors(
    WeightMapType& weights,
    const char* convLayerName,
    const char* scaleLayerName,
    const char* shiftLayerName,
    _In_reads_(4) const uint32_t* filterSizes,
    DirectX::ResourceUploadBatch& uploadBatch,
    _Out_writes_(1) ID3D12Resource** filterWeightResourceOut,
    _Out_writes_opt_(1) ID3D12Resource** biasWeightResourceOut)
{
    // There are two types of weights for the convolutions: The convolution filters themselves, and scale/shift
    // weights used to normalize and bias the results. The final layer doesn't use scale and shift weights, so
    // these are optional.

    bool useScaleShift = true;
    if (scaleLayerName == nullptr)
    {
        assert(shiftLayerName == nullptr);
        useScaleShift = false;
    }
    
    CreateWeightResource(filterSizes, filterWeightResourceOut);
    if (useScaleShift)
    {
        uint32_t biasSizes[] = { 1, filterSizes[0], 1, 1 };	// One bias per output channel
        CreateWeightResource(biasSizes, biasWeightResourceOut);

        // The scale weights will be premultiplied into the filter weights, so they don't need
        // a separate resource.
    }
    else
    {
        if (biasWeightResourceOut)
            biasWeightResourceOut = nullptr;
    }

    // Convert weight values to FP16
    WeightsType filterWeights = weights[convLayerName];
    WeightsType scaleWeights, shiftWeights;
    if (useScaleShift)
    {
        scaleWeights = weights[scaleLayerName];
        shiftWeights = weights[shiftLayerName];
    }

    std::vector<uint16_t> filterWeightsFP16;
    std::vector<uint16_t> biasWeightsFP16;

    const uint32_t N = filterSizes[0];
    const uint32_t C = filterSizes[1];
    const uint32_t H = filterSizes[2];
    const uint32_t W = filterSizes[3];

    for (uint32_t n = 0; n < N; n++)
    {
        switch (m_tensorLayout)
        {
        case TensorLayout::NHWC:
            // We need to convert the weights from NCHW to NHWC.
            for (uint32_t h = 0; h < H; h++)
                for (uint32_t w = 0; w < W; w++)
                    for (uint32_t c = 0; c < C; c++)
                    {
                        // Apply the scale weight now so we don't need a normalization layer
                        uint32_t idx = w + h * W + c * H*W + n * C*H*W;
                        float scaledWeight = useScaleShift ?
                            filterWeights[idx] * scaleWeights[n] :
                            filterWeights[idx];
                        filterWeightsFP16.push_back(Float16Compressor::compress(scaledWeight));
                    }
            break;

        default:
            // Weights are already in the right order
            for (uint32_t i = 0; i < C*H*W; i++)
            {
                // Apply the scale weight now so we don't need a normalization layer
                uint32_t idx = n * C*H*W + i;
                float scaledWeight = useScaleShift ?
                    filterWeights[idx] * scaleWeights[n] :
                    filterWeights[idx];
                filterWeightsFP16.push_back(Float16Compressor::compress(scaledWeight));
            }
        }

        if (useScaleShift)
        {
            // Technically this is initialBias*scale+shift, but the initial bias is 0
            biasWeightsFP16.push_back(Float16Compressor::compress(shiftWeights[n]));
        }
    }

    // Upload to the GPU
    D3D12_SUBRESOURCE_DATA weightsData = {};
    weightsData.pData = filterWeightsFP16.data();
    uploadBatch.Upload(*filterWeightResourceOut, 0, &weightsData, 1);

    if (useScaleShift)
    {
        weightsData.pData = biasWeightsFP16.data();
        uploadBatch.Upload(*biasWeightResourceOut, 0, &weightsData, 1);
    }
}

void Sample::CreateWeightResource(
    _In_reads_(4) const uint32_t* tensorSizes,
    _Out_writes_(1) ID3D12Resource** d3dResourceOut)
{
    uint32_t strides[4];
    GetStrides(tensorSizes, m_tensorLayout, strides);
    uint64_t bufferSize = DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE_FLOAT16, 4, tensorSizes, strides);

    D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(d3dResourceOut)
    ));
}

void Sample::InitializeDirectMLResources()
{
    auto commandList = m_deviceResources->GetCommandList();
    commandList->Reset(m_deviceResources->GetCommandAllocator(), nullptr);

    // Create operator initializers and descriptor heap for binding
    size_t upsampleOpDescriptorCount, convOpDescriptorCount, additionOpDescriptorCount;
    size_t upsampleDescriptorsIdx, convDescriptorsIdx, additionDescriptorsIdx;

    {
        // The same descriptor heap will be used for both initializing and executing operators. These each happen
        // at different times, so we reuse the same descriptor slots. GetDescriptorCount() ensures there are enough
        // slots for both cases.
        DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(c_numUpsampleLayers, m_dmlUpsampleOps[0].GetAddressOf(), IID_PPV_ARGS(m_dmlOpInitializers[e_opUpsample].GetAddressOf())));
        upsampleOpDescriptorCount = GetDescriptorCount(c_numUpsampleLayers, m_dmlUpsampleOps[0].GetAddressOf(), m_dmlOpInitializers[e_opUpsample].Get());

        DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(c_numConvLayers, m_dmlConvOps[0].GetAddressOf(), IID_PPV_ARGS(m_dmlOpInitializers[e_opConv].GetAddressOf())));
        convOpDescriptorCount = GetDescriptorCount(c_numConvLayers, m_dmlConvOps[0].GetAddressOf(), m_dmlOpInitializers[e_opConv].Get());

        DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(1, m_dmlAddResidualOp.GetAddressOf(), IID_PPV_ARGS(m_dmlOpInitializers[e_opAdd].GetAddressOf())));
        additionOpDescriptorCount = GetDescriptorCount(1, m_dmlAddResidualOp.GetAddressOf(), m_dmlOpInitializers[e_opAdd].Get());
        
        upsampleDescriptorsIdx = 0;
        convDescriptorsIdx = upsampleDescriptorsIdx + upsampleOpDescriptorCount * c_numUpsampleLayers;
        additionDescriptorsIdx = convDescriptorsIdx + convOpDescriptorCount * c_numConvLayers;
        size_t descriptorCount = additionDescriptorsIdx + additionOpDescriptorCount;

        m_dmlDescriptorHeap = std::make_unique<DescriptorHeap>(m_deviceResources->GetD3DDevice(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            descriptorCount);

        // Operator initialization dispatches will use this heap right away
        ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);
    }

    // Create any persistent resources required for the operators.
    {
        for (int i = 0; i < c_numUpsampleLayers + c_numConvLayers + 1; i++)
        {
            IDMLCompiledOperator* currentOp;
            ID3D12Resource** persistentResource;
            if (i < c_numUpsampleLayers)
            {
                currentOp = m_dmlUpsampleOps[i].Get();
                persistentResource = m_modelUpsamplePersistentResources[i].ReleaseAndGetAddressOf();
            }
            else if (i < c_numUpsampleLayers + c_numConvLayers)
            {
                currentOp = m_dmlConvOps[i - c_numUpsampleLayers].Get();
                persistentResource = m_modelConvPersistentResources[i - c_numUpsampleLayers].ReleaseAndGetAddressOf();
            }
            else
            {
                currentOp = m_dmlAddResidualOp.Get();
                persistentResource = m_modelAddPersistentResource.ReleaseAndGetAddressOf();
            }

            auto bindingProps = currentOp->GetBindingProperties();

            if (bindingProps.PersistentResourceSize > 0)
            {
                D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bindingProps.PersistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
                    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &resourceDesc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(persistentResource)));
            }
        }
    }

    // When binding input and output resources, take note of which temp resource is used at the time:
    // Layer		| Input							| Output
    // Upsample[0]	| m_modelInput					| m_modelOutput
    // Conv[0]		| m_modelInput					| m_modelIntermediateResult[0]
    // Conv[1]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Conv[2]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
    // Upsample[1]	| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Conv[3]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
    // Conv[4]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Conv[5]		| m_modelIntermediateResult[1]	| m_modelIntermediateResult[0]
    // Conv[6]		| m_modelIntermediateResult[0]	| m_modelIntermediateResult[1]
    // Addition		| m_modelIntermediateResult[1], m_modelOutput | m_modelOutput
    
    const DML_BUFFER_BINDING emptyBufferBinding = { nullptr, 0, 0 };
    const DML_BINDING_DESC emptyBindingDesc = { DML_BINDING_TYPE_NONE, nullptr };

    // Bind D3D resources

    Microsoft::WRL::ComPtr<IDMLBindingTable> initBindingTable;

    // Upsample layers
    {
        // Bind resources for initialization.
        auto bindingProps = m_dmlOpInitializers[e_opUpsample]->GetBindingProperties();
        // The DML API guarantees that initialization never uses a persistent resource.
        assert(bindingProps.PersistentResourceSize == 0);
        
        DML_BINDING_TABLE_DESC tableDesc = {
            m_dmlOpInitializers[e_opUpsample].Get(),
            m_dmlDescriptorHeap->GetCpuHandle(upsampleDescriptorsIdx),
            m_dmlDescriptorHeap->GetGpuHandle(upsampleDescriptorsIdx),
            bindingProps.RequiredDescriptorCount
        };
        DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable)));
        
        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        DML_BUFFER_BINDING upsamplePersistentBuffers[c_numUpsampleLayers];
        DML_BINDING_DESC upsamplePersistentBindings[c_numUpsampleLayers];
        for (int i = 0; i < c_numUpsampleLayers; i++)
        {
            if (m_modelUpsamplePersistentResources[i].Get() != nullptr)
            {
                upsamplePersistentBuffers[i] = { m_modelUpsamplePersistentResources[i].Get(), 0, m_modelUpsamplePersistentResources[i]->GetDesc().Width };
                upsamplePersistentBindings[i] = { DML_BINDING_TYPE_BUFFER, &upsamplePersistentBuffers[i] };
            }
            else
                upsamplePersistentBindings[i] = emptyBindingDesc;
        }

        // The inputs will vary each frame, so don't bind inputs at initialization.
        initBindingTable->BindInputs(0, nullptr);
        initBindingTable->BindOutputs(c_numUpsampleLayers, upsamplePersistentBindings);
        BindTempResourceIfNeeded(bindingProps, initBindingTable.Get(), m_modelInitTemporaryResources[e_opUpsample].ReleaseAndGetAddressOf());

        // Run initialization
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializers[e_opUpsample].Get(), initBindingTable.Get());

        // Bind resources for execution
        for (int i = 0; i < c_numUpsampleLayers; i++)
        {
            bindingProps = m_dmlUpsampleOps[i]->GetBindingProperties();

            tableDesc = {
                m_dmlUpsampleOps[i].Get(),
                m_dmlDescriptorHeap->GetCpuHandle(upsampleDescriptorsIdx + i * upsampleOpDescriptorCount),
                m_dmlDescriptorHeap->GetGpuHandle(upsampleDescriptorsIdx + i * upsampleOpDescriptorCount),
                bindingProps.RequiredDescriptorCount
            };
            DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(m_dmlUpsampleBindings[i].ReleaseAndGetAddressOf())));

            auto inputResource = (i == 0) ? m_modelInput : m_modelIntermediateResult[0];
            auto outputResource = (i == 0) ? m_modelOutput : m_modelIntermediateResult[1];

            DML_BUFFER_BINDING inputBufferBinding = { inputResource.Get(), 0, inputResource->GetDesc().Width };
            DML_BINDING_DESC inputBinding = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
            DML_BUFFER_BINDING outputBufferBinding = { outputResource.Get(), 0, outputResource->GetDesc().Width };
            DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBufferBinding };

            m_dmlUpsampleBindings[i]->BindInputs(1, &inputBinding);
            m_dmlUpsampleBindings[i]->BindOutputs(1, &outputBinding);
            BindTempResourceIfNeeded(bindingProps, m_dmlUpsampleBindings[i].Get(), m_modelUpsampleTemporaryResources[i].ReleaseAndGetAddressOf());

            if (m_modelUpsamplePersistentResources[i].Get() != nullptr)
                m_dmlUpsampleBindings[i]->BindPersistentResource(&upsamplePersistentBindings[i]);
        }
    }

    // Convolution layers
    {
        // Bind resources for initialization
        auto bindingProps = m_dmlOpInitializers[e_opConv]->GetBindingProperties();
        assert(bindingProps.PersistentResourceSize == 0);

        DML_BINDING_TABLE_DESC tableDesc = {
            m_dmlOpInitializers[e_opConv].Get(),
            m_dmlDescriptorHeap->GetCpuHandle(convDescriptorsIdx),
            m_dmlDescriptorHeap->GetGpuHandle(convDescriptorsIdx),
            bindingProps.RequiredDescriptorCount
        };
        DX::ThrowIfFailed(initBindingTable->Reset(&tableDesc));

#if DML_MANAGED_WEIGHTS
        // Bind the weight tensors at initialization instead of at execution. This lets DirectML reformat them
        // and improve performance on some hardware.
        DML_BUFFER_BINDING convBufferBindings[][3] = {
            { emptyBufferBinding, { m_modelConvFilterWeights[0].Get(), 0, m_modelConvFilterWeights[0]->GetDesc().Width }, { m_modelConvBiasWeights[0].Get(), 0, m_modelConvBiasWeights[0]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[1].Get(), 0, m_modelConvFilterWeights[1]->GetDesc().Width }, { m_modelConvBiasWeights[1].Get(), 0, m_modelConvBiasWeights[1]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[2].Get(), 0, m_modelConvFilterWeights[2]->GetDesc().Width }, { m_modelConvBiasWeights[2].Get(), 0, m_modelConvBiasWeights[2]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[3].Get(), 0, m_modelConvFilterWeights[3]->GetDesc().Width }, { m_modelConvBiasWeights[3].Get(), 0, m_modelConvBiasWeights[3]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[4].Get(), 0, m_modelConvFilterWeights[4]->GetDesc().Width }, { m_modelConvBiasWeights[4].Get(), 0, m_modelConvBiasWeights[4]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[5].Get(), 0, m_modelConvFilterWeights[5]->GetDesc().Width }, { m_modelConvBiasWeights[5].Get(), 0, m_modelConvBiasWeights[5]->GetDesc().Width } },
            { emptyBufferBinding, { m_modelConvFilterWeights[6].Get(), 0, m_modelConvFilterWeights[6]->GetDesc().Width }, emptyBufferBinding }	// last layer has no bias
        };

        DML_BUFFER_ARRAY_BINDING convBufferArrayBindings[] = {
            { 3, convBufferBindings[0] },
            { 3, convBufferBindings[1] },
            { 3, convBufferBindings[2] },
            { 3, convBufferBindings[3] },
            { 3, convBufferBindings[4] },
            { 3, convBufferBindings[5] },
            { 3, convBufferBindings[6] },
        };

        DML_BINDING_DESC convInBindings[] = {
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[0] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[1] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[2] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[3] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[4] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[5] },
            { DML_BINDING_TYPE_BUFFER_ARRAY, &convBufferArrayBindings[6] }
        };

        initBindingTable->BindInputs(c_numConvLayers, convInBindings);
#else
        initBindingTable->BindInputs(0, nullptr);
#endif

        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        DML_BUFFER_BINDING convPersistentBuffers[c_numConvLayers];
        DML_BINDING_DESC convPersistentBindings[c_numConvLayers];
        for (int i = 0; i < c_numConvLayers; i++)
        {
            if (m_modelConvPersistentResources[i].Get() != nullptr)
            {
                convPersistentBuffers[i] = { m_modelConvPersistentResources[i].Get(), 0, m_modelConvPersistentResources[i]->GetDesc().Width };
                convPersistentBindings[i] = { DML_BINDING_TYPE_BUFFER, &convPersistentBuffers[i] };
            }
            else
                convPersistentBindings[i] = emptyBindingDesc;
        }

        initBindingTable->BindOutputs(c_numConvLayers, convPersistentBindings);
        BindTempResourceIfNeeded(bindingProps, initBindingTable.Get(), m_modelInitTemporaryResources[e_opConv].ReleaseAndGetAddressOf());

        // Run initialization
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializers[e_opConv].Get(), initBindingTable.Get());

        // Bind resources for execution
        for (int i = 0; i < c_numConvLayers; i++)
        {
            bindingProps = m_dmlConvOps[i]->GetBindingProperties();

            tableDesc = {
                m_dmlConvOps[i].Get(),
                m_dmlDescriptorHeap->GetCpuHandle(convDescriptorsIdx + i * convOpDescriptorCount),
                m_dmlDescriptorHeap->GetGpuHandle(convDescriptorsIdx + i * convOpDescriptorCount),
                bindingProps.RequiredDescriptorCount
            };
            DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(m_dmlConvBindings[i].ReleaseAndGetAddressOf())));

            // See table at the beginning of the function for the mapping of ops to resources.
            auto inputResource = (i == 0) ? m_modelInput : ((i == 1 || i == 4 || i == 6) ? m_modelIntermediateResult[0] : m_modelIntermediateResult[1]);
            auto outputResource = (i == 1 || i == 4 || i == 6) ? m_modelIntermediateResult[1] : m_modelIntermediateResult[0];

            DML_BUFFER_BINDING inputBufferBinding = { inputResource.Get(), 0, inputResource->GetDesc().Width };
            DML_BINDING_DESC inputBinding = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };

            DML_BUFFER_BINDING outputBufferBinding = { outputResource.Get(), 0, outputResource->GetDesc().Width };
            DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBufferBinding };

#if DML_MANAGED_WEIGHTS
            // The weights are stored in the persistent resource and shouldn't be bound separately.
            DML_BINDING_DESC inputBindings[] = { inputBinding, emptyBindingDesc, emptyBindingDesc };
#else
            // Bind the weight resources
            DML_BUFFER_BINDING filterBufferBinding = { m_modelConvFilterWeights[i].Get(), 0, m_modelConvFilterWeights[i]->GetDesc().Width };
            DML_BINDING_DESC filterBinding = { DML_BINDING_TYPE_BUFFER, &filterBufferBinding };

            DML_BUFFER_BINDING biasBufferBinding;
            DML_BINDING_DESC biasBinding;
            if (i == 6)
            {
                biasBinding = emptyBindingDesc;	// last layer has no bias
            }
            else
            {
                biasBufferBinding = { m_modelConvBiasWeights[i].Get(), 0, m_modelConvBiasWeights[i]->GetDesc().Width };
                biasBinding = { DML_BINDING_TYPE_BUFFER, &biasBufferBinding };
            }

            DML_BINDING_DESC inputBindings[] = { inputBinding, filterBinding, biasBinding };
#endif
            m_dmlConvBindings[i]->BindInputs(3, inputBindings);
            m_dmlConvBindings[i]->BindOutputs(1, &outputBinding);
            BindTempResourceIfNeeded(bindingProps, m_dmlConvBindings[i].Get(), m_modelConvTemporaryResources[i].ReleaseAndGetAddressOf());

            if (m_modelConvPersistentResources[i].Get() != nullptr)
                m_dmlConvBindings[i]->BindPersistentResource(&convPersistentBindings[i]);
        }
    }

    // Addition layer
    {
        // Bind resources for initialization.
        auto bindingProps = m_dmlOpInitializers[e_opAdd]->GetBindingProperties();
        assert(bindingProps.PersistentResourceSize == 0);

        DML_BINDING_TABLE_DESC tableDesc = {
            m_dmlOpInitializers[e_opAdd].Get(),
            m_dmlDescriptorHeap->GetCpuHandle(additionDescriptorsIdx),
            m_dmlDescriptorHeap->GetGpuHandle(additionDescriptorsIdx),
            bindingProps.RequiredDescriptorCount
        };
        DX::ThrowIfFailed(initBindingTable->Reset(&tableDesc));
                
        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        DML_BUFFER_BINDING addPersistentBuffer;
        DML_BINDING_DESC addPersistentBinding;
        if (m_modelAddPersistentResource.Get() != nullptr)
        {
            addPersistentBuffer = { m_modelAddPersistentResource.Get(), 0, m_modelAddPersistentResource->GetDesc().Width };
            addPersistentBinding = { DML_BINDING_TYPE_BUFFER, &addPersistentBuffer };
        }
        else
            addPersistentBinding = emptyBindingDesc;

        initBindingTable->BindInputs(0, nullptr);
        initBindingTable->BindOutputs(1, &addPersistentBinding);
        BindTempResourceIfNeeded(bindingProps, initBindingTable.Get(), m_modelInitTemporaryResources[e_opAdd].ReleaseAndGetAddressOf());

        // Run initialization
        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializers[e_opAdd].Get(), initBindingTable.Get());

        // Bind resources for execution
        {
            bindingProps = m_dmlAddResidualOp->GetBindingProperties();

            tableDesc = {
                m_dmlAddResidualOp.Get(),
                m_dmlDescriptorHeap->GetCpuHandle(additionDescriptorsIdx),
                m_dmlDescriptorHeap->GetGpuHandle(additionDescriptorsIdx),
                bindingProps.RequiredDescriptorCount
            };
            DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(m_dmlAddResidualBinding.ReleaseAndGetAddressOf())));

            // m_modelOutput will already hold the result of the first upsample operation. We add the result of
            // the last convolution (the residual) to it in-place to get the final result.
            DML_BUFFER_BINDING input0BufferBinding = { m_modelIntermediateResult[1].Get(), 0, m_modelIntermediateResult[1]->GetDesc().Width };
            DML_BINDING_DESC input0Binding = { DML_BINDING_TYPE_BUFFER, &input0BufferBinding };
            DML_BUFFER_BINDING input1BufferBinding = { m_modelOutput.Get(), 0, m_modelOutput->GetDesc().Width };
            DML_BINDING_DESC input1Binding = { DML_BINDING_TYPE_BUFFER, &input1BufferBinding };
            DML_BUFFER_BINDING outputBufferBinding = { m_modelOutput.Get(), 0, m_modelOutput->GetDesc().Width };
            DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBufferBinding };

            DML_BINDING_DESC inputBindings[] = { input0Binding, input1Binding };
            m_dmlAddResidualBinding->BindInputs(2, inputBindings);
            m_dmlAddResidualBinding->BindOutputs(1, &outputBinding);
            BindTempResourceIfNeeded(bindingProps, m_dmlAddResidualBinding.Get(), m_modelAddTemporaryResource.ReleaseAndGetAddressOf());

            if (m_modelAddPersistentResource.Get() != nullptr)
                m_dmlAddResidualBinding->BindPersistentResource(&addPersistentBinding);
        }
    }

    DX::ThrowIfFailed(commandList->Close());
    m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    // Wait until initialization has been finished on the GPU.
    m_deviceResources->WaitForGpu();

#if DML_MANAGED_WEIGHTS
    // These have been copied to DML-managed resources and are no longer needed.
    for (int i = 0; i < c_numConvLayers; i++)
    {
        m_modelConvFilterWeights[i].Reset();
        if (i < c_numConvLayers - 1)    // Last layer has no bias
        {
            m_modelConvBiasWeights[i].Reset();
        }
    }
#endif
}

void Sample::BindTempResourceIfNeeded(DML_BINDING_PROPERTIES& bindingProps, IDMLBindingTable* initBindingTable, ID3D12Resource** tempResource)
{
    if (bindingProps.TemporaryResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bindingProps.TemporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(tempResource)));

        DML_BUFFER_BINDING tempBuffer = { *tempResource, 0, (*tempResource)->GetDesc().Width };
        DML_BINDING_DESC tempBinding = { DML_BINDING_TYPE_BUFFER, &tempBuffer };
        initBindingTable->BindTemporaryResource(&tempBinding);
    }
}

void Sample::CreateUIResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    
    m_lineBatch = std::make_unique<PrimitiveBatch<VertexPositionColor>>(device);

    RenderTargetState rtState(m_deviceResources->GetBackBufferFormat(), m_deviceResources->GetDepthBufferFormat());
    EffectPipelineStateDescription epd(&VertexPositionColor::InputLayout, CommonStates::AlphaBlend,
        CommonStates::DepthDefault, CommonStates::CullNone, rtState, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
    m_lineEffect = std::make_unique<BasicEffect>(device, EffectFlags::VertexColor, epd);

    SpriteBatchPipelineStateDescription spd(rtState, &CommonStates::AlphaBlend);
    ResourceUploadBatch uploadBatch(device);
    uploadBatch.Begin();

    m_spriteBatch = std::make_unique<SpriteBatch>(device, uploadBatch, spd);

    wchar_t strFilePath[MAX_PATH] = {};
    DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_30.spritefont");
    m_labelFont = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descLabelFont),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descLabelFont));

    DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_30_Bold.spritefont");
    m_labelFontBold = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descLabelFontBold),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descLabelFontBold));

    DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_18.spritefont");
    m_legendFont = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descLegendFont),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descLegendFont));

    DX::FindMediaFile(strFilePath, MAX_PATH, L"XboxOneControllerLegendSmall.spritefont");
    m_ctrlFont = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descCtrlFont),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descCtrlFont));

    auto finish = uploadBatch.End(m_deviceResources->GetCommandQueue());
    finish.wait();
}

// Allocate all memory resources that change on a window SizeChanged event.
void Sample::CreateWindowSizeDependentResources()
{
    auto viewport = m_deviceResources->GetScreenViewport();

    auto proj = DirectX::SimpleMath::Matrix::CreateOrthographicOffCenter(0.f, static_cast<float>(viewport.Width),
        static_cast<float>(viewport.Height), 0.f, 0.f, 1.f);
    m_lineEffect->SetProjection(proj);

    m_spriteBatch->SetViewport(viewport);
}

void Sample::OnDeviceLost()
{
    m_lineEffect.reset();
    m_lineBatch.reset();
    m_spriteBatch.reset();
    m_labelFont.reset();
    m_labelFontBold.reset();
    m_legendFont.reset();
    m_ctrlFont.reset();
    m_fontDescriptorHeap.reset();

    m_player.reset();

    m_texPipelineStateNN.Reset();
    m_texPipelineStateLinear.Reset();
    m_texRootSignatureNN.Reset();
    m_texRootSignatureLinear.Reset();
    m_tensorRenderPipelineState.Reset();
    m_tensorRenderRootSignature.Reset();
    m_texture.Reset();
    m_videoTexture.Reset();
    m_finalResultTexture.Reset();
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();

    m_SRVDescriptorHeap.reset();
    m_RTVDescriptorHeap.reset();
    m_zoomedVertexHeap.Reset();

    m_computePSO.Reset();
    m_computeRootSignature.Reset();

    m_dmlDevice.Reset();
    m_dmlCommandRecorder.Reset();

    m_modelInput.Reset();
    m_modelOutput.Reset();
    for (int i = 0; i < c_numIntermediateBuffers; i++)
    {
        m_modelIntermediateResult[i].Reset();
    }
    
    for (int i = 0; i < e_opCount; i++)
    {
        m_dmlOpInitializers[i].Reset();
        m_modelInitTemporaryResources[i].Reset();
    }
    for (int i = 0; i < c_numUpsampleLayers; i++)
    {
        m_dmlUpsampleOps[i].Reset();
        m_modelUpsamplePersistentResources[i].Reset();
        m_modelUpsampleTemporaryResources[i].Reset();
        m_dmlUpsampleBindings[i].Reset();
    }
    for (int i = 0; i < c_numConvLayers; i++)
    {
        m_modelConvFilterWeights[i].Reset();
        m_modelConvBiasWeights[i].Reset();
        m_modelConvPersistentResources[i].Reset();
        m_modelConvTemporaryResources[i].Reset();
        m_dmlConvOps[i].Reset();
        m_dmlConvBindings[i].Reset();
    }
    m_dmlAddResidualOp.Reset();
    m_modelAddPersistentResource.Reset();
    m_modelAddTemporaryResource.Reset();
    m_dmlAddResidualBinding.Reset();

    m_dmlDescriptorHeap.reset();

    m_graphicsMemory.reset();
}

void Sample::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion
