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

#if !(USE_DMLX)
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
#else 
            m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlGraph.Get(), m_dmlBindingTable.Get());
#endif
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
        m_SRVDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            e_srvDescCount);

        m_RTVDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            e_rtvDescCount);

        m_fontDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
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

void Sample::CreateWeightTensors(
    WeightMapType& weights,
    const char* convLayerName,
    const char* scaleLayerName,
    const char* shiftLayerName,
    dml::Span<const uint32_t> filterSizes,
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
    
    CreateWeightResource(filterSizes.data(), filterWeightResourceOut);
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

void Sample::GetStrides(
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
#if !(USE_DMLX)
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
#endif
    m_dmlOpInitializer.Reset();
    m_dmlGraph.Reset();
    m_modelTemporaryResource.Reset();
    m_modelPersistentResource.Reset();

    for (int i = 0; i < c_numConvLayers; i++)
    {
        m_modelConvFilterWeights[i].Reset();
        m_modelConvBiasWeights[i].Reset();
#if !(USE_DMLX)
        m_modelConvPersistentResources[i].Reset();
        m_modelConvTemporaryResources[i].Reset();
        m_dmlConvOps[i].Reset();
        m_dmlConvBindings[i].Reset();
#endif
    }
#if !(USE_DMLX)
    m_dmlAddResidualOp.Reset();
    m_modelAddPersistentResource.Reset();
    m_modelAddTemporaryResource.Reset();
    m_dmlAddResidualBinding.Reset();
#endif

    m_dmlDescriptorHeap.reset();

    m_graphicsMemory.reset();
}

void Sample::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion
