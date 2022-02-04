//--------------------------------------------------------------------------------------
// yolov4.cpp
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "pch.h"

#include "yolov4.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"

#include "TensorExtents.h"
#include "TensorUtil.h"
#include "TensorView.h"

const wchar_t* c_videoPath = L"grca-grand-canyon-association-park-store_1280x720.mp4";
const wchar_t* c_imagePath = L"grca-BA-bike-shop_1280x720.jpg";

extern void ExitSample();

using namespace DirectX;

using Microsoft::WRL::ComPtr;

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

    // Maps and copies the contents out of a readback heap.
    template <typename T>
    std::vector<T> CopyReadbackHeap(ID3D12Resource* readbackHeap)
    {
        static_assert(std::is_pod_v<T>);

        uint64_t sizeInBytes = readbackHeap->GetDesc().Width;
        size_t sizeInElements = static_cast<size_t>(sizeInBytes / sizeof(T));

        void* src;
        DX::ThrowIfFailed(readbackHeap->Map(0, nullptr, &src));

        std::vector<T> dst(sizeInElements);
        memcpy(dst.data(), src, sizeInElements * sizeof(T));

        readbackHeap->Unmap(0, nullptr);

        return dst;
    }

    // Returns true if any of the supplied floats are inf or NaN, false otherwise.
    static bool IsInfOrNan(dml::Span<const float> vals)
    {
        for (float val : vals)
        {
            if (std::isinf(val) || std::isnan(val))
            {
                return true;
            }
        }

        return false;
    }

    // Given two axis-aligned bounding boxes, computes the area of intersection divided by the area of the union of
    // the two boxes.
    static float ComputeIntersectionOverUnion(const Prediction& a, const Prediction& b)
    {
        float aArea = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        float bArea = (b.xmax - b.xmin) * (b.ymax - b.ymin);

        // Determine the bounds of the intersection rectangle
        float interXMin = std::max(a.xmin, b.xmin);
        float interYMin = std::max(a.ymin, b.ymin);
        float interXMax = std::min(a.xmax, b.xmax);
        float interYMax = std::min(a.ymax, b.ymax);

        float intersectionArea = std::max(0.0f, interXMax - interXMin) * std::max(0.0f, interYMax - interYMin);
        float unionArea = aArea + bArea - intersectionArea;

        return (intersectionArea / unionArea);
    }

    // Given a set of predictions, applies the non-maximal suppression (NMS) algorithm to select the "best" of
    // multiple overlapping predictions.
    static std::vector<Prediction> ApplyNonMaximalSuppression(dml::Span<const Prediction> allPredictions, float threshold)
    {
        std::unordered_map<uint32_t, std::vector<Prediction>> predsByClass;
        for (const auto& pred : allPredictions)
        {
            predsByClass[pred.predictedClass].push_back(pred);
        }

        std::vector<Prediction> selected;

        for (auto& kvp : predsByClass)
        {
            std::vector<Prediction>& proposals = kvp.second;

            while (!proposals.empty())
            {
                // Find the proposal with the highest score
                auto max_iter = std::max_element(proposals.begin(), proposals.end(),
                    [](const Prediction& lhs, const Prediction& rhs) {
                        return lhs.score < rhs.score;
                    });

                // Move it into the "selected" array
                selected.push_back(*max_iter);
                proposals.erase(max_iter);

                // Compare this selected prediction with all the remaining propsals. Compute their IOU and remove any
                // that are greater than the threshold.
                for (auto it = proposals.begin(); it != proposals.end(); it)
                {
                    float iou = ComputeIntersectionOverUnion(selected.back(), *it);

                    if (iou > threshold)
                    {
                        // Remove this element
                        it = proposals.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }
        }

        return selected;
    }

    // Helper function for fomatting strings. Format(os, a, b, c) is equivalent to os << a << b << c.
    template <typename T>
    std::ostream& Format(std::ostream& os, T&& arg)
    {
        return (os << std::forward<T>(arg));
    }

    template <typename T, typename... Ts>
    std::ostream& Format(std::ostream& os, T&& arg, Ts&&... args)
    {
        os << std::forward<T>(arg);
        return Format(os, std::forward<Ts>(args)...);
    }
}

Sample::Sample()
    : m_ctrlConnected(false)
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

    PIXEndEvent();
}
#pragma endregion

void Sample::GetModelPredictions(
    const ModelOutput& modelOutput,
    const YoloV4Constants::BBoxData& constants,
    std::vector<Prediction>* out)
{
    // Convenience
    float xyScale = constants.xyScale;
    float stride = constants.stride;
    const auto& anchors = constants.anchors;

    // There are 3 anchors per scale, and each anchor is an (x,y) coordinate, so the anchors array should have 6
    // values total.
    assert(anchors.size() == 6);

    // DirectML writes the final output data in NHWC, where the C channel contains the bounding box & probabilities 
    // for each prediction.
    const uint32_t predTensorN = modelOutput.desc.sizes[0];
    const uint32_t predTensorH = modelOutput.desc.sizes[1];
    const uint32_t predTensorW = modelOutput.desc.sizes[2];
    const uint32_t predTensorC = modelOutput.desc.sizes[3];

    // YoloV4 predicts 3 boxes per scale, so we expect 3 separate predictions here
    assert(predTensorN == 3);

    // Width should contain the bounding box x/y/w/h, a confidence score, the probability for max class, and the class index
    assert(predTensorC == 7);

    struct PotentialPrediction
    {
        float bx;
        float by;
        float bw;
        float bh;
        float confidence;
        float classMaxProbability;
        float classIndexAsFloat;
    };

    // The output tensor should be large enough to hold the expected number of predictions.
    assert(predTensorN * predTensorH * predTensorW * sizeof(PotentialPrediction) <= modelOutput.desc.totalTensorSizeInBytes);
    std::vector<PotentialPrediction> tensorData = CopyReadbackHeap<PotentialPrediction>(modelOutput.readback.Get());

    // Scale the boxes to be relative to the original image size
    auto viewport = m_deviceResources->GetScreenViewport();
    float xScale = (float)viewport.Width / YoloV4Constants::c_inputWidth;
    float yScale = (float)viewport.Height / YoloV4Constants::c_inputHeight;

    uint32_t currentPredIndex = 0;
    for (uint32_t n = 0; n < predTensorN; ++n)
    {
        for (uint32_t h = 0; h < predTensorH; ++h)
        {
            for (uint32_t w = 0; w < predTensorW; ++w)
            {
                const PotentialPrediction& currentPred = tensorData[currentPredIndex++];

                // Discard boxes with low scores
                float score = currentPred.confidence * currentPred.classMaxProbability;
                if (score < YoloV4Constants::c_scoreThreshold)
                {
                    continue;
                }

                // We need to do some postprocessing on the raw values before we return them

                // Apply xyScale. Need to apply offsets of half a grid cell here, to ensure the scaling is
                // centered around zero.
                float bx = xyScale * (currentPred.bx - 0.5f) + 0.5f;
                float by = xyScale * (currentPred.by - 0.5f) + 0.5f;

                // Transform the x/y from being relative to the grid cell, to being relative to the whole image
                bx = (bx + (float)w) * stride;
                by = (by + (float)h) * stride;

                // Scale the w/h by the supplied anchors
                float bw = currentPred.bw * anchors[n * 2];
                float bh = currentPred.bh * anchors[n * 2 + 1];

                // Convert x,y,w,h to xmin,ymin,xmax,ymax
                float xmin = bx - bw / 2;
                float ymin = by - bh / 2;
                float xmax = bx + bw / 2;
                float ymax = by + bh / 2;

                xmin *= xScale;
                ymin *= yScale;
                xmax *= xScale;
                ymax *= yScale;

                // Clip values out of range
                xmin = std::clamp(xmin, 0.0f, (float)viewport.Width);
                ymin = std::clamp(ymin, 0.0f, (float)viewport.Height);
                xmax = std::clamp(xmax, 0.0f, (float)viewport.Width);
                ymax = std::clamp(ymax, 0.0f, (float)viewport.Height);

                // Discard invalid boxes
                if (xmax <= xmin || ymax <= ymin || IsInfOrNan({ xmin, ymin, xmax, ymax }))
                {
                    continue;
                }

                Prediction pred = {};
                pred.xmin = xmin;
                pred.ymin = ymin;
                pred.xmax = xmax;
                pred.ymax = ymax;
                pred.score = score;
                pred.predictedClass = static_cast<uint32_t>(currentPred.classIndexAsFloat);
                out->push_back(pred);
            }
        }
    }
}

#pragma region Frame Render
// Draws the scene.
void Sample::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

    // Prepare the command list to render a new frame.
    m_deviceResources->Prepare();
    Clear();
    
    auto commandList = m_deviceResources->GetCommandList();
        
    // Render the result to the screen

    auto viewport = m_deviceResources->GetScreenViewport();
    auto scissorRect = m_deviceResources->GetScissorRect();

    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render to screen");

        commandList->OMSetRenderTargets(1, &m_deviceResources->GetRenderTargetView(), FALSE, nullptr);

        commandList->SetGraphicsRootSignature(m_texRootSignatureLinear.Get());
        commandList->SetPipelineState(m_texPipelineStateLinear.Get());

        auto heap = m_SRVDescriptorHeap->Heap();
        commandList->SetDescriptorHeaps(1, &heap);

        commandList->SetGraphicsRootDescriptorTable(0,
            m_SRVDescriptorHeap->GetGpuHandle(e_descTexture));

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
    
    // Render the UI
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render UI");

        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);

        auto size = m_deviceResources->GetOutputSize();
        auto safe = SimpleMath::Viewport::ComputeTitleSafeArea(size.right, size.bottom);

        // Draw the text HUD.
        ID3D12DescriptorHeap* fontHeaps[] = { m_fontDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(fontHeaps), fontHeaps);
                
        m_spriteBatch->Begin(commandList);

        float xCenter = static_cast<float>(safe.left + (safe.right - safe.left) / 2);

        const wchar_t* mainLegend = m_ctrlConnected ?
            L"[View] Exit   [X] Play/Pause"
            : L"ESC - Exit     ENTER - Play/Pause";
        SimpleMath::Vector2 mainLegendSize = m_legendFont->MeasureString(mainLegend);
        auto mainLegendPos = SimpleMath::Vector2(xCenter - mainLegendSize.x / 2, static_cast<float>(safe.bottom) - m_legendFont->GetLineSpacing());

        // Render a drop shadow by drawing the text twice with a slight offset.
        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
            mainLegend, mainLegendPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.0f, 0.0f, 0.0f, 0.25f));
        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
            mainLegend, mainLegendPos, ATG::Colors::White);

        const wchar_t* modeLabel = L"Object detection model:";
        SimpleMath::Vector2 modeLabelSize = m_labelFontBold->MeasureString(modeLabel);
        auto modeLabelPos = SimpleMath::Vector2(safe.right - modeLabelSize.x, static_cast<float>(safe.top));

        m_labelFontBold->DrawString(m_spriteBatch.get(), modeLabel, modeLabelPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFontBold->DrawString(m_spriteBatch.get(), modeLabel, modeLabelPos, ATG::Colors::White);

        const wchar_t* modeType = L"YOLO V4";
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

    // Readback the raw data from the model, compute the model's predictions, and render the bounding boxes
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render predictions");

        // Retrieve the predictions from the raw model outputs
        std::vector<Prediction> preds;
        GetModelPredictions(m_modelSOutput, YoloV4Constants::BBoxData::Small(), &preds);
        GetModelPredictions(m_modelMOutput, YoloV4Constants::BBoxData::Medium(), &preds);
        GetModelPredictions(m_modelLOutput, YoloV4Constants::BBoxData::Large(), &preds);

        // Apply NMS to select the best boxes
        preds = ApplyNonMaximalSuppression(preds, YoloV4Constants::c_nmsThreshold);

        // Print some debug information about the predictions
        if (preds.size() != 0)
        {
            std::stringstream ss;
            Format(ss, "# of predictions: ", preds.size(), "\n");
            
            for (const auto& pred : preds)
            {
                const char* className = YoloV4Constants::c_classes[pred.predictedClass];
                int xmin = static_cast<int>(std::round(pred.xmin));
                int ymin = static_cast<int>(std::round(pred.ymin));
                int xmax = static_cast<int>(std::round(pred.xmax));
                int ymax = static_cast<int>(std::round(pred.ymax));

                Format(ss, "  ", className, ": score ", pred.score, ", box (", xmin, ",", ymin, "),(", xmax, ",", ymax, ")\n");
            }

            OutputDebugStringA(ss.str().c_str());

            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissorRect);

            // Draw bounding box outlines
            m_lineEffect->Apply(commandList);
            m_lineBatch->Begin(commandList);
            for (const auto& pred : preds)
            {
                VertexPositionColor upperLeft(SimpleMath::Vector3(pred.xmin, pred.ymin, 0.f), ATG::Colors::White);
                VertexPositionColor lowerLeft(SimpleMath::Vector3(pred.xmin, pred.ymax, 0.f), ATG::Colors::White);
                VertexPositionColor upperRight(SimpleMath::Vector3(pred.xmax, pred.ymin, 0.f), ATG::Colors::White);
                VertexPositionColor lowerRight(SimpleMath::Vector3(pred.xmax, pred.ymax, 0.f), ATG::Colors::White);

                m_lineBatch->DrawLine(upperLeft, upperRight);
                m_lineBatch->DrawLine(upperRight, lowerRight);
                m_lineBatch->DrawLine(lowerRight, lowerLeft);
                m_lineBatch->DrawLine(lowerLeft, upperLeft);
            }
            m_lineBatch->End();

            // Draw predicted class labels
            m_spriteBatch->Begin(commandList);
            for (const auto& pred : preds)
            {
                const char* classText = YoloV4Constants::c_classes[pred.predictedClass];
                std::wstring classTextW(classText, classText + strlen(classText));

                // Render a drop shadow by drawing the text twice with a slight offset.
                DX::DrawControllerString(m_spriteBatch.get(), m_labelFont.get(), m_ctrlFont.get(),
                    classTextW.c_str(), SimpleMath::Vector2(pred.xmin, pred.ymin) + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.0f, 0.0f, 0.0f, 0.25f));
                DX::DrawControllerString(m_spriteBatch.get(), m_labelFont.get(), m_ctrlFont.get(),
                    classTextW.c_str(), SimpleMath::Vector2(pred.xmin, pred.ymin), ATG::Colors::White);
            }
            m_spriteBatch->End();
        }

        PIXEndEvent(commandList);
    }

    // 
    // Kick off the compute work that will be used to render the next frame. We do this now so that the data will be
    // ready by the time the next frame comes around.
    // 

#if USE_VIDEO
    // Get the latest video frame
    RECT r = { 0, 0, static_cast<LONG>(m_origTextureWidth), static_cast<LONG>(m_origTextureHeight) };
    MFVideoNormalizedRect rect = { 0.0f, 0.0f, 1.0f, 1.0f };
    m_player->TransferFrame(m_sharedVideoTexture, rect, r);
#endif

    // Convert image to tensor format (original texture -> model input)
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Convert input image");

        ID3D12DescriptorHeap* pHeaps[] = { m_SRVDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

        commandList->SetComputeRootSignature(m_computeRootSignature.Get());

        ImageLayoutCB imageLayoutCB = {};
        imageLayoutCB.Height = m_origTextureHeight;
        imageLayoutCB.Width = m_origTextureWidth;
        imageLayoutCB.UseNhwc = false;

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

        m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlGraph.Get(), m_dmlBindingTable.Get());

        // Note that we don't need to barrier these back to UNORDERED_ACCESS once we're done, because they'll
        // automatically be demoted to COMMON once the commandlist is executed
        D3D12_RESOURCE_BARRIER barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
            CD3DX12_RESOURCE_BARRIER::Transition(m_modelSOutput.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_modelMOutput.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_modelLOutput.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);

        // Copy result into readback heaps
        commandList->CopyResource(m_modelSOutput.readback.Get(), m_modelSOutput.output.Get());
        commandList->CopyResource(m_modelMOutput.readback.Get(), m_modelMOutput.output.Get());
        commandList->CopyResource(m_modelLOutput.readback.Get(), m_modelLOutput.output.Get());

        PIXEndEvent(commandList);
    }

    // Show the new frame.
    PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");

    m_deviceResources->Present();

    PIXEndEvent(m_deviceResources->GetCommandQueue());

    m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());
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

        DX::ThrowIfFailed(
            device->CreateSharedHandle(
                m_videoTexture.Get(),
                nullptr,
                GENERIC_ALL,
                nullptr,
                &m_sharedVideoTexture));

        CreateShaderResourceView(device, m_videoTexture.Get(), m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));
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
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();

    m_SRVDescriptorHeap.reset();

    m_computePSO.Reset();
    m_computeRootSignature.Reset();

    m_dmlDevice.Reset();
    m_dmlCommandRecorder.Reset();

    m_modelInput.Reset();
    m_modelSOutput = {};
    m_modelMOutput = {};
    m_modelLOutput = {};
    m_dmlOpInitializer.Reset();
    m_dmlGraph.Reset();
    m_modelTemporaryResource.Reset();
    m_modelPersistentResource.Reset();

    m_dmlDescriptorHeap.reset();

    m_graphicsMemory.reset();
}

void Sample::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion
