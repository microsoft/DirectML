#include "ui.h"

#include <fstream>

#include <wincodec.h>
#include <wil/result.h>
#include <wrl/client.h>

UserInterface::UserInterface()
{
    // create image
    THROW_IF_FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    Microsoft::WRL::ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    THROW_IF_FAILED(wicFactory->CreateDecoderFromFilename(
        LR"(C:\src\ort_sr_demo\zebra.jpg)",
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &decoder
    ));

    UINT frameCount;
    THROW_IF_FAILED(decoder->GetFrameCount(&frameCount));

    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    THROW_IF_FAILED(decoder->GetFrame(0, &frame));

    UINT width, height;
    THROW_IF_FAILED(frame->GetSize(&width, &height));


    WICPixelFormatGUID pixelFormat;
    THROW_IF_FAILED(frame->GetPixelFormat(&pixelFormat));

    Microsoft::WRL::ComPtr<IWICBitmapSource> bitmapSource = frame;

    // convert to 24bppRGB (most ML models expect 3 channels, not 4)
    constexpr bool modelExpectsRGB = true;
    WICPixelFormatGUID desiredFormat = modelExpectsRGB ? GUID_WICPixelFormat24bppRGB : GUID_WICPixelFormat32bppBGR;
    if (pixelFormat != desiredFormat)
    {
        Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
        THROW_IF_FAILED(wicFactory->CreateFormatConverter(&converter));

        THROW_IF_FAILED(converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat24bppRGB,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0f,
            WICBitmapPaletteTypeCustom
        ));

        Microsoft::WRL::ComPtr<IWICBitmap> bitmap;
        THROW_IF_FAILED(wicFactory->CreateBitmapFromSource(
            converter.Get(), 
            WICBitmapCacheOnLoad, 
            &bitmap
        ));

        bitmapSource = bitmap;
    }

    // bitmapSource->CopyPixels();


    // GUID_WICPixelFormat24bppBGR
    // GUID_WICPixelFormat24bppRGB
    // GUID_WICPixelFormat32bppBGR - has alpha channel
    


}


void UserInterface::RenderFrame(uint32_t windowWidth, uint32_t windowHeight)
{
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

    ImGui::Begin("canvas", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);

    static ImVec2 scrolling(0.0f, 0.0f);
    static bool opt_enable_grid = true;

    ImGui::Checkbox("Enable grid", &opt_enable_grid);
    ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");

    // Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
    if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
    if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

    // Draw border and background color
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

    // This will catch our interactions
    ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool is_hovered = ImGui::IsItemHovered(); // Hovered
    const bool is_active = ImGui::IsItemActive();   // Held
    const ImVec2 origin(canvas_p0.x + scrolling.x, canvas_p0.y + scrolling.y); // Lock scrolled origin
    const ImVec2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

    // Pan (we use a zero mouse threshold when there's no context menu)
    // You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
    const float mouse_threshold_for_pan = 0.0f;
    if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, mouse_threshold_for_pan))
    {
        scrolling.x += io.MouseDelta.x;
        scrolling.y += io.MouseDelta.y;
    }

    // Draw grid + all lines in the canvas
    draw_list->PushClipRect(canvas_p0, canvas_p1, true);
    if (opt_enable_grid)
    {
        const float GRID_STEP = 64.0f;
        for (float x = fmodf(scrolling.x, GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
            draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
        for (float y = fmodf(scrolling.y, GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
            draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
    }

    draw_list->PopClipRect();

    ImGui::End();
    ImGui::PopStyleVar();
}