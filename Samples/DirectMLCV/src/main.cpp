#include "pch.h"
#include "app.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    App app(1200, 800, L"DirectML CV");
    return app.Start(hInstance, nCmdShow);
}