#pragma once

#include <cstdint>
#include "imgui.h"

class UserInterface
{
public:
    UserInterface();
    void RenderFrame(uint32_t windowWidth, uint32_t windowHeight);

private:
};