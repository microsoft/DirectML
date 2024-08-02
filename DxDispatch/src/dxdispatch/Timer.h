#pragma once

struct Timer
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    inline Timer() { start = std::chrono::steady_clock::now(); }
    inline Timer& Start() { start = std::chrono::steady_clock::now(); return *this; }
    inline Timer& End() { end = std::chrono::steady_clock::now(); return *this; }
    inline double DurationInMilliseconds() { return std::chrono::duration<double>(end - start).count() * 1000; }
};