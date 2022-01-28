FetchContent_Declare(
    dxheaders
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers
    GIT_TAG        v1.4.9
    GIT_SHALLOW    ON
)

FetchContent_MakeAvailable(dxheaders)