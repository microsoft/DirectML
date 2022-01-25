FetchContent_Declare(
    wil
    GIT_REPOSITORY https://github.com/Microsoft/wil
    GIT_TAG        2e225973d6c2ecf17fb4d376ddbeedb6db7dd82f
)

FetchContent_GetProperties(wil)
if(NOT wil_POPULATED)
    set(WIL_BUILD_PACKAGING OFF CACHE INTERNAL "Sets option to build the packaging, default on")
    set(WIL_BUILD_TESTS OFF CACHE INTERNAL "Sets option to build the unit tests, default on")
    FetchContent_Populate(wil)
    add_subdirectory(${wil_SOURCE_DIR} ${wil_BINARY_DIR} EXCLUDE_FROM_ALL)
    add_library(Microsoft::WIL ALIAS WIL)
endif()