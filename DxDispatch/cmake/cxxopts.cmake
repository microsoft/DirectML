FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts
    GIT_TAG        v3.2.1
)

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "Set to ON to build examples")
set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "Set to ON to build tests")
set(CXXOPTS_ENABLE_INSTALL OFF CACHE INTERNAL "Generate the install target")
FetchContent_MakeAvailable(cxxopts)