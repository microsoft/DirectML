FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        10.2.1
)

set(FMT_DOC OFF CACHE INTERNAL "Generate the doc target.")
set(FMT_INSTALL OFF CACHE INTERNAL "Generate the install target.")
set(FMT_TEST OFF CACHE INTERNAL "Generate the test target.")
set(FMT_FUZZ OFF CACHE INTERNAL "Generate the fuzz target.")
set(FMT_CUDA_TEST OFF CACHE INTERNAL "Generate the cuda-test target.")
FetchContent_MakeAvailable(fmt)