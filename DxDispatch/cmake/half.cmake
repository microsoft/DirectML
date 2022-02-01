FetchContent_Declare(
    half
    PREFIX half
    URL "https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip/download"
)
FetchContent_MakeAvailable(half)

add_library(half INTERFACE)
add_library(Half::Half ALIAS half)

target_include_directories(half INTERFACE "${half_SOURCE_DIR}/include")