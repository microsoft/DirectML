FetchContent_Declare(
    half
    PREFIX half
    URL "https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip/download"
)

if(NOT half_POPULATED)
    FetchContent_Populate(half)
endif()

add_library(half INTERFACE IMPORTED)
add_library(Half::Half ALIAS half)

target_include_directories(half INTERFACE ${half_SOURCE_DIR}/include)