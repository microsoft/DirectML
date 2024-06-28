FetchContent_Declare(
    flatbuffer
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG        v23.5.26
    GIT_SHALLOW    ON
)

FetchContent_GetProperties(flatbuffer)

if(NOT flatbuffer_POPULATED)
    FetchContent_Populate(flatbuffer)
endif()

add_library(flatbuffer INTERFACE)
target_include_directories(flatbuffer INTERFACE "${flatbuffer_SOURCE_DIR}/include")