FetchContent_Declare(
    directmlx
    URL https://raw.githubusercontent.com/microsoft/DirectML/91cc5e5e823d582938c3407ec65e8e4a96e020a1/Libraries/DirectMLX.h
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

if(NOT directmlx_POPULATED)
    FetchContent_Populate(directmlx)
endif()

add_library(directmlx INTERFACE IMPORTED GLOBAL)
add_library(Microsoft::DirectMLX ALIAS directmlx)

target_include_directories(directmlx INTERFACE ${directmlx_SOURCE_DIR})