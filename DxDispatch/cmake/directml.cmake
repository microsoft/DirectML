FetchContent_Declare(
    directml
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.8.0
    URL_HASH SHA256=63cc7491e300bc3d7af7638605eac70889f5eb4617459dad5942696bb8f3c25b
)

FetchContent_MakeAvailable(directml)

add_library(directml SHARED IMPORTED)
add_library(Microsoft::DirectML ALIAS directml)

target_include_directories(directml INTERFACE ${directml_SOURCE_DIR}/include)

if(WIN32)
    set(directml_PLATFORM ${CMAKE_GENERATOR_PLATFORM})
    if(NOT directml_PLATFORM)
        set(directml_PLATFORM ${CMAKE_VS_PLATFORM_NAME})
    endif()
    if(NOT directml_PLATFORM)
        set(directml_PLATFORM Win32)
    endif()
    
    if (${directml_PLATFORM} MATCHES Win32)
        set(directml_BIN_PATH ${directml_SOURCE_DIR}/bin/x86-win)
    elseif (${directml_PLATFORM} MATCHES x64)
        set(directml_BIN_PATH ${directml_SOURCE_DIR}/bin/x64-win)
    elseif (${directml_PLATFORM} MATCHES ARM)
        set(directml_BIN_PATH ${directml_SOURCE_DIR}/bin/arm-win)
    elseif (${directml_PLATFORM} MATCHES ARM64)
        set(directml_BIN_PATH ${directml_SOURCE_DIR}/bin/arm64-win)
    endif()

    set_property(TARGET directml PROPERTY IMPORTED_LOCATION ${directml_BIN_PATH}/DirectML.dll)
    set_property(TARGET directml PROPERTY IMPORTED_IMPLIB ${directml_BIN_PATH}/DirectML.lib)
    set_property(TARGET directml PROPERTY DEBUG_DLL_PATH ${directml_BIN_PATH}/DirectML.Debug.dll)

    install(
        FILES 
            ${directml_BIN_PATH}/DirectML.dll
            ${directml_BIN_PATH}/DirectML.Debug.dll
        DESTINATION bin
    )
endif()