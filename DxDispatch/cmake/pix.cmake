FetchContent_Declare(
    pix
    URL https://www.nuget.org/api/v2/package/WinPixEventRuntime/1.0.210818001
    URL_HASH SHA256=70910c2d58b555693ba340b2f5e92dcb2f3d54690ee98130d2ebf7aef5730307
)

FetchContent_MakeAvailable(pix)

add_library(pix SHARED IMPORTED)
add_library(Microsoft::PIX ALIAS pix)

target_include_directories(pix INTERFACE ${pix_SOURCE_DIR}/include)

if(WIN32)
    set(pix_PLATFORM ${CMAKE_GENERATOR_PLATFORM})
    if(NOT pix_PLATFORM)
        set(pix_PLATFORM ${CMAKE_VS_PLATFORM_NAME})
    endif()
    if(NOT pix_PLATFORM)
        set(pix_PLATFORM x64)
    endif()

    if(${pix_PLATFORM} MATCHES x64)
        set(pix_BIN_PATH ${pix_SOURCE_DIR}/bin/x64)
    elseif (${pix_PLATFORM} MATCHES ARM64)
        set(pix_BIN_PATH ${pix_SOURCE_DIR}/bin/ARM64)
    endif()

    set_property(TARGET pix PROPERTY IMPORTED_LOCATION ${pix_BIN_PATH}/WinPixEventRuntime.dll)
    set_property(TARGET pix PROPERTY IMPORTED_IMPLIB ${pix_BIN_PATH}/WinPixEventRuntime.lib)

    install(FILES ${pix_BIN_PATH}/WinPixEventRuntime.dll DESTINATION bin)
endif()