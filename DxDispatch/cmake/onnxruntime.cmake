# =============================================================================
# Helper function to introduce a target representing an ONNX Runtime redist.
#
# The main function is `add_onnxruntime_target`, which has the following parameters:
#
# - CACHE_PREFIX : string used to prefix cache variables associated with the function.
#
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - ONNXRUNTIME_TYPE
#       nuget  : Use NuGet distribution of ONNX Runtime.
#       none   : No dependeny on ONNX Runtime (stub target).
#
# - ONNXRUNTIME_NUGET_ID : ID of the ONNX Runtime NuGet package (TYPE == nuget).
# - ONNXRUNTIME_NUGET_VERSION : Version of the ONNX Runtime NuGet package (TYPE == nuget).
# - ONNXRUNTIME_NUGET_HASH : SHA256 hash of the ONNX Runtime NuGet package (TYPE == nuget).
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_onnxruntime_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_ONNXRUNTIME_TYPE
    if(TARGET_XBOX OR (TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|ARM64$"))
        set(default_type nuget)
    else()
        set(default_type none)
    endif()
    set(${prefix}_ONNXRUNTIME_TYPE 
        nuget
        CACHE STRING "ONNXRUNTIME dependency type"
    )
    set_property(CACHE ${prefix}_ONNXRUNTIME_TYPE PROPERTY STRINGS nuget gdk none)

    # <PREFIX>_ONNXRUNTIME_NUGET_ID
    set(${prefix}_ONNXRUNTIME_NUGET_ID
        Microsoft.ML.OnnxRuntime.DirectML
        CACHE STRING "ID of the ONNX Runtime NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_ONNXRUNTIME_NUGET_VERSION
    set(${prefix}_ONNXRUNTIME_NUGET_VERSION
        1.11.0
        CACHE STRING "Version of the ONNX Runtime NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_ONNXRUNTIME_NUGET_HASH
    set(${prefix}_ONNXRUNTIME_NUGET_HASH 
        17ac3a5b3f7b4566aee9f29f39859f8ed26eb18cfbdfd2f06cb05ed230b484e2
        CACHE STRING "SHA256 hash of the ONNX Runtime NuGet package (TYPE == nuget)."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a NuGet distribution.
# -----------------------------------------------------------------------------
function(init_onnxruntime_target_nuget target_name pkg_id pkg_version pkg_hash)
    if(NOT (TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|X86|ARM64$"))
        message(FATAL_ERROR "ONNX Runtime NuGet isn't available on this platform")
    endif()

    set(content ${target_name}_content)
    FetchContent_Declare(
        ${content}
        URL "https://www.nuget.org/api/v2/package/${pkg_id}/${pkg_version}"
        URL_HASH SHA256=${pkg_hash}
    )
    FetchContent_MakeAvailable(${content})

    target_include_directories(${target_name} INTERFACE "${${content}_SOURCE_DIR}/build/native/include")

    set(bin_path "${${content}_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}")
    target_link_libraries(${target_name} INTERFACE "${bin_path}/native/onnxruntime.lib")
    target_append_redist_file(${target_name} "${bin_path}/native/onnxruntime.dll")

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "NuGet (${pkg_id}.${pkg_version})")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_onnxruntime_target_stub target_name)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_onnxruntime_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_onnxruntime_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_TYPE}")
    set(nuget_id "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_NUGET_ID}")
    set(nuget_version "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_NUGET_VERSION}")
    set(nuget_hash "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_NUGET_HASH}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER "${type}" type)
    if(type STREQUAL nuget)
        init_onnxruntime_target_nuget(${target_name} ${nuget_id} ${nuget_version} ${nuget_hash})
    elseif(type STREQUAL none)
        init_onnxruntime_target_stub(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()