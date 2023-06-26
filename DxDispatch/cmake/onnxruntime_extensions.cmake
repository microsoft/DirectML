# =============================================================================
# Helper function to introduce a target representing an ONNX Runtime extensions.
#
# The main function is `add_onnxruntime_extensions_target`, which has the following parameters:
#
# - CACHE_PREFIX : string used to prefix cache variables associated with the function.
#
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - ONNXRUNTIME_EXTENSIONS_TYPE
#       nuget  : Use NuGet distribution of ONNX Runtime.
#       none   : No dependency on ONNX Runtime (stub target).
#
# - ONNXRUNTIME_EXTENSIONS_NUGET_ID : ID of the ONNX Runtime NuGet package (TYPE == nuget).
# - ONNXRUNTIME_EXTENSIONS_NUGET_VERSION : Version of the ONNX Runtime NuGet package (TYPE == nuget).
# - ONNXRUNTIME_EXTENSIONS_NUGET_HASH : SHA256 hash of the ONNX Runtime NuGet package (TYPE == nuget).
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_onnxruntime_extensions_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_ONNXRUNTIME_EXTENSIONS_TYPE
    if(TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X86|X64|ARM|ARM64$")
        set(default_type nuget)
    else()
        set(default_type none)
    endif()
    set(${prefix}_ONNXRUNTIME_EXTENSIONS_TYPE 
        "${default_type}" 
        CACHE STRING "ONNXRUNTIME extensions dependency type"
    )
    set_property(CACHE ${prefix}_ONNXRUNTIME_EXTENSIONS_TYPE PROPERTY STRINGS nuget local none)

    # <PREFIX>_ONNXRUNTIME_EXTENSIONS_NUGET_ID
    set(${prefix}_ONNXRUNTIME_EXTENSIONS_NUGET_ID
        Microsoft.ML.OnnxRuntime.Extensions
        CACHE STRING "ID of the ONNX Runtime NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_ONNXRUNTIME_EXTENSIONS_NUGET_VERSION
    set(${prefix}_ONNXRUNTIME_EXTENSIONS_NUGET_VERSION
        0.8.0
        CACHE STRING "Version of the ONNX Runtime NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_ONNXRUNTIME_EXTENSIONS_NUGET_HASH
    set(${prefix}_ONNXRUNTIME_EXTENSIONS_NUGET_HASH 
        543276F2DBCE9C7F22641FFFE3CAD3E7B9223650B3B66D822F5F5503F53EB301
        CACHE STRING "SHA256 hash of the ONNX Runtime NuGet package (TYPE == nuget)."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a NuGet distribution.
# -----------------------------------------------------------------------------
function(init_onnxruntime_extensions_target_nuget target_name pkg_id pkg_version pkg_hash)
    if(NOT (TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|X86|ARM|ARM64$"))
        message(FATAL_ERROR "ONNX Runtime extensions NuGet isn't available on this platform")
    endif()

    set(content ${target_name}_content)
    FetchContent_Declare(
        ${content}
        URL "https://www.nuget.org/api/v2/package/${pkg_id}/${pkg_version}"
        URL_HASH SHA256=${pkg_hash}
    )
    FetchContent_MakeAvailable(${content})

    set(bin_path "${${content}_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}")
    target_append_redist_file(${target_name} "${bin_path}/native/ortextensions.dll")

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "NuGet (${pkg_id}.${pkg_version})")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_onnxruntime_extensions_target_stub target_name)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
    target_compile_definitions(${target_name} INTERFACE ONNXRUNTIME_EXTENSIONS_NONE)
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_onnxruntime_extensions_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_onnxruntime_extensions_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_EXTENSIONS_TYPE}")
    set(nuget_id "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_EXTENSIONS_NUGET_ID}")
    set(nuget_version "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_EXTENSIONS_NUGET_VERSION}")
    set(nuget_hash "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_EXTENSIONS_NUGET_HASH}")
    set(local_path "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_EXTENSIONS_LOCAL_PATH}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER "${type}" type)
    if(type STREQUAL nuget)
        init_onnxruntime_extensions_target_nuget(${target_name} ${nuget_id} ${nuget_version} ${nuget_hash})
    elseif(type STREQUAL none)
        init_onnxruntime_extensions_target_stub(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()