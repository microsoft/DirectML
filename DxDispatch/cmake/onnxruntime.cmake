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
    if(TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|ARM64$")
        set(default_type nuget)
    elseif(TARGET_XBOX)
        set(default_type local)
    else()
        set(default_type none)
    endif()
    set(${prefix}_ONNXRUNTIME_TYPE 
        "${default_type}" 
        CACHE STRING "ONNXRUNTIME dependency type"
    )
    set_property(CACHE ${prefix}_ONNXRUNTIME_TYPE PROPERTY STRINGS nuget local none)

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

    # <PREFIX>_ONNXRUNTIME_LOCAL_PATH
    set(${prefix}_ONNXRUNTIME_LOCAL_PATH
        ""
        CACHE STRING "Path to a local build of ONNX Runtime (TYPE == local)."
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
# Init using a local build.
# -----------------------------------------------------------------------------
function(init_onnxruntime_target_local target_name local_path)
    if(NOT IS_DIRECTORY "${local_path}")
        message(FATAL_ERROR "'${local_path}' is not a directory. You must set DXD_ONNXRUNTIME_LOCAL_PATH to a directory containing pre-built DirectML.")
    endif()

    # onnxruntime.dll is required.
    set(onnxruntime_dll_path "${local_path}/bin/onnxruntime.dll")
    if(NOT EXISTS ${onnxruntime_dll_path})
        message(FATAL_ERROR "Could not find '${onnxruntime_dll_path}'")
    endif()
    target_append_redist_file(${target_name} ${onnxruntime_dll_path})

    # onnxruntime_providers_shared.dll is required.
    set(onnxruntime_providers_dll_path "${local_path}/bin/onnxruntime_providers_shared.dll")
    if(NOT EXISTS ${onnxruntime_providers_dll_path})
        message(FATAL_ERROR "Could not find '${onnxruntime_providers_dll_path}'")
    endif()
    target_append_redist_file(${target_name} ${onnxruntime_providers_dll_path})

    # onnxruntime.lib is required.
    set(onnxruntime_lib_path "${local_path}/lib/onnxruntime.lib")
    if(NOT EXISTS ${onnxruntime_lib_path})
        message(FATAL_ERROR "Could not find '${onnxruntime_lib_path}'")
    endif()
    target_link_libraries(${target_name} INTERFACE ${onnxruntime_lib_path})

    # Include dir must exist with dml_provider_factory.h in it.
    set(dml_provider_h "${local_path}/include/onnxruntime/core/providers/dml/dml_provider_factory.h")
    if(NOT EXISTS ${dml_provider_h})
        message(FATAL_ERROR "Could not find '${dml_provider_h}'")
    endif()
    
    target_include_directories(${target_name} INTERFACE
        "${local_path}/include/onnxruntime/core/session"
        "${local_path}/include/onnxruntime/core/providers/cpu"
        "${local_path}/include/onnxruntime/core/providers/dml"
    )

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Local")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_onnxruntime_target_stub target_name)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
    target_compile_definitions(${target_name} INTERFACE ONNXRUNTIME_NONE)
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
    set(local_path "${${ARG_CACHE_PREFIX}_ONNXRUNTIME_LOCAL_PATH}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER "${type}" type)
    if(type STREQUAL nuget)
        init_onnxruntime_target_nuget(${target_name} ${nuget_id} ${nuget_version} ${nuget_hash})
    elseif(type STREQUAL local)
        init_onnxruntime_target_local(${target_name} "${local_path}")
    elseif(type STREQUAL none)
        init_onnxruntime_target_stub(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()