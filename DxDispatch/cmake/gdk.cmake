# =============================================================================
# Helper function to introduce a target representing a GDK.
#
# The main function is `add_gdk_target`, which has the following parameters:
#
# - CACHE_PREFIX : string used to prefix cache variables associated with the function.
# 
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - GDK_TYPE
#       system : Use system installation of the GDK.
#       none   : No dependency on the GDK (adds a stub target).
#
# - GDK_SYSTEM_EDITION : Edition of the GDK (TYPE == system).
# =============================================================================

include_guard()
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_gdk_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_GDK_TYPE
    if(TARGET_XBOX)
        set(default_type system)
    else()
        set(default_type none)
    endif()
    set(${prefix}_GDK_TYPE 
        "${default_type}"
        CACHE STRING "GDK dependency type"
    )
    set_property(CACHE ${prefix}_GDK_TYPE PROPERTY STRINGS system none)

    # <PREFIX>_GDK_SYSTEM_EDITION
    string(REGEX REPLACE "(\\\\|/)$" "" default_system_edition "$ENV{GameDKLatest}")
    cmake_path(GET default_system_edition FILENAME default_system_edition)
    set(${prefix}_GDK_SYSTEM_EDITION
        "${default_system_edition}"
        CACHE STRING "Edition of the GDK (TYPE == system)"
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a system installation of the GDK.
# -----------------------------------------------------------------------------
function(init_gdk_target_system target_name edition)
    if(NOT edition)
        message(FATAL_ERROR "GDK edition is not specified")
    endif()

    if(NOT TARGET_XBOX)
        message(FATAL_ERROR "GDK as a system dependency requires an Xbox target")
    endif()

    set(dxcompiler_path "$ENV{GameDK}${edition}/GXDK/bin/${TARGET_XBOX_TYPE}/dxcompiler_${TARGET_XBOX_FILE_SUFFIX}.dll")

    set_property(TARGET ${target_name} PROPERTY VS_GLOBAL_XdkEditionTarget "${edition}")
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "System (${edition})")
    set_property(TARGET ${target_name} PROPERTY DX_COMPILER_PATH "${dxcompiler_path}")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_gdk_target_stub target_name)
    target_compile_definitions(${target_name} INTERFACE GDK_NONE)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_gdk_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_gdk_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_GDK_TYPE}")
    set(system_edition "${${ARG_CACHE_PREFIX}_GDK_SYSTEM_EDITION}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER "${type}" type)
    if(type STREQUAL system)
        init_gdk_target_system(${target_name} "${system_edition}")
    elseif(type STREQUAL none)
        init_gdk_target_stub(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()