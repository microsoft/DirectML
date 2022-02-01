include_guard()

# Any extra argument is interpreted as the dst_path. If not given the dst_path
# is simply the filename from of the src_path.
#
# Example: target_append_redist_file(foo c:/users/foo/bar.txt) 
#          'c:/users/foo/bar.txt' -> 'bar.txt'
#
# Example: target_append_redist_file(foo c:/users/foo/bar.txt subdir/cat.txt) 
#          'c:/users/foo/bar.txt' -> 'subdir/cat.txt'
#
function(target_append_redist_file target_name src_path)
    # Any extra argument is interpreted as the dst_path. If not given the dst_path
    # is simply the filename from of the src_path.
    cmake_path(GET src_path FILENAME dst_path)
    if(ARGC GREATER 2)
        set(dst_path ${ARGV2})
    endif()
    set_property(TARGET ${target_name} APPEND PROPERTY REDIST_SRC ${src_path})
    set_property(TARGET ${target_name} APPEND PROPERTY REDIST_DST ${dst_path})
endfunction()

# Copies redist files associated with the named target's top-level dependencies.
# WARNING: this doesn't recurse into dependencies of dependencies.
function(target_copy_redist_dependencies target_name)
    get_target_property(target_dependencies ${target_name} LINK_LIBRARIES)

    foreach(target IN LISTS target_dependencies)
        if(TARGET ${target})
            get_property(tgt_redist_src TARGET ${target} PROPERTY REDIST_SRC)
            get_property(tgt_redist_dst TARGET ${target} PROPERTY REDIST_DST)
            foreach(src dst IN ZIP_LISTS tgt_redist_src tgt_redist_dst)
                add_custom_command(
                    TARGET ${target_name} 
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src}" "$<TARGET_FILE_DIR:${target_name}>/${dst}"
                )

                cmake_path(GET dst PARENT_PATH dst_dir)
                install(FILES "${src}" DESTINATION "bin/${dst_dir}")
            endforeach()
        endif()
    endforeach()
endfunction()