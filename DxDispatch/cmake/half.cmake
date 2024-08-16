# See ThirdPartyNotices.txt for details and attributions.
# https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip/download

add_library(half INTERFACE)
add_library(Half::Half ALIAS half)

target_include_directories(half INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ExternalDependencies/half-2.1.0)