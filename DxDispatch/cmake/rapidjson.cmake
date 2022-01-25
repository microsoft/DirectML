FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson
    GIT_TAG        v1.1.0
    GIT_SHALLOW    ON
)

FetchContent_GetProperties(rapidjson)

if(NOT rapidjson_POPULATED)
    FetchContent_Populate(rapidjson)
endif()

add_library(rapidjson INTERFACE IMPORTED GLOBAL)
add_library(rapidjson::rapidjson ALIAS rapidjson)
target_include_directories(rapidjson INTERFACE ${rapidjson_SOURCE_DIR}/include)