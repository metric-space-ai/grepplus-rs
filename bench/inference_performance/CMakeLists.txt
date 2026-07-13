cmake_minimum_required(VERSION 3.16)
project(greppy_llama_contract LANGUAGES C CXX)

set(LLAMA_CPP_SOURCE "" CACHE PATH "Path to a pinned llama.cpp source checkout")
if(NOT EXISTS "${LLAMA_CPP_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "Set LLAMA_CPP_SOURCE to a pinned llama.cpp source checkout")
endif()

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Hash one self-contained llama.cpp baseline binary" FORCE)

add_subdirectory("${LLAMA_CPP_SOURCE}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(greppy-llama-contract llama_contract.cpp)
target_compile_features(greppy-llama-contract PRIVATE cxx_std_17)
target_include_directories(
    greppy-llama-contract
    PRIVATE "${LLAMA_CPP_SOURCE}/vendor"
)
target_link_libraries(greppy-llama-contract PRIVATE llama)

if(MSVC)
    target_compile_options(greppy-llama-contract PRIVATE /W4 /permissive-)
else()
    target_compile_options(greppy-llama-contract PRIVATE -Wall -Wextra -Wpedantic)
endif()
