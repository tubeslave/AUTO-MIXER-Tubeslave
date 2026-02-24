cmake_minimum_required(VERSION 3.15)
project(AutoFaderCore VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(DANTE_VERIFY_ONLY "Build only Dante verification tool" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(USE_DANTE "Enable Dante Application Library integration" ON)
option(USE_SIMD "Enable SIMD optimizations (SSE/AVX)" ON)

# Platform-specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
elseif(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7+
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -O3  # Maximum optimization
        -ffast-math  # Fast math for DSP
    )
    
    # SIMD optimizations (architecture-specific)
    if(USE_SIMD)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
            # ARM NEON for Apple Silicon / ARM
            message(STATUS "Using ARM NEON SIMD optimizations")
            # ARM already has NEON enabled by default in -O3
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            # x86_64 SSE/AVX
            message(STATUS "Using x86_64 SSE4.2/AVX SIMD optimizations")
            add_compile_options(-msse4.2 -mavx)
        endif()
    endif()
elseif(MSVC)
    add_compile_options(
        /W4
        /O2  # Maximum optimization
        $<$<BOOL:${USE_SIMD}>:/arch:AVX2>
    )
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib/dante/include
)

# Find required libraries
find_package(Threads REQUIRED)

# FFT Library (try FFTW3, fallback to KissFFT)
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 fftw3f)
endif()

if(FFTW3_FOUND)
    message(STATUS "Using FFTW3 for FFT")
    add_definitions(-DUSE_FFTW3)
    include_directories(${FFTW3_INCLUDE_DIRS})
    link_directories(${FFTW3_LIBRARY_DIRS})
else()
    message(STATUS "FFTW3 not found, using built-in FFT implementation")
    add_definitions(-DUSE_BUILTIN_FFT)
endif()

# Dante Application Library (SDK v1.3.0.2 uses libdal.dylib)
set(DANTE_LIBS "")  # Initialize empty list
if(USE_DANTE)
    if(APPLE)
        # Try SDK v1.3.0.2 naming first (libdal.dylib), fallback to old naming
        if(EXISTS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libdal.dylib")
            set(DANTE_LIB_NAME "libdal.dylib")
        else()
            set(DANTE_LIB_NAME "libdante.dylib")
        endif()
    elseif(WIN32)
        set(DANTE_LIB_NAME "dante.lib")
    else()
        set(DANTE_LIB_NAME "libdante.so")
    endif()
    
    set(DANTE_LIB_PATH "${CMAKE_SOURCE_DIR}/lib/dante/lib/${DANTE_LIB_NAME}")
    
    if(EXISTS ${DANTE_LIB_PATH})
        message(STATUS "Dante library found: ${DANTE_LIB_PATH}")
        add_definitions(-DUSE_DANTE)
        # Add include path for SDK v1.3.0.2 structure
        include_directories(${CMAKE_SOURCE_DIR}/lib/dante/include)
        
        # SDK v1.3.0.2 requires additional libraries
        list(APPEND DANTE_LIBS ${DANTE_LIB_PATH})
        
        # Add libdal_api.a (static library)
        if(EXISTS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libdal_api.a")
            list(APPEND DANTE_LIBS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libdal_api.a")
        endif()
        
        # Add libaud-protobuf-lite.a (protobuf support)
        if(EXISTS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libaud-protobuf-lite.a")
            list(APPEND DANTE_LIBS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libaud-protobuf-lite.a")
        endif()
        
        # Add libsodium.a (crypto support)
        if(EXISTS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libsodium.a")
            list(APPEND DANTE_LIBS "${CMAKE_SOURCE_DIR}/lib/dante/lib/libsodium.a")
        endif()
        
        message(STATUS "Dante libraries: ${DANTE_LIBS}")
    else()
        message(WARNING "Dante library not found at ${DANTE_LIB_PATH}")
        message(WARNING "Dante support will be disabled. See DANTE_SETUP.md for installation instructions.")
        set(USE_DANTE OFF)
    endif()
endif()

# Source files
set(CORE_SOURCES
    src/dsp/lufs_meter.cpp
    src/dsp/true_peak_meter.cpp
    src/dsp/spectral_analyzer.cpp
    src/metrics/audio_metrics.cpp
    src/metrics/metrics_exporter.cpp
    src/bridge/shared_memory.cpp
)

if(USE_DANTE)
    list(APPEND CORE_SOURCES src/dante/dante_receiver.cpp)
endif()

# Main executable
if(NOT DANTE_VERIFY_ONLY)
    add_executable(auto_fader_core
        src/main.cpp
        ${CORE_SOURCES}
    )
    
    target_link_libraries(auto_fader_core
        Threads::Threads
        ${CMAKE_DL_LIBS}
        $<$<BOOL:${FFTW3_FOUND}>:${FFTW3_LIBRARIES}>
        $<$<BOOL:${USE_DANTE}>:${DANTE_LIBS}>
    )
    
    # Set RPATH for macOS/Linux
    if(APPLE)
        # Add RPATH to find Dante library
        if(USE_DANTE)
            set_target_properties(auto_fader_core PROPERTIES
                INSTALL_RPATH "@executable_path/../lib/dante/lib;@executable_path/../lib"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
            # Also set build RPATH for testing
            set_target_properties(auto_fader_core PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib/dante/lib"
            )
        else()
            set_target_properties(auto_fader_core PROPERTIES
                INSTALL_RPATH "@executable_path/../lib"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    elseif(UNIX)
        set_target_properties(auto_fader_core PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

# Dante verification tool
if(USE_DANTE AND DANTE_VERIFY_ONLY)
    add_executable(verify_dante
        src/dante/verify_dante.cpp
    )
    
    target_link_libraries(verify_dante
        ${DANTE_LIB_PATH}
        Threads::Threads
    )
endif()

# Tests
if(BUILD_TESTS AND NOT DANTE_VERIFY_ONLY)
    enable_testing()
    
    add_executable(test_lufs_meter
        test/test_lufs_meter.cpp
        src/dsp/lufs_meter.cpp
    )
    target_link_libraries(test_lufs_meter Threads::Threads)
    add_test(NAME LUFS_Meter COMMAND test_lufs_meter)
    
    add_executable(test_spectral_analyzer
        test/test_spectral_analyzer.cpp
        src/dsp/spectral_analyzer.cpp
    )
    target_link_libraries(test_spectral_analyzer
        Threads::Threads
        $<$<BOOL:${FFTW3_FOUND}>:${FFTW3_LIBRARIES}>
    )
    add_test(NAME Spectral_Analyzer COMMAND test_spectral_analyzer)
    
    add_executable(test_shared_memory
        test/test_shared_memory.cpp
        src/bridge/shared_memory.cpp
    )
    target_link_libraries(test_shared_memory Threads::Threads)
    add_test(NAME Shared_Memory COMMAND test_shared_memory)
endif()

# Installation
install(TARGETS auto_fader_core
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "====== Auto Fader Core Configuration ======")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Dante support: ${USE_DANTE}")
message(STATUS "SIMD optimizations: ${USE_SIMD}")
message(STATUS "FFT library: ${FFTW3_FOUND}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "==========================================")
message(STATUS "")
