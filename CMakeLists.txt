cmake_minimum_required(VERSION 3.22)
set(SC_MODULE doc-classifier)
project(simply-cpp-${SC_MODULE} HOMEPAGE_URL "https://github.com/roelofrossouw/simply-cpp-doc-classifier" LANGUAGES CXX)
include(cmake/sc_bootstrap.cmake)

option(SC_DOC_CLASSIFIER_BUILD_TESTS "Build simply-cpp-doc-classifier tests" ${PROJECT_IS_TOP_LEVEL})

get_sc_version()
message(STATUS "SimplyCPP ${SC_MODULE} Version: ${SC_VERSION}")

find_or_install_package(sc-core simply-cpp-dev simply-cpp 3.0.0)
find_or_install_package(sc-image simply-cpp-image-dev simply-cpp-image 6.0.0)
# OCR reads the cards' printed words.
find_or_install_package(sc-ai simply-cpp-ai-dev simply-cpp-ai 3.0.0) # its sc::ocr without the CPU memory arena and with onnx::gpu_memory_limit() (built in)
# Outlines, lines and perspective, on the pixels themselves (sc-image keeps OpenCV out of its header).
# OpenCV 5 (Homebrew) moved contours, line fitting and transforms into geometry; 4 (Ubuntu) has them
# in imgproc, and the RANSAC fits in calib3d.
find_package(OpenCV QUIET)
if (OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
    set(SC_DOC_CLASSIFIER_OPENCV_COMPONENTS core imgproc imgcodecs geometry)
else ()
    set(SC_DOC_CLASSIFIER_OPENCV_COMPONENTS core imgproc imgcodecs calib3d)
endif ()
find_or_install_package(OpenCV libopencv-dev opencv COMPONENTS ${SC_DOC_CLASSIFIER_OPENCV_COMPONENTS})
# sc::doc::identity's results as JSON (to_json(), from_json()).
find_or_install_package(nlohmann_json nlohmann-json3-dev nlohmann-json)

set(SOURCE_OBJECTS)
set(SOURCE_LIBRARIES)
set(SOURCE_LINK_LIBRARIES)

add_sc_object(sc_doc_classifier
        LINK_LIBRARIES sc::sc-core sc::sc-image sc::sc-ai ${OpenCV_LIBS}
        PUBLIC_LINK_LIBRARIES sc::sc-core sc::sc-image sc::sc-ai ${OpenCV_LIBS})
# South African identity documents read for their fields (was sc::identity in sc-ai until 2.6.0).
add_sc_object(identity
        LINK_LIBRARIES sc::sc-core sc::sc-image sc::sc-ai nlohmann_json::nlohmann_json
        PUBLIC_LINK_LIBRARIES sc::sc-core sc::sc-image sc::sc-ai nlohmann_json::nlohmann_json)

add_sc_libraries(DESCRIPTION "simply-cpp doc-classifier: documents found in photographs and scans, identified, cut out straight, and South African identity documents read")
target_compile_features(sc-doc-classifier PUBLIC cxx_std_20)
target_compile_features(sc-doc-classifier-shared PUBLIC cxx_std_20)

install_sc_module(PATH_VARS CMAKE_INSTALL_INCLUDEDIR)
install(FILES cmake/SimplyCppFunctions.cmake cmake/sc_bootstrap.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sc-doc-classifier
        COMPONENT development)

if (SC_DOC_CLASSIFIER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests sc-doc-classifier-test)
endif ()

add_sc_example(sc-doc-classifier-contours)
add_sc_example(sc-doc-classifier-lines)
add_sc_example(sc-doc-classifier-text)
add_sc_example(sc-doc-classifier-identity)

package_sc_module(DEV_DEPENDS "simply-cpp-dev (>= 3.0.0~)" "simply-cpp-image-dev (>= 6.0.0~)" "simply-cpp-ai-dev (>= 3.0.0~)" libopencv-dev nlohmann-json3-dev)
