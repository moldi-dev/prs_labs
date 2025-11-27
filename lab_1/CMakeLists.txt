cmake_minimum_required(VERSION 3.31)
project(PRSLab1)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PRSLab1
    main.cpp
    src/color_spaces/spaces.cpp
    src/common/misc.cpp
    src/slider/slider.cpp
    src/common/file/file_utils.cpp
    src/common/logger/logger.cpp
    )

target_link_libraries(PRSLab1 PRIVATE
    ${OpenCV_LIBS}
    fmt::fmt
)