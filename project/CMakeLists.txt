cmake_minimum_required(VERSION 3.31)
project(PRSProject)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)

include(FetchContent)
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui
        GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PRSProject
        main.cpp
        ${IMGUI_SOURCES}
        src/color_spaces/spaces.cpp
        src/common/misc.cpp
        src/slider/slider.cpp
        src/common/file/file_utils.cpp
        src/common/logger/logger.cpp
        src/utils/BarcodeUtils.cpp
        src/utils/BarcodeUtils.h
)

target_include_directories(PRSProject PRIVATE
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(PRSProject PRIVATE
        ${OpenCV_LIBS}
        fmt::fmt
        glfw
        OpenGL::GL
        ${X11_LIBRARIES}
)