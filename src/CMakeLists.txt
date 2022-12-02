cmake_minimum_required(VERSION 3.13)
project(rectangle)

set(CMAKE_CXX_STANDART 17)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_custom_target(shaders)
add_custom_command(TARGET shaders COMMAND glslc ${CMAKE_SOURCE_DIR}/shader.vert -o vert.spv)
add_custom_command(TARGET shaders COMMAND glslc ${CMAKE_SOURCE_DIR}/shader.frag -o frag.spv)

add_executable(rectangle rectangle.cxx)
target_compile_options(rectangle PUBLIC -O2)
add_dependencies(rectangle shaders)
target_link_libraries(rectangle glfw ${GLFW_LIBRARIES} ${Vulkan_LIBRARIES})
