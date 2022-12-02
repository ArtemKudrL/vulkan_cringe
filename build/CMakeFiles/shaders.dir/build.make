# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/m410y/Prog/Cpp/vulkan_cringe/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/m410y/Prog/Cpp/vulkan_cringe/build

# Utility rule file for shaders.

# Include any custom commands dependencies for this target.
include CMakeFiles/shaders.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/shaders.dir/progress.make

shaders: CMakeFiles/shaders.dir/build.make
	glslc /home/m410y/Prog/Cpp/vulkan_cringe/src/shader.vert -o vert.spv
	glslc /home/m410y/Prog/Cpp/vulkan_cringe/src/shader.frag -o frag.spv
.PHONY : shaders

# Rule to build all files generated by this target.
CMakeFiles/shaders.dir/build: shaders
.PHONY : CMakeFiles/shaders.dir/build

CMakeFiles/shaders.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/shaders.dir/cmake_clean.cmake
.PHONY : CMakeFiles/shaders.dir/clean

CMakeFiles/shaders.dir/depend:
	cd /home/m410y/Prog/Cpp/vulkan_cringe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/m410y/Prog/Cpp/vulkan_cringe/src /home/m410y/Prog/Cpp/vulkan_cringe/src /home/m410y/Prog/Cpp/vulkan_cringe/build /home/m410y/Prog/Cpp/vulkan_cringe/build /home/m410y/Prog/Cpp/vulkan_cringe/build/CMakeFiles/shaders.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/shaders.dir/depend

