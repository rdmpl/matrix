
cmake_minimum_required(VERSION 2.8)
project(PSMLIBAPI)
set(CMAKE_CXX_STANDARD 17)
find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=6 --dirty --always --tags
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE  GIT_REPO_VERSION
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd --date=short
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE  GIT_REPO_DATE
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE  GIT_REPO_HASH
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(WRITE ${PROJECT_BINARY_DIR}/git_version.h
	"#define GIT_VERSION \"${GIT_REPO_VERSION}\"\n#define GIT_DATE \"${GIT_REPO_DATE}\"\n#define GIT_HASH \"${GIT_REPO_HASH}\"\n"
)
include_directories(${PROJECT_SOURCE_DIR}/src)

if (UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mssse3 -fPIC -g -Wall -Werror -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function")
endif()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

AUX_SOURCE_DIRECTORY(${CMAKE_SOURCE_DIR}/cmd CMD)

macro(add_unit_test name)
  add_executable(${name} ${PROJECT_SOURCE_DIR}/test/${name}.cc)
  target_link_libraries(${name} ${LINK_TARGETS})
  add_test(NAME ${name} COMMAND ${name} --benchmark_min_time=0.01)
endmacro(add_unit_test)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/test TEST_SRCS)
foreach(FILE ${TEST_SRCS})
string(REGEX REPLACE ".+/\(.+\)\\..*" "\\1" TEST ${FILE})
add_unit_test(${TEST})	
endforeach(FILE ${PROJECT_SOURCE_DIR}/test)

