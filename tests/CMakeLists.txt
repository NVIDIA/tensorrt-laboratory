include_directories(${GTEST_INCLUDE_DIRS})

add_executable(yais_test
  TestMemory.cc
  TestMemoryStack.cc
  TestPool.cc
)

target_link_libraries(yais_test PRIVATE yais ${GTEST_BOTH_LIBRARIES})

add_test(NAME yais_test COMMAND $<TARGET_FILE:yais_test>)

find_package(benchmark)
if(benchmark_FOUND)
#  add_subdirectory(benchmark)
endif(benchmark_FOUND)