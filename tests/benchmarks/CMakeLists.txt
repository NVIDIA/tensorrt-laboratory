


add_executable(yais-benchmarks
  main.cc
  BenchPool.cc
  BenchThreadPool.cc
)

target_link_libraries(yais-benchmarks PRIVATE yais)
target_link_libraries(yais-benchmarks PRIVATE benchmark)

add_test(NAME yais-benchmarks COMMAND $<TARGET_FILE:yais-benchmarks>)