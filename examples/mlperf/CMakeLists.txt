#add_executable(mlperf-demo.x
#    demo.cc
#)

#target_link_libraries(mlperf-demo.x
#)

add_executable(mlperf-infer.x
    main.cc
)

target_link_libraries(mlperf-infer.x
   tensorrt-playground::tensorrt
   gflags
)

add_executable(mlperf-test.x
    test.cc
)

target_link_libraries(mlperf-test.x
   tensorrt-playground::tensorrt
   gflags
)
