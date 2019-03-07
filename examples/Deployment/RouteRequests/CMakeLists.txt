add_executable(test_image_service.x
  test_service.cc
)

target_link_libraries(test_image_service.x
PUBLIC
  nvrpc
  deploy-image-client-protos
  gflags
)
