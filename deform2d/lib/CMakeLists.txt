#set(SOURCE_FILES main.cpp)
aux_source_directory(. LIB_DIR)
cuda_add_executable(deform2d_new_pytorch ${LIB_DIR})
#add_executable(deform3d_new_pytorch ${SOURCE_FILES})