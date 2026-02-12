cmake_minimum_required(VERSION 3.16)
project(adam_deploy_real)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# include paths
include_directories(
    /home/fantao/Documents/MOT_HUB/dds_cpp/include
    /home/fantao/dds-cpp-install/include/ddscxx
    /home/fantao/cyclonedds-install/include
    # /usr/local/include/ddscxx
    /usr/local/include/iceoryx/v2.0.2
)

link_directories(/home/fantao/third_party/sdk_cpp/lib/
                /home/fantao/dds-cpp-install/lib/)
# Torch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libtorch/share/cmake")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS}
                    /home/fantao/Documents/MOT_HUB/dds_cpp/include/unitree/idl/pnd_adam/
)

# sources
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST}
/home/fantao/Documents/MOT_HUB/dds_cpp/example/lite/src_idl/LowCmd_.cpp
/home/fantao/Documents/MOT_HUB/dds_cpp/example/lite/src_idl/LowState_.cpp
/home/fantao/Documents/MOT_HUB/dds_cpp/example/lite/src_idl/IMUState_.cpp
/home/fantao/Documents/MOT_HUB/dds_cpp/example/lite/src_idl/MotorCmd_.cpp
/home/fantao/Documents/MOT_HUB/dds_cpp/example/lite/src_idl/MotorState_.cpp
)

# yaml-cpp
set(YAML_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp)
add_subdirectory(${YAML_CPP_DIR})

# link libraries using absolute paths
target_link_libraries(${PROJECT_NAME} unitree_sdk2 ddsc ddscxx ${TORCH_LIBRARIES}
    yaml-cpp::yaml-cpp
    pthread
)
