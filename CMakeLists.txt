# CMakeList.txt: LinearRegression 的 CMake 專案，在此包含來源及定義
# 專案專屬邏輯。
#
cmake_minimum_required (VERSION 3.8)

project ("RandomForest" LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)	
set(CMAKE_CUDA_STANDARD 17)

# 將來源新增到此專案的可執行檔。
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(. SRC_LIST)
add_executable (${PROJECT_NAME} ${SRC_LIST})


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

# TODO: 新增測試，並視需要安裝目標。
