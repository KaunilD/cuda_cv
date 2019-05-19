cmake_minimum_required (VERSION 3.8)

find_package( CUDA REQUIRED )
find_package( CUCV REQUIRED )

add_executable(
	example "example.h" "example.cpp"
)
target_link_libraries(example libcudacv)