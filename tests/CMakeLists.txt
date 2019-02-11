
file(GLOB TEST_SOURCES
		${TEST_FOLDER}main.cpp
		${TEST_FOLDER}algorithms/**.cpp
		${TEST_FOLDER}algorithms/**.cu
		${TEST_FOLDER}algorithms/sort/**.cpp
		${TEST_FOLDER}algorithms/sort/**.cu
		${TEST_FOLDER}allocators/**.cpp
		${TEST_FOLDER}allocators/**.cu
		${TEST_FOLDER}containers/**.cpp
		${TEST_FOLDER}containers/**.cu
		${TEST_FOLDER}containers/concurrent/**.cpp
		${TEST_FOLDER}containers/concurrent/**.cu
		${TEST_FOLDER}containers/concurrent/hash_tables/**.cpp
		${TEST_FOLDER}containers/concurrent/hash_tables/**.cu
		${TEST_FOLDER}containers/concurrent/hash_tables/fixed/**.cpp
		${TEST_FOLDER}containers/concurrent/hash_tables/fixed/**.cu
		${TEST_FOLDER}numeric/**.cpp
		${TEST_FOLDER}numeric/**.cu
		${TEST_FOLDER}random/**.cpp
		${TEST_FOLDER}random/**.cu
		${TEST_FOLDER}utility/**.cpp
		${TEST_FOLDER}utility/**.cu
		${TEST_FOLDER}grid/algorithms/**.cpp
		${TEST_FOLDER}grid/algorithms/**.cu
		${TEST_FOLDER}grid/numeric/**.cpp
		${TEST_FOLDER}grid/numeric/**.cu
		${TEST_FOLDER}kernel/algorithms/**.cpp
		${TEST_FOLDER}kernel/algorithms/**.cu
		${TEST_FOLDER}kernel/numeric/**.cpp
		${TEST_FOLDER}kernel/numeric/**.cu
)

add_executable(gstl_tests ${TEST_SOURCES})

target_include_directories(gstl_tests
	PUBLIC
		${GSTL_INCLUDES}
		${THIRDPARTY_INCLUDES}
		${TEST_FOLDER}
)

