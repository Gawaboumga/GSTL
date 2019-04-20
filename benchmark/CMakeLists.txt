
file(GLOB EXCLUSIVE_SCAN_SOURCES
		${BENCHMARK_FOLDER}main.cpp
		${BENCHMARK_FOLDER}exclusive_scan.cu
)

add_executable(gstl_bechmark_exclusive_scan ${EXCLUSIVE_SCAN_SOURCES})

target_include_directories(gstl_bechmark_exclusive_scan
	PUBLIC
		${GSTL_INCLUDES}
		${THIRDPARTY_INCLUDES}
		${TEST_FOLDER}
)


file(GLOB MERGE_SOURCES
		${BENCHMARK_FOLDER}main.cpp
		${BENCHMARK_FOLDER}merge.cu
)

add_executable(gstl_bechmark_merge ${MERGE_SOURCES})

target_include_directories(gstl_bechmark_merge
	PUBLIC
		${GSTL_INCLUDES}
		${THIRDPARTY_INCLUDES}
		${TEST_FOLDER}
)

file(GLOB REDUCE_SOURCES
		${BENCHMARK_FOLDER}main.cpp
		${BENCHMARK_FOLDER}reduce.cu
)

add_executable(gstl_bechmark_reduce ${REDUCE_SOURCES})

target_include_directories(gstl_bechmark_reduce
	PUBLIC
		${GSTL_INCLUDES}
		${THIRDPARTY_INCLUDES}
		${TEST_FOLDER}
)

file(GLOB STABLE_SORT_SOURCES
		${BENCHMARK_FOLDER}main.cpp
		${BENCHMARK_FOLDER}stable_sort.cu
)

add_executable(gstl_bechmark_stable_sort ${STABLE_SORT_SOURCES})

target_include_directories(gstl_bechmark_stable_sort
	PUBLIC
		${GSTL_INCLUDES}
		${THIRDPARTY_INCLUDES}
		${TEST_FOLDER}
)
