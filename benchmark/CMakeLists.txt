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
