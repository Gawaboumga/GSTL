#pragma once

#ifndef GPU_ASSERT
#define GPU_ASSERT

#include <gstl/debug_configuration.hpp>

#include <cassert>
#include <cstdio>

__device__ inline void INTERNAL_ENSURE(bool ok, const char* message, const char* file, int line)
{
#ifdef GPU_DEBUG
	if (!ok)
	{
		if (message)
			printf("%s\n", message);

		int thid = threadIdx.x + blockIdx.x * blockDim.x;
		printf("%d %s %d | ", thid, file, line);
		assert(false);
	}
#endif
}

#define CREATE_2(x, msg) INTERNAL_ENSURE(x, msg, __FILE__, __LINE__)
#define CREATE_1(x) CREATE_2(x, nullptr);
#define CREATE_0() CREATE_1(true);

#define FUNC_CHOOSER(_f1, _f2, _f3, ...) _f3
#define FUNC_RECOMPOSER(argsWithParentheses) FUNC_CHOOSER argsWithParentheses
#define CHOOSE_FROM_ARG_COUNT(...) FUNC_RECOMPOSER((__VA_ARGS__, CREATE_2, CREATE_1, ))
#define NO_ARG_EXPANDER() ,,CREATE_0
#define MACRO_CHOOSER(...) CHOOSE_FROM_ARG_COUNT(NO_ARG_EXPANDER __VA_ARGS__ ())
#define ENSURE(...) MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#endif // GPU_ASSERT
