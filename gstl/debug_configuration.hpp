#pragma once

#ifndef GPU_DEBUG_CONFIGURATION
#define GPU_DEBUG_CONFIGURATION

#define GPU_DEBUG_ALL

#define GPU_DEBUG

#ifdef GPU_DEBUG_ALL
	#define GPU_DEBUG_ALGORITHM
	#define GPU_DEBUG_ALLOCATED_MEMORY
	#define GPU_DEBUG_ARRAY
	#define GPU_DEBUG_ATOMIC
	#define GPU_DEBUG_BITS
	#define GPU_DEBUG_FAST_INTEGER
	#define GPU_DEBUG_OUT_OF_MEMORY
	#define GPU_DEBUG_OUT_OF_RANGE
	#define GPU_DEBUG_VECTOR
#endif // GPU_DEBUG_ALL

#endif // GPU_DEBUG_CONFIGURATION
