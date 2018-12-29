#include <Catch2/catch.hpp>
#include <cuda/api_wrappers.hpp>

#include <gstl/containers/array.cuh>
#include <string>

using memory_type = gpu::array<char, sizeof(char) * (1 << 25)>;
GPU_DEVICE memory_type local_memory;

inline GPU_DEVICE memory_type& get_memory()
{
	return local_memory;
}

template <class Function>
inline bool launch(Function f)
{
	unsigned int blocks_per_grid = 1u;
	unsigned int threads_per_block = 256u;
	cuda::launch(
		f,
		{ blocks_per_grid, threads_per_block }
	);

	auto status = cuda::outstanding_error::get();
	cuda::throw_if_error(status, "Failed to launch kernel");

	return true;
}
