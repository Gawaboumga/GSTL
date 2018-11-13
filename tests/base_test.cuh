#include <Catch2/catch.hpp>
#include <cuda/api_wrappers.hpp>

#include <string>

template <class Function>
inline bool launch(Function f)
{
	unsigned int blocks_per_grid = 1u;
	unsigned int threads_per_block = 256u;
	cuda::launch(
		f,
		{ blocks_per_grid, threads_per_block }
	);

	return true;
}
