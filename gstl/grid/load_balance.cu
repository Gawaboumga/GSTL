#include <gstl/grid/load_balance.cuh>

namespace gpu
{
	template <class RandomIt>
	GPU_DEVICE pair<RandomIt, RandomIt> load_balance(grid_t grid, RandomIt first, RandomIt last)
	{
		offset_t len = distance(first, last);

		offset_t block_quantity = len / grid.grid_dim().x;
		offset_t bonus = len - block_quantity * grid.grid_dim().x;

		gpu::block_t block = gpu::this_thread_block();
		offset_t offset = block_quantity * block.group_index().x;

		if (block.group_index().x >= grid.grid_dim().x - bonus)
		{
			offset += block.group_index().x - (grid.grid_dim().x - bonus);
			block_quantity += 1;
		}
		
		return { first + offset, first + offset + block_quantity };
	}
}
