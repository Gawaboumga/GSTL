#include <gstl/utility/group_result.cuh>

#include <gstl/utility/ballot.cuh>

namespace gpu
{
	template <typename T>
	GPU_DEVICE group_result<T>::group_result(T value) :
		m_value(value)
	{
	}

	template <typename T>
	GPU_DEVICE T group_result<T>::broadcast(block_t g)
	{
		return shfl(g, m_value, 0);
	}

	template <typename T>
	template <class BlockTile>
	GPU_DEVICE T group_result<T>::broadcast(BlockTile g)
	{
		return shfl(g, m_value, 0);
	}

	template <typename T>
	GPU_DEVICE group_result<T>::operator T()
	{
		return m_value;
	}
}
