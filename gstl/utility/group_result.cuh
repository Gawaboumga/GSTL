#pragma once

#ifndef GPU_UTILITY_GROUP_RESULT_HPP
#define GPU_UTILITY_GROUP_RESULT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <typename T>
	class group_result
	{
		public:
			group_result() = default;
			GPU_DEVICE group_result(T value);
			group_result(group_result&&) = default;

			GPU_DEVICE T broadcast(block_t g);
			template <class BlockTile>
			GPU_DEVICE T broadcast(BlockTile g);

			GPU_DEVICE operator T();
			group_result& operator=(group_result&&) = default;

		private:
			T m_value;
	};
}

#include <gstl/utility/group_result.cu>

#endif // GPU_UTILITY_GROUP_RESULT_HPP
