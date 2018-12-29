#include <gstl/allocators/linear_allocator.cuh>

#include <gstl/utility/group_result.cuh>

namespace gpu
{
	template <typename T>
	template <unsigned int N>
	GPU_DEVICE linear_allocator<T>::linear_allocator(array<byte_type, N>& fixed_memory) :
		base_allocator { fixed_memory },
		m_offset{ 0 }
	{
	}

	template <typename T>
	GPU_DEVICE linear_allocator<T>::linear_allocator(byte_type* memory, size_type total_size) :
		base_allocator{ memory, total_size },
		m_offset{ 0 }
	{
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T> linear_allocator<T>::allocate(block_t g, size_type n)
	{
		group_result<T*> res_ptr;
		if (g.thread_rank() == 0)
		{
			auto aligned_size = n * sizeof(T);
			if (aligned_size % DEFAULT_PADDING != 0)
				aligned_size += (DEFAULT_PADDING - aligned_size % DEFAULT_PADDING);
		#if defined(GPU_DEBUG_OUT_OF_MEMORY)
			if (m_offset + aligned_size > m_total_size)
			{
				printf("Not enough memory");
				ENSURE(m_offset + aligned_size > m_total_size);
			}
		#endif // GPU_DEBUG_OUT_OF_MEMORY

			byte_type* new_ptr = m_memory + m_offset;
			m_offset += aligned_size;

			res_ptr = group_result<T*>(reinterpret_cast<T*>(new_ptr));
		}

		T* ptr = res_ptr.broadcast(g);
		return allocated_memory<T>(ptr, n);
	}

	template <typename T>
	template <class BlockTile>
	GPU_DEVICE allocated_memory<T> linear_allocator<T>::allocate(BlockTile g, size_type n)
	{
		group_result<T*> res_ptr;
		if (g.thread_rank() == 0)
		{
			auto aligned_size = n * sizeof(T);
			if (aligned_size % DEFAULT_PADDING != 0)
				aligned_size += (DEFAULT_PADDING - aligned_size % DEFAULT_PADDING);
		#if defined(GPU_DEBUG_OUT_OF_MEMORY)
			if (m_offset + aligned_size > m_total_size)
			{
				printf("Not enough memory");
				ENSURE(m_offset + aligned_size > m_total_size);
			}
		#endif // GPU_DEBUG_OUT_OF_MEMORY

			byte_type* new_ptr = m_memory + m_offset;
			m_offset += aligned_size;

			res_ptr = group_result<T*>(reinterpret_cast<T*>(new_ptr));
		}

		T* ptr = res_ptr.broadcast(g);
		return allocated_memory<T>(ptr, n);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T> linear_allocator<T>::allocate(size_type n)
	{
		auto aligned_size = n * sizeof(T);
		if (aligned_size % DEFAULT_PADDING != 0)
			aligned_size += (DEFAULT_PADDING - aligned_size % DEFAULT_PADDING);
	#if defined(GPU_DEBUG_OUT_OF_MEMORY)
		if (m_offset + aligned_size > m_total_size)
		{
			printf("Not enough memory");
			ENSURE(m_offset + aligned_size > m_total_size);
		}
	#endif // GPU_DEBUG_OUT_OF_MEMORY

		byte_type* new_ptr = m_memory + m_offset;
		m_offset += aligned_size;

		T* ptr = reinterpret_cast<T*>(new_ptr);
		return allocated_memory<T>(ptr, n);
	}

	template <typename T>
	GPU_DEVICE void linear_allocator<T>::clear(block_t g)
	{
		if (g.thread_rank() == 0)
			m_offset = 0;
	}

	template <typename T>
	template <class BlockTile>
	GPU_DEVICE void linear_allocator<T>::clear(BlockTile g)
	{
		if (g.thread_rank() == 0)
			m_offset = 0;
	}

	template <typename T>
	GPU_DEVICE void linear_allocator<T>::clear()
	{
		m_offset = 0;
	}

	template <typename T>
	GPU_DEVICE void linear_allocator<T>::deallocate(block_t g, allocated_memory<T>& ptr, size_type n)
	{
	}

	template <typename T>
	template <class BlockTile>
	GPU_DEVICE void linear_allocator<T>::deallocate(BlockTile g, allocated_memory<T>& ptr, size_type n)
	{
	}

	template <typename T>
	GPU_DEVICE void linear_allocator<T>::deallocate(allocated_memory<T>& ptr, size_type n)
	{
	}

	template <typename T>
	GPU_DEVICE typename linear_allocator<T>::size_type linear_allocator<T>::memory_consumed() const
	{
		return m_offset;
	}
}
