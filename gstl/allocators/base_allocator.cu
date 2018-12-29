#include <gstl/allocators/base_allocator.cuh>

namespace gpu
{
	template <class DerivedAllocator>
	GPU_DEVICE typename base_allocator<DerivedAllocator>::byte_type* base_allocator<DerivedAllocator>::begin() const
	{
		return m_memory;
	}

	template <class DerivedAllocator>
	GPU_DEVICE typename base_allocator<DerivedAllocator>::byte_type* base_allocator<DerivedAllocator>::end() const
	{
		return m_memory + m_total_size;
	}

	template <class DerivedAllocator>
	template <unsigned int N>
	GPU_DEVICE base_allocator<DerivedAllocator>::base_allocator(array<byte_type, N>& fixed_memory) :
		m_memory{ fixed_memory.data() },
		m_total_size{ fixed_memory.size() }
	{
	}

	template <class DerivedAllocator>
	GPU_DEVICE base_allocator<DerivedAllocator>::base_allocator(byte_type* memory, size_type total_size) :
		m_memory{ memory },
		m_total_size{ total_size }
	{
	}

	template <class DerivedAllocator>
	template <typename T>
	GPU_DEVICE allocated_memory<T> base_allocator<DerivedAllocator>::allocate(block_t g, size_type n)
	{
		return static_cast<DerivedAllocator*>(this)->allocate<T>(g, n);
	}

	template <class DerivedAllocator>
	template <class BlockTile, typename T>
	GPU_DEVICE allocated_memory<T> base_allocator<DerivedAllocator>::allocate(BlockTile g, size_type n)
	{
		return static_cast<DerivedAllocator*>(this)->allocate<T>(g, n);
	}

	template <class DerivedAllocator>
	template <typename T>
	GPU_DEVICE allocated_memory<T> base_allocator<DerivedAllocator>::allocate(size_type n)
	{
		return static_cast<DerivedAllocator*>(this)->allocate<T>(n);
	}

	template <class DerivedAllocator>
	GPU_DEVICE void base_allocator<DerivedAllocator>::clear(block_t g)
	{
		return static_cast<DerivedAllocator*>(this)->clear(g);
	}

	template <class DerivedAllocator>
	template <class BlockTile>
	GPU_DEVICE void base_allocator<DerivedAllocator>::clear(BlockTile g)
	{
		return static_cast<DerivedAllocator*>(this)->clear(g);
	}

	template <class DerivedAllocator>
	GPU_DEVICE void base_allocator<DerivedAllocator>::clear()
	{
		return static_cast<DerivedAllocator*>(this)->clear();
	}

	template <class DerivedAllocator>
	template <typename T>
	GPU_DEVICE void base_allocator<DerivedAllocator>::deallocate(block_t g, allocated_memory<T>& ptr, size_type n)
	{
		static_cast<DerivedAllocator*>(this)->deallocate(g, ptr);
	#ifdef GPU_DEBUG_ALLOCATED_MEMORY
		if (g.thread_rank() == 0)
			ptr.invalidate();
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <class DerivedAllocator>
	template <class BlockTile, typename T>
	GPU_DEVICE void base_allocator<DerivedAllocator>::deallocate(BlockTile g, allocated_memory<T>& ptr, size_type n)
	{
		static_cast<DerivedAllocator*>(this)->deallocate(g, ptr);
	#ifdef GPU_DEBUG_ALLOCATED_MEMORY
		if (g.thread_rank() == 0)
			ptr.invalidate();
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <class DerivedAllocator>
	template <typename T>
	GPU_DEVICE void base_allocator<DerivedAllocator>::deallocate(allocated_memory<T>& ptr, size_type n)
	{
		static_cast<DerivedAllocator*>(this)->deallocate(ptr);
	#ifdef GPU_DEBUG_ALLOCATED_MEMORY
		ptr.invalidate();
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}
}
