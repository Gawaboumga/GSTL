#include <gstl/memory/allocator_traits.cuh>

#include <gstl/utility/limits.cuh>

namespace gpu
{
	template <class Allocator>
	GPU_DEVICE typename allocator_traits<Allocator>::pointer allocator_traits<Allocator>::allocate(Allocator& a, size_type n)
	{
		return a.allocate(n);
	}

	template <class Allocator>
	template <class Thread>
	GPU_DEVICE typename allocator_traits<Allocator>::pointer allocator_traits<Allocator>::allocate(Thread g, Allocator& a, size_type n)
	{
		return a.allocate(g, n);
	}

	template <class Allocator>
	GPU_DEVICE typename allocator_traits<Allocator>::pointer allocator_traits<Allocator>::allocate(Allocator& a, size_type n, const_void_pointer hint)
	{
		return a.allocate(n, hint);
	}

	template <class Allocator>
	template <class Thread>
	GPU_DEVICE typename allocator_traits<Allocator>::pointer allocator_traits<Allocator>::allocate(Thread g, Allocator& a, size_type n, const_void_pointer hint)
	{
		return a.allocate(g, n, hint);
	}

	template <class Allocator>
	template <class T, class... Args>
	GPU_DEVICE void allocator_traits<Allocator>::construct(Allocator& a, T* p, Args&&... args)
	{
		::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
	}

	template <class Allocator>
	template <class Thread, class T, class... Args>
	GPU_DEVICE void allocator_traits<Allocator>::construct(Thread g, Allocator& a, T* p, Args&&... args)
	{
		::new (static_cast<void*>(p)) T(g, std::forward<Args>(args)...);
	}

	template <class Allocator>
	template <class T, class... Args>
	GPU_DEVICE void allocator_traits<Allocator>::construct_n(Allocator& a, T* p, size_type n, Args&&... args)
	{
		do
		{
			construct(a, p, std::forward<Args>(args)...);
			++p;
			--n;
		} while (n > 0);
	}

	template <class Allocator>
	template <class Thread, class T, class... Args>
	GPU_DEVICE void allocator_traits<Allocator>::construct_n(Thread g, Allocator& a, T* p, size_type n, Args&&... args)
	{
		do
		{
			construct(a, p + g.thread_rank(), std::forward<Args>(args)...);
			p += g.size();
			n -= g.size();
		} while (n > 0);
	}

	template <class Allocator>
	template <class ForwardIt, class T, class... Args>
	GPU_DEVICE void allocator_traits<Allocator>::construct_range(Allocator& a, T* p, ForwardIt first, ForwardIt last)
	{
		for (; first != last; ++first, ++p)
			construct(a, p, *first);
	}

	template <class Allocator>
	template <class Thread, class ForwardIt, class T, class... Args>
	GPU_DEVICE void allocator_traits<Allocator>::construct_range(Thread g, Allocator& a, T* p, ForwardIt first, ForwardIt last)
	{
		auto len = distance(first, last);
		do
		{
			construct(a, p + g.thread_rank(), *(first + g.thread_rank()));
			p += g.size();
			first += g.size();
			len -= g.size();
		} while (len > 0);
	}

	template <class Allocator>
	GPU_DEVICE void allocator_traits<Allocator>::deallocate(Allocator& a, pointer p, size_type n)
	{
		return a.deallocate(p, n);
	}

	template <class Allocator>
	template <class T>
	GPU_DEVICE void allocator_traits<Allocator>::destroy(Allocator& a, T* p)
	{
		p->~T();
	}

	template <class Allocator>
	GPU_DEVICE typename allocator_traits<Allocator>::size_type allocator_traits<Allocator>::max_size(const Allocator& a)
	{
		return numeric_limits<size_type>::max() / sizeof(value_type);
	}
}
