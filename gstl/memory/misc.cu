#include <gstl/memory/misc.cuh>

namespace gpu
{
	GPU_DEVICE void* align(size_t alignment, size_t size, void*& ptr, size_t& space)
	{
		void* r = nullptr;
		if (size <= space)
		{
			char* p1 = static_cast<char*>(ptr);
			char* p2 = reinterpret_cast<char*>(reinterpret_cast<size_t>(p1 + (alignment - 1)) & -alignment);
			size_t d = static_cast<size_t>(p2 - p1);
			if (d <= space - size)
			{
				r = p2;
				ptr = r;
				space -= d;
			}
		}
		return r;
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T* addressof(T& arg) noexcept
	{
		return reinterpret_cast<T*>(&const_cast<char&>(reinterpret_cast<const volatile char&>(arg)));
	}
}
