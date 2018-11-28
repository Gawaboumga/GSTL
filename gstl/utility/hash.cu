#include <gstl/utility/hash.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Size, size_t = sizeof(Size) * 8>
		struct murmur2_or_cityhash;

		template <class Size>
		struct murmur2_or_cityhash<Size, 32>
		{
			Size operator()(const void* key, Size len)
			{
				const Size m = 0x5bd1e995;
				const Size r = 24;
				Size h = len;
				const unsigned char* data = static_cast<const unsigned char*>(key);
				for (; len >= 4; data += 4, len -= 4)
				{
					Size k = *reinterpret_cast<const Size*>(data);
					k *= m;
					k ^= k >> r;
					k *= m;
					h *= m;
					h ^= k;
				}

				switch (len)
				{
					case 3:
						h ^= data[2] << 16;
					case 2:
						h ^= data[1] << 8;
					case 1:
						h ^= data[0];
						h *= m;
				}

				h ^= h >> 13;
				h *= m;
				h ^= h >> 15;
				return h;
			}
		};

		template <typename T, size_t = sizeof(T) / sizeof(size_t)>
		struct scalar_hash;

		template <typename T>
		struct scalar_hash<T, 0>
		{
			GPU_DEVICE size_t operator()(T v) const noexcept
			{
				union
				{
					T t;
					size_t a;
				} u;
				u.a = 0;
				u.t = v;
				return u.a;
			}
		};

		template <typename T>
		struct scalar_hash<T, 1>
		{
			GPU_DEVICE size_t operator()(T v) const noexcept
			{
				union
				{
					T t;
					size_t a;
				} u;
				u.t = v;
				return u.a;
			}
		};

		template <typename T>
		struct scalar_hash<T, 2>
		{
			GPU_DEVICE size_t operator()(T v) const noexcept
			{
				union
				{
					T t;
					struct
					{
						size_t a;
						size_t b;
					} s;
				} u;
				u.t = v;
				return murmur2_or_cityhash<size_t>()(&u, sizeof(u));
			}
		};

		template <typename T>
		struct scalar_hash<T, 3>
		{
			GPU_DEVICE size_t operator()(T v) const noexcept
			{
				union
				{
					T t;
					struct
					{
						size_t a;
						size_t b;
						size_t c;
					} s;
				} u;
				u.t = v;
				return murmur2_or_cityhash<size_t>()(&u, sizeof(u));
			}
		};

		template <typename T>
		struct scalar_hash<T, 4>
		{
			GPU_DEVICE size_t operator()(T v) const noexcept
			{
				union
				{
					T t;
					struct
					{
						size_t a;
						size_t b;
						size_t c;
						size_t d;
					} s;
				} u;
				u.t = v;
				return murmur2_or_cityhash<size_t>()(&u, sizeof(u));
			}
		};
	}

	template <>
	struct hash<bool>
	{
		GPU_DEVICE size_t operator()(bool v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<char>
	{
		GPU_DEVICE size_t operator()(char v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<signed char>
	{
		GPU_DEVICE size_t operator()(signed char v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<unsigned char>
	{
		GPU_DEVICE size_t operator()(unsigned char v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<char16_t>
	{
		GPU_DEVICE size_t operator()(char16_t v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<char32_t>
	{
		GPU_DEVICE size_t operator()(char32_t v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<short>
	{
		GPU_DEVICE size_t operator()(short v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<unsigned short>
	{
		GPU_DEVICE size_t operator()(unsigned short v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<int>
	{
		GPU_DEVICE size_t operator()(int v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<unsigned int>
	{
		GPU_DEVICE size_t operator()(unsigned int v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<long>
	{
		GPU_DEVICE size_t operator()(long v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<long long> : public detail::scalar_hash<long long>
	{
	};

	template <>
	struct hash<unsigned long>
	{
		GPU_DEVICE size_t operator()(unsigned long v) const noexcept
		{
			return static_cast<size_t>(v);
		}
	};

	template <>
	struct hash<unsigned long long> : public detail::scalar_hash<unsigned long long>
	{
	};

	template <>
	struct hash<float> : public detail::scalar_hash<float>
	{
		GPU_DEVICE size_t operator()(float v) const noexcept
		{
			if (v == 0.0)
				return 0;

			return detail::scalar_hash<float>::operator()(v);
		}
	};

	template <>
	struct hash<double> : public detail::scalar_hash<double>
	{
		GPU_DEVICE size_t operator()(double v) const noexcept
		{
			if (v == 0.0)
				return 0;

			return detail::scalar_hash<double>::operator()(v);
		}
	};

	template <>
	struct hash<std::nullptr_t>
	{
		GPU_DEVICE size_t operator()(std::nullptr_t) const noexcept
		{
			return 662607004ull;
		}
	};

	template <typename T>
	struct hash<T*>
	{
		GPU_DEVICE size_t operator()(T* v) const noexcept
		{
			return hash<uintptr_t>()(v);
		}
	};
}
