#pragma once

////////////////////////////////////////
// Pre-C++17 support for char8_t.
#undef __U8
#undef U8

#ifdef __cpp_lib_char8_t

// For wrapping string literals in case the compiler lacks u8"x".
#define __U8(text) u8##text
#define U8(text) __U8(text)

#else

// Missing char8_t primitive. To avoid sign extension bugs when reading
// characters 128..255, this must be unsigned (negative characters make
// no sense at all, and any compilers that default to signed characters
// encourage such bugs).
using char8_t = unsigned char;

// For wrapping string literals in case the compiler lacks u8"x".
#define U8(text) (reinterpret_cast<char8_t const*>(text))

namespace std
{
    using u8string = std::basic_string<char8_t>;
    using u8string_view = std::basic_string_view<char8_t>;
}

#endif // __cpp_lib_char8_t

////////////////////////////////////////
// Pre-C++20 minimal span support.
#ifndef __cpp_lib_span

template <typename T>
class minimal_span
{
public:
    minimal_span() = default;

    constexpr minimal_span(minimal_span<T>& s) = default;

    template<typename ContiguousContainerType>
    constexpr minimal_span(ContiguousContainerType&& container)
    :   begin_(std::data(container)),
        end_(begin_ + std::size(container))
    {
    }

    constexpr minimal_span(std::initializer_list<T> i)
    :   begin_(std::data(i)),
        end_(begin_ + std::size(i))
    {
    }

    minimal_span(T* begin, T* end)
    :   begin_(begin),
        end_(end)
    {
    }

    minimal_span(T* begin, size_t elementCount)
    :   begin_(begin),
        end_(begin + elementCount)
    {
    }

    T* data() noexcept { return begin_; }
    T* begin() noexcept { return begin_; }
    T* end() noexcept { return end_; }
    T const* data() const noexcept { return begin_; }
    T const* begin() const noexcept { return begin_; }
    T const* end() const noexcept { return end_; }
    bool empty() const noexcept { return end_ == begin_; }
    size_t size() const noexcept { return end_ - begin_; }
    size_t size_bytes() const noexcept { return sizeof(T) * size(); }
    T& operator[](size_t index) const noexcept { return begin_[index]; }
    minimal_span<T> subspan(size_t index, size_t count) const noexcept { return minimal_span<T>(begin_ + index, begin_ + index + count); }
    minimal_span<T> subrange(size_t begin, size_t end) const noexcept { return minimal_span<T>(begin_ + begin, begin_ + end); }
    minimal_span<T> first(size_t count) const noexcept { return minimal_span<T>(begin_, begin_ + count); }
    minimal_span<T> last(size_t count) const noexcept { return minimal_span<T>(end_ - count, end_); }

    T& front() noexcept { return *begin_; }
    T& back()  noexcept { return *(end_ - 1); }
    T const& front() const noexcept { return *begin_; }
    T const& back()  const noexcept { return *(end_ - 1); }
    T consume_front() noexcept { return *begin_++; }
    T consume_back()  noexcept { return *--end_; }
    void pop_front() noexcept { ++begin_; }
    void pop_back()  noexcept { --end_; }
    void pop_front(size_t n) noexcept { begin_ += n; }
    void pop_back(size_t n)  noexcept { end_ -= n; }

protected:
    T* begin_ = nullptr;
    T* end_ = nullptr;
};

namespace std
{
    template <typename T>
    using span = ::minimal_span<T>;
}

#endif // __cpp_lib_span

////////////////////////////////////////
// Pre-C++23 starts_with and ends_with helper.
// 
// e.g. starts_with(some_string_view, "prefix");
template <typename ContainerType1, typename ContainerType2>
bool starts_with(ContainerType1&& fullSequence, ContainerType2&& prefix)
{
    return fullSequence.size() >= prefix.size()
        && std::equal(fullSequence.begin(), fullSequence.begin() + prefix.size(), prefix.begin(), prefix.end());
}

// e.g. ends_with(some_string_view, "suffix");
template <typename ContainerType1, typename ContainerType2>
bool ends_with(ContainerType1&& fullSequence, ContainerType2&& suffix)
{
    return fullSequence.size() >= suffix.size()
        && std::equal(fullSequence.end() -  + suffix.size(), fullSequence.end(), suffix.begin(), suffix.end());
}
