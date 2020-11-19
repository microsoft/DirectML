//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

template <size_t Size>
class StackAllocator
{
public:
    StackAllocator() = default;

    // Non-copiable, non-movable
    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    StackAllocator(StackAllocator&&) = delete;
    StackAllocator& operator=(StackAllocator&&) = delete;

    template <typename T>
    T* Allocate(size_t count = 1)
    {
        static_assert(std::is_trivial_v<T>,
            "This class may only be used to allocate trivial types, as it does not invoke constructors.");

        // Allocate from the fixed bucket before falling back to dynamic
        Bucket* lastBucket = m_dynamic.empty() ? static_cast<Bucket*>(&m_fixed) : static_cast<Bucket*>(&m_dynamic.back());

        size_t sizeInBytes = sizeof(T) * count;
        void* memory = lastBucket->TryAllocate(sizeInBytes, alignof(T));

        if (!memory)
        {
            // Not enough capacity remains; allocate a new dynamic bucket
            size_t minimumSize = sizeInBytes;
            m_dynamic.emplace_back(minimumSize);

            memory = m_dynamic.back().TryAllocate(sizeInBytes, alignof(T));
        }

        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }

    void Reset()
    {
        m_fixed.allocatedSize = 0;
        m_dynamic.clear();
    }

private:
    struct Bucket
    {
        void* data;
        size_t allocatedSize;
        size_t capacity;

        Bucket() = default;

        // Non-copiable, non-movable
        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;
        Bucket(Bucket&&) = delete;
        Bucket& operator=(Bucket&&) = delete;

        template <typename T>
        static T RoundUpToMultiple(T value, T multiple)
        {
            static_assert(std::is_integral_v<T>);

            T remainder = value % multiple;
            if (remainder != 0)
            {
                value += multiple - remainder;
            }

            return value;
        }

        void* TryAllocate(size_t sizeInBytes, size_t alignment)
        {
            size_t alignedOffset = RoundUpToMultiple(allocatedSize, alignment);
            size_t newAllocatedSize = alignedOffset + sizeInBytes;

            if (newAllocatedSize > capacity)
            {
                return nullptr; // Not enough capacity
            }

            allocatedSize = newAllocatedSize;
            return static_cast<byte*>(data) + alignedOffset;
        }
    };

    struct FixedBucket : Bucket
    {
        std::array<byte, Size> stack;

        FixedBucket()
        {
            this->data = stack.data();
            this->allocatedSize = 0;
            this->capacity = stack.size();
        }
    };

    struct DynamicBucket : Bucket
    {
        explicit DynamicBucket(size_t minimumSize)
        {
            this->allocatedSize = 0;
            this->capacity = RoundUpToMultiple<size_t>(minimumSize, 4096); // Round up to nearest page granularity

            this->data = VirtualAlloc(nullptr, this->capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            ThrowIfNull(this->data);
        }

        ~DynamicBucket()
        {
            if (this->data)
            {
                (void)VirtualFree(this->data, 0, MEM_RELEASE);
            }
        }
    };

    // This allocator first retrieves memory from a fixed-size stack-allocated array before falling back to dynamically
    // allocated memory if the fixed stack array is exhausted.
    FixedBucket m_fixed;
    std::deque<DynamicBucket> m_dynamic;
};

template <size_t StackSize>
class DmlTypeConverter : public StackAllocator<StackSize>
{
public:
    template <typename T>
    T* AllocateUsing(const T& src)
    {
        T* t = this->Allocate<T>();
        *t = src;
        return t;
    }

    template <typename T>
    T* AllocateArrayUsing(dml::Span<const T> src)
    {
        T* t = this->Allocate<T>(src.size());
        std::copy(src.begin(), src.end(), t);
        return t;
    }

    DML_BUFFER_BINDING Convert(const DmlBufferBinding& src)
    {
        DML_BUFFER_BINDING dst;
        dst.Buffer = src.buffer;
        dst.Offset = src.offset;
        dst.SizeInBytes = src.sizeInBytes;
        return dst;
    }

    DML_BUFFER_ARRAY_BINDING Convert(const DmlBufferArrayBinding& src)
    {
        const size_t count = src.bindings.size();
        DML_BUFFER_BINDING* bindings = this->Allocate<DML_BUFFER_BINDING>(count);
        for (size_t i = 0; i < count; ++i)
        {
            bindings[i] = Convert(src.bindings[i]);
        }

        DML_BUFFER_ARRAY_BINDING dst;
        dst.BindingCount = static_cast<UINT>(count);
        dst.Bindings = bindings;
        return dst;
    }

    DML_BINDING_DESC ToBindingDesc(const DmlNoneBinding& src)
    {
        DML_BINDING_DESC dst;
        dst.Type = DML_BINDING_TYPE_NONE;
        dst.Desc = nullptr;
        return dst;
    }

    DML_BINDING_DESC ToBindingDesc(const DmlBufferBinding& src)
    {
        auto* binding = this->Allocate<DML_BUFFER_BINDING>();
        *binding = Convert(src);

        DML_BINDING_DESC dst;
        dst.Type = DML_BINDING_TYPE_BUFFER;
        dst.Desc = binding;
        return dst;
    }

    DML_BINDING_DESC ToBindingDesc(const DmlBufferArrayBinding& src)
    {
        auto* binding = this->Allocate<DML_BUFFER_ARRAY_BINDING>();
        *binding = Convert(src);

        DML_BINDING_DESC dst;
        dst.Type = DML_BINDING_TYPE_BUFFER_ARRAY;
        dst.Desc = binding;
        return dst;
    }
};
