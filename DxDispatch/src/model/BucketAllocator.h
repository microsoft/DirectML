// MIT License
// Copyright (c) Microsoft Corporation
//
// NOTE: this is a modified version of StackAllocator.h from the DirectML execution provider in the onnxruntime repo (licensed under MIT):
// https://github.com/microsoft/onnxruntime/blob/8d737f977056444a307f1b7f0bcd402fba62d790/onnxruntime/core/providers/dml/DmlExecutionProvider/src/External/DirectMLHelpers/ApiHelpers.h#L98

#pragma once

#include <deque>
#include <wil/result.h>

class BucketAllocator
{
public:
    BucketAllocator() { m_buckets.emplace_back(1); }

    BucketAllocator(const BucketAllocator&) = delete;
    BucketAllocator& operator=(const BucketAllocator&) = delete;

    BucketAllocator& operator=(BucketAllocator&& other)
    {
        if (this != &other)
        {
            std::swap(m_buckets, other.m_buckets);
        }
        return *this;
    }

    BucketAllocator(BucketAllocator&& other)
    {
        std::swap(m_buckets, other.m_buckets);
    }

    template <typename T>
    T* Allocate(size_t count = 1)
    {
        static_assert(std::is_trivial_v<T> || std::is_same_v<T, half_float::half>, "This class does not invoke constructors.");

        size_t sizeInBytes = sizeof(T) * count;

        void* memory = m_buckets.back().TryAllocate(sizeInBytes, alignof(T));
        if (!memory)
        {
            m_buckets.emplace_back(sizeInBytes);
            memory = m_buckets.back().TryAllocate(sizeInBytes, alignof(T));
        }

        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }

private:
    struct Bucket
    {
        void* data = nullptr;
        size_t allocatedSize = 0;
        size_t capacity = 0;

        explicit Bucket(size_t minimumSize)
        {
            this->allocatedSize = 0;
            this->capacity = RoundUpToMultiple<size_t>(minimumSize, 4096);
            this->data = VirtualAlloc(nullptr, this->capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            THROW_LAST_ERROR_IF_NULL(this->data);
        }

        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;

        Bucket& operator=(Bucket&& other)
        {
            std::swap(data, other.data);
            std::swap(allocatedSize, other.allocatedSize);
            std::swap(capacity, other.capacity);
            return *this;
        }

        Bucket(Bucket&& other)
        {
            std::swap(data, other.data);
            std::swap(allocatedSize, other.allocatedSize);
            std::swap(capacity, other.capacity);
        }

        ~Bucket()
        {
            if (data)
            {
                (void)VirtualFree(data, 0, MEM_RELEASE);
            }
        }

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

    std::deque<Bucket> m_buckets;
};