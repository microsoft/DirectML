// MIT License
// Copyright (c) Microsoft Corporation
//
// NOTE: this is a modified version of StackAllocator.h from the DirectML execution provider in the onnxruntime repo (licensed under MIT):
// https://github.com/microsoft/onnxruntime/blob/8d737f977056444a307f1b7f0bcd402fba62d790/onnxruntime/core/providers/dml/DmlExecutionProvider/src/External/DirectMLHelpers/ApiHelpers.h#L98

#pragma once

#include <deque>
#include <wil/result.h>
#include <onnxruntime_cxx_api.h>
#ifndef WIN32
#include <sys/mman.h>
#include <cerrno>
#endif

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
        static_assert(std::is_trivial_v<T> || std::is_same_v<T, Ort::Float16_t>, "This class does not invoke constructors.");

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
#ifdef WIN32
            this->data = VirtualAlloc(nullptr, this->capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            THROW_LAST_ERROR_IF_NULL(this->data);
#else
            this->data = mmap(nullptr, this->capacity, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (this->data == MAP_FAILED)
            {
                switch (errno)
                {
                case EINVAL: THROW_HR(E_INVALIDARG); break;
                case ENOMEM: THROW_HR(E_OUTOFMEMORY); break;
                case EMFILE: THROW_HR(E_OUTOFMEMORY); break;
                default: THROW_HR(E_UNEXPECTED); break;
                }
            }
#endif
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
#ifdef WIN32
                (void)VirtualFree(data, 0, MEM_RELEASE);
#else
                int result = munmap(this->data, this->capacity);
#endif
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
            return static_cast<std::byte*>(data) + alignedOffset;
        }
    };

    std::deque<Bucket> m_buckets;
};