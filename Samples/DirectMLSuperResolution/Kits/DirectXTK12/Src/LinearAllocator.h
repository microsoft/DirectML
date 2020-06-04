//--------------------------------------------------------------------------------------
// LinearAllocator.h
//
// A linear allocator. When Allocate is called it will try to return you a pointer into
// existing graphics memory. If there is no space left from what is allocated, more 
// pages are allocated on-the-fly.
//
// Each allocation must be smaller or equal to pageSize. It is not necessary but is most
// efficient for the sizes to be some fraction of pageSize. pageSize does not determine 
// the size of the physical pages underneath the virtual memory (that's given by the
// XMemAttributes) but is how much additional memory the allocator should allocate 
// each time you run out of space.
//
// preallocatePages specifies how many pages to initially allocate. Specifying zero will 
// preallocate two pages by default.
//
// This class is NOT thread safe. You should protect this with the appropriate sync
// primitives or, even better, use one linear allocator per thread.
//
// Pages are freed once the GPU is done with them. As such, you need to specify when a 
// page is in use and when it is no longer in use. Use RetirePages to prompt the 
// allocator to check if pages are no longer being used by the GPU. Use InsertFences to
// mark all used pages as in-use by the GPU, removing them from the available pages 
// list. It is recommended you call RetirePages and InsertFences once a frame, usually
// just before Present().
//
// Why is RetirePages decoupled from InsertFences? It's possible that you might want to 
// reclaim pages more frequently than locking used pages. For example, if you find the 
// allocator is burning through pages too quickly you can call RetirePages to reclaim 
// some that the GPU has finished with. However, this adds additional CPU overhead so it
// is left to you to decide. In most cases this is sufficient:
//
//      allocator.RetirePages();
//      allocator.InsertFences( pContext, 0 );
//      Present(...);
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#pragma once

#include <atomic>


namespace DirectX
{
    class LinearAllocatorPage
    {
    public:
        LinearAllocatorPage() noexcept;

        LinearAllocatorPage(LinearAllocatorPage&&) = delete;
        LinearAllocatorPage& operator= (LinearAllocatorPage&&) = delete;

        LinearAllocatorPage(LinearAllocatorPage const&) = delete;
        LinearAllocatorPage& operator=(LinearAllocatorPage const&) = delete;

        size_t Suballocate(_In_ size_t size, _In_ size_t alignment);

        void* BaseMemory() const { return mMemory; }
        ID3D12Resource* UploadResource() const { return mUploadResource.Get(); }
        D3D12_GPU_VIRTUAL_ADDRESS GpuAddress() const { return mGpuAddress; }
        size_t BytesUsed() const { return mOffset; }
        size_t Size() const { return mSize; }

        void AddRef() { mRefCount.fetch_add(1); }
        int32_t RefCount() const { return mRefCount.load(); }
        void Release();

    protected:
        friend class LinearAllocator;

        LinearAllocatorPage*                    pPrevPage;
        LinearAllocatorPage*                    pNextPage;

        void*                                   mMemory;
        Microsoft::WRL::ComPtr<ID3D12Resource>  mUploadResource;
        Microsoft::WRL::ComPtr<ID3D12Fence>     mFence;
        uint64_t                                mPendingFence;
        D3D12_GPU_VIRTUAL_ADDRESS               mGpuAddress;
        size_t                                  mOffset;
        size_t                                  mSize;

    private:
        std::atomic<int32_t>                    mRefCount;
    };

    class LinearAllocator
    {
    public:        
        // These values will be rounded up to the nearest 64k.
        // You can specify zero for incrementalSizeBytes to increment
        // by 1 page (64k).
        LinearAllocator(
            _In_ ID3D12Device* pDevice,
            _In_ size_t pageSize,
            _In_ size_t preallocateBytes = 0);

        LinearAllocator(LinearAllocator&&) = default;
        LinearAllocator& operator= (LinearAllocator&&) = default;

        LinearAllocator(LinearAllocator const&) = delete;
        LinearAllocator& operator=(LinearAllocator const&) = delete;

        ~LinearAllocator();

        LinearAllocatorPage* FindPageForAlloc(_In_ size_t requestedSize, _In_ size_t alignment);

        // Call this at least once a frame to check if pages have become available.
        void RetirePendingPages();

        // Call this after you submit your work to the driver.
        // (e.g. immediately before Present.)
        void FenceCommittedPages(_In_ ID3D12CommandQueue* commandQueue);

        // Throws away all currently unused pages
        void Shrink();

        // Statistics
        size_t CommittedPageCount() const { return m_numPending; }
        size_t TotalPageCount() const { return m_totalPages; }
        size_t CommittedMemoryUsage() const { return m_numPending * m_increment; }
        size_t TotalMemoryUsage() const { return m_totalPages * m_increment; }
        size_t PageSize() const { return m_increment; }

#if defined(_DEBUG) || defined(PROFILE)
        // Debug info
        const wchar_t* GetDebugName() const { return m_debugName.c_str(); }
        void SetDebugName(const wchar_t* name);
        void SetDebugName(const char* name);
#endif

    private:
        Microsoft::WRL::ComPtr<ID3D12Device>    m_device;
        LinearAllocatorPage*                    m_pendingPages; // Pages in use by the GPU
        LinearAllocatorPage*                    m_usedPages;    // Pages to be submitted to the GPU
        LinearAllocatorPage*                    m_unusedPages;  // Pages not being used right now
        size_t                                  m_increment;
        size_t                                  m_numPending;
        size_t                                  m_totalPages;
        
        LinearAllocatorPage* GetPageForAlloc(size_t sizeBytes, size_t alignment);
        LinearAllocatorPage* GetCleanPageForAlloc();

        LinearAllocatorPage* FindPageForAlloc(LinearAllocatorPage* list, size_t sizeBytes, size_t alignment);

        LinearAllocatorPage* GetNewPage();

        void UnlinkPage(LinearAllocatorPage* page);
        void LinkPage(LinearAllocatorPage* page, LinearAllocatorPage*& list);
        void LinkPageChain(LinearAllocatorPage* page, LinearAllocatorPage*& list);
        void ReleasePage(LinearAllocatorPage* page);
        void FreePages(LinearAllocatorPage* list);

#if defined(_DEBUG) || defined(PROFILE)
        std::wstring m_debugName;

        static void ValidateList(LinearAllocatorPage* list);
        void ValidatePageLists();

        void SetPageDebugName(LinearAllocatorPage* list);
#endif
    };
}
