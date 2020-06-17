//--------------------------------------------------------------------------------------
// File: LinearAllocator.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "DirectXHelpers.h"
#include "PlatformHelpers.h"
#include "LinearAllocator.h"

// Set this to 1 to enable some additional debug validation
#define VALIDATE_LISTS 0

#if VALIDATE_LISTS
#   include <unordered_set>
#endif

using namespace DirectX;
using Microsoft::WRL::ComPtr;

LinearAllocatorPage::LinearAllocatorPage() noexcept
    : pPrevPage(nullptr)
    , pNextPage(nullptr)
    , mMemory(nullptr)
    , mPendingFence(0)
    , mGpuAddress{}
    , mOffset(0)
    , mSize(0)
    , mRefCount(1)
{
}

size_t LinearAllocatorPage::Suballocate(_In_ size_t size, _In_ size_t alignment)
{
    size_t offset = AlignUp(mOffset, alignment);
    if (offset + size > mSize)
    {
        // Use of suballocate should be limited to pages with free space,
        // so really shouldn't happen.
        throw std::exception("LinearAllocatorPage::Suballocate");
    }
    mOffset = offset + size;
    return offset;
}

void LinearAllocatorPage::Release()
{
    assert(mRefCount > 0); 

    if (mRefCount.fetch_sub(1) == 1)
    {
        mUploadResource->Unmap(0, nullptr);
        delete this;
    }
}


//--------------------------------------------------------------------------------------
LinearAllocator::LinearAllocator(
    _In_ ID3D12Device* pDevice,
    _In_ size_t pageSize,
    _In_ size_t preallocateBytes)
    : m_device(pDevice)
    , m_pendingPages(nullptr)
    , m_usedPages(nullptr)
    , m_unusedPages(nullptr)
    , m_increment(pageSize)
    , m_numPending(0)
    , m_totalPages(0)
{
#if defined(_DEBUG) || defined(PROFILE)
    m_debugName = L"LinearAllocator";
#endif

    size_t preallocatePageCount = ((preallocateBytes + pageSize - 1) / pageSize);
    for (size_t preallocatePages = 0; preallocateBytes != 0 && preallocatePages < preallocatePageCount; ++preallocatePages)
    {
        if (GetNewPage() == nullptr)
        {
            DebugTrace("LinearAllocator failed to preallocate pages (%zu required bytes, %zu pages)\n",
                preallocatePageCount * m_increment, preallocatePageCount);
            throw std::bad_alloc();
        }
    }
}

LinearAllocator::~LinearAllocator()
{
    // Must wait for all pending fences!
    while (m_pendingPages != nullptr)
    {
        RetirePendingPages();
    }

    assert(m_pendingPages == nullptr);

    // Return all the memory
    FreePages(m_unusedPages);
    FreePages(m_usedPages);

    m_pendingPages = nullptr;
    m_usedPages = nullptr;
    m_unusedPages = nullptr;
    m_increment = 0;
}

LinearAllocatorPage* LinearAllocator::FindPageForAlloc(_In_ size_t size, _In_ size_t alignment)
{
#ifdef _DEBUG
    if (size > m_increment)
        throw std::out_of_range("Size must be less or equal to the allocator's increment");
    if (alignment > m_increment)
        throw std::out_of_range("Alignment must be less or equal to the allocator's increment");
    if (size == 0)
        throw std::exception("Cannot honor zero size allocation request.");
#endif

    auto page = GetPageForAlloc(size, alignment);
    if (!page)
    {
        return nullptr;
    }

    return page;
}

// Call this after you submit your work to the driver.
void LinearAllocator::FenceCommittedPages(_In_ ID3D12CommandQueue* commandQueue)
{
    // No pending pages
    if (m_usedPages == nullptr)
        return;

    // For all the used pages, fence them
    UINT numReady = 0;
    LinearAllocatorPage* readyPages = nullptr;
    LinearAllocatorPage* unreadyPages = nullptr;
    LinearAllocatorPage* nextPage = nullptr;
    for (auto page = m_usedPages; page != nullptr; page = nextPage)
    {
        nextPage = page->pNextPage;

        // Disconnect from the list
        page->pPrevPage = nullptr;

        // This implies the allocator is the only remaining reference to the page, and therefore the memory is ready for re-use.
        if (page->RefCount() == 1)
        {
            // Signal the fence
            numReady++;
            ThrowIfFailed(commandQueue->Signal(page->mFence.Get(), ++page->mPendingFence));

            // Link to the ready pages list
            page->pNextPage = readyPages;
            if (readyPages) readyPages->pPrevPage = page;
            readyPages = page;
        }
        else
        {
            // Link to the unready list
            page->pNextPage = unreadyPages;
            if (unreadyPages) unreadyPages->pPrevPage = page;
            unreadyPages = page;
        }
    }

    // Replace the used pages list with the new unready list
    m_usedPages = unreadyPages;

    // Append all those pages from the ready list to the pending list
    if (numReady > 0)
    {
        m_numPending += numReady;
        LinkPageChain(readyPages, m_pendingPages);
    }

#if VALIDATE_LISTS
    ValidatePageLists();
#endif
}

// Call this once a frame after all of your driver submissions.
// (immediately before or after Present-time)
void LinearAllocator::RetirePendingPages()
{
    // For each page that we know has a fence pending, check it. If the fence has passed,
    // we can mark the page for re-use.
    auto page = m_pendingPages;
    while (page != nullptr)
    {
        LinearAllocatorPage* nextPage = page->pNextPage;

        assert(page->mPendingFence != 0);

        if (page->mFence->GetCompletedValue() >= page->mPendingFence)
        {
            // Fence has passed. It is safe to use this page again.
            ReleasePage(page);
        }

        page = nextPage;
    }
}

void LinearAllocator::Shrink()
{
    FreePages(m_unusedPages);
    m_unusedPages = nullptr;

#if VALIDATE_LISTS
    ValidatePageLists();
#endif
}

LinearAllocatorPage* LinearAllocator::GetCleanPageForAlloc()
{
    // Grab the first unused page, if one exists. Else, allocate a new page.
    auto page = m_unusedPages;
    if (!page)
    {
        // Allocate a new page
        page = GetNewPage();
        if (!page)
        {
            return nullptr;
        }
    }

    // Mark this page as used
    UnlinkPage(page);
    LinkPage(page, m_usedPages);

    assert(page->mOffset == 0);

    return page;
}

LinearAllocatorPage* LinearAllocator::GetPageForAlloc(
    size_t sizeBytes,
    size_t alignment)
{
    // Fast path
    if (sizeBytes == m_increment && (alignment == 0 || alignment == m_increment))
    {
        return GetCleanPageForAlloc();
    }

    // Find a page in the pending pages list that has space.
    auto page = FindPageForAlloc(m_usedPages, sizeBytes, alignment);
    if (!page)
    {
        page = GetCleanPageForAlloc();
    }

    return page;
}

LinearAllocatorPage* LinearAllocator::FindPageForAlloc(
    LinearAllocatorPage* list,
    size_t sizeBytes,
    size_t alignment)
{
    for (auto page = list; page != nullptr; page = page->pNextPage)
    {
        size_t offset = AlignUp(page->mOffset, alignment);
        if (offset + sizeBytes <= m_increment)
            return page;
    }
    return nullptr;
}

LinearAllocatorPage* LinearAllocator::GetNewPage()
{
    CD3DX12_HEAP_PROPERTIES uploadHeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(m_increment);

    // Allocate the upload heap
    ComPtr<ID3D12Resource> spResource;
    HRESULT hr = m_device->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(spResource.ReleaseAndGetAddressOf()));
    if (FAILED(hr))
    {
        if (hr != E_OUTOFMEMORY)
        {
            DebugTrace("LinearAllocator::GetNewPage resource allocation failed due to unexpected error %08X\n", hr);
        }
        return nullptr;
    }

#if defined(_DEBUG) || defined(PROFILE)
    spResource->SetName(m_debugName.empty() ? L"LinearAllocator" : m_debugName.c_str());
#endif

    // Get a pointer to the memory
    void* pMemory = nullptr;
    ThrowIfFailed(spResource->Map(0, nullptr, &pMemory));
    memset(pMemory, 0, m_increment);

    // Create a fence
    ComPtr<ID3D12Fence> spFence;
    hr = m_device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_GRAPHICS_PPV_ARGS(spFence.ReleaseAndGetAddressOf()));
    if (FAILED(hr))
    {
        DebugTrace("LinearAllocator::GetNewPage failed to allocate fence with error %08X\n", hr);
        return nullptr;
    }

    SetDebugObjectName(spFence.Get(), L"LinearAllocator");

    // Add the page to the page list
    auto page = new LinearAllocatorPage;
    page->mSize = m_increment;
    page->mMemory = pMemory;
    page->pPrevPage = nullptr;
    page->pNextPage = m_unusedPages;
    page->mGpuAddress = spResource->GetGPUVirtualAddress();
    page->mUploadResource.Swap(spResource);
    page->mFence.Swap(spFence);

    // Set as head of the list
    page->pNextPage = m_unusedPages;
    if (m_unusedPages) m_unusedPages->pPrevPage = page;
    m_unusedPages = page;
    m_totalPages++;

#if VALIDATE_LISTS
    ValidatePageLists();
#endif

    return page;
}

void LinearAllocator::UnlinkPage(LinearAllocatorPage* page)
{
    if (page->pPrevPage)
        page->pPrevPage->pNextPage = page->pNextPage;

    // Check that it isn't the head of any of our tracked lists
    else if (page == m_unusedPages)
        m_unusedPages = page->pNextPage;
    else if (page == m_usedPages)
        m_usedPages = page->pNextPage;
    else if (page == m_pendingPages)
        m_pendingPages = page->pNextPage;

    if (page->pNextPage)
        page->pNextPage->pPrevPage = page->pPrevPage;

    page->pNextPage = nullptr;
    page->pPrevPage = nullptr;

#if VALIDATE_LISTS
    ValidatePageLists();
#endif
}

void LinearAllocator::LinkPageChain(LinearAllocatorPage* page, LinearAllocatorPage*& list)
{
#if VALIDATE_LISTS
    // Walk the chain and ensure it's not in the list twice
    for (LinearAllocatorPage* cur = list; cur != nullptr; cur = cur->pNextPage)
    {
        assert(cur != page);
    }
#endif
    assert(page->pPrevPage == nullptr);
    assert(list == nullptr || list->pPrevPage == nullptr);

    // Follow chain to the end and append
    LinearAllocatorPage* lastPage = nullptr;
    for (lastPage = page; lastPage->pNextPage != nullptr; lastPage = lastPage->pNextPage) {}

    lastPage->pNextPage = list;
    if (list)
        list->pPrevPage = lastPage;

    list = page;

#if VALIDATE_LISTS
    ValidatePageLists();
#endif
}

void LinearAllocator::LinkPage(LinearAllocatorPage* page, LinearAllocatorPage*& list)
{
#if VALIDATE_LISTS
    // Walk the chain and ensure it's not in the list twice
    for (LinearAllocatorPage* cur = list; cur != nullptr; cur = cur->pNextPage)
    {
        assert(cur != page);
    }
#endif
    assert(page->pNextPage == nullptr);
    assert(page->pPrevPage == nullptr);
    assert(list == nullptr || list->pPrevPage == nullptr);

    page->pNextPage = list;
    if (list)
        list->pPrevPage = page;

    list = page;

#if VALIDATE_LISTS
    ValidatePageLists();
#endif
}

void LinearAllocator::ReleasePage(LinearAllocatorPage* page)
{
    assert(m_numPending > 0);
    m_numPending--;

    UnlinkPage(page);
    LinkPage(page, m_unusedPages);

    // Reset the page offset (effectively erasing the memory)
    page->mOffset = 0;

#ifdef _DEBUG
    memset(page->mMemory, 0, m_increment);
#endif

#if VALIDATE_LISTS
    ValidatePageLists();
#endif
}

void LinearAllocator::FreePages(LinearAllocatorPage* page)
{
    while (page != nullptr)
    {
        LinearAllocatorPage* nextPage = page->pNextPage;

        page->Release();

        page = nextPage;
        assert(m_totalPages > 0);
        m_totalPages--;
    }
}

#if VALIDATE_LISTS
void LinearAllocator::ValidateList(LinearAllocatorPage* list)
{
    for (auto page = list, *lastPage = nullptr;
        page != nullptr;
        lastPage = page, page = page->pNextPage)
    {
        if (page->pPrevPage != lastPage)
        {
            throw std::exception("Broken link to previous");
        }
    }
}

void LinearAllocator::ValidatePageLists()
{
    ValidateList(m_pendingPages);
    ValidateList(m_usedPages);
    ValidateList(m_unusedPages);
}
#endif

#if defined(_DEBUG) || defined(PROFILE)
void LinearAllocator::SetDebugName(const char* name)
{
    wchar_t wname[MAX_PATH] = {};
    int result = MultiByteToWideChar(CP_UTF8, 0, name, static_cast<int>(strlen(name)), wname, MAX_PATH);
    if (result > 0)
    {
        SetDebugName(wname);
    }
}

void LinearAllocator::SetDebugName(const wchar_t* name)
{
    m_debugName = name;

    // Rename existing pages
    SetPageDebugName(m_pendingPages);
    SetPageDebugName(m_usedPages);
    SetPageDebugName(m_unusedPages);
}

void LinearAllocator::SetPageDebugName(LinearAllocatorPage* list)
{
    for (auto page = list; page != nullptr; page = page->pNextPage)
    {
        page->mUploadResource->SetName(m_debugName.c_str());
    }
}
#endif

