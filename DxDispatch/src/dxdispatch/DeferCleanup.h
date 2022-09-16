//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

//
// The DeferCleanup function allows a lambda to be used to create a simple one-off RAII class.
// For (a contrived) example:
//      IUnknown* myObj;
//      myObj->AddRef();
//      
//      auto deferredCleanup = DeferCleanup([&]()
//      {
//         myObj->Release();
//      });
//
// The cleanup function will be run as soon as the "deferredCleanup" object goes out of scope.
// 
// This class is marked [[nodiscard]] to prevent callers from inadvertently throwing away the return value:
//      auto x = DeferCleanup([&]() { ... }); // okay
//      DeferCleanup([&]() { ... }); // warning: discarding return value of function
//
template <typename FunctorType>
struct [[nodiscard]] DeferCleanupType
{
public:
    DeferCleanupType() = default;
    explicit DeferCleanupType(FunctorType f)
        : m_f(f)
    {}

    DeferCleanupType(DeferCleanupType&& other) noexcept
        : m_f(std::move(other.m_f))
    {
        other.Detach();
    }

    DeferCleanupType& operator=(DeferCleanupType&& other) noexcept
    {
        if (this != &other)
        {
            Close();
            m_f = std::move(other.m_f);
            other.Detach();
        }
        return *this;
    }

    // noncopyable
    DeferCleanupType(const DeferCleanupType&) = delete;
    DeferCleanupType& operator=(const DeferCleanupType&) = delete;

    ~DeferCleanupType() noexcept
    {
        Close();
    }

    // If not already in a detached state, executes the cleanup functor and detaches it from this object.
    void Close() noexcept
    {
        if (m_f)
        {
            (*m_f)();
        }
        m_f.reset();
    }

    // Clears the state of this object without running the cleanup functor.
    void Detach() noexcept
    {
        m_f.reset();
    }

    // Returns true if this object has a value, false if not (e.g. if default constructed, or previously
    // closed/detached)
    explicit operator bool() const noexcept
    {
        return m_f.has_value();
    }

private:
    std::optional<FunctorType> m_f;
};

template <typename FunctorType>
DeferCleanupType<FunctorType> DeferCleanup(FunctorType f) { return DeferCleanupType<FunctorType>(f); }
