// MODIFICATION_MESSAGE

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

/// \file
/// Configuration macros.

#ifndef TRTLAB_MEMORY_CONFIG_H_INCLUDED
#define TRTLAB_MEMORY_CONFIG_H_INCLUDED

#if !defined(DOXYGEN)
#define TRTLAB_MEMORY_IMPL_IN_CONFIG_H
#include "config_impl.h"
#undef TRTLAB_MEMORY_IMPL_IN_CONFIG_H
#endif

// general compatibility headers
// #include <foonathan/constexpr.hpp>
// #include <foonathan/noexcept.hpp>
// #include <foonathan/exception_support.hpp>
// #include <foonathan/hosted_implementation.hpp>

// log prefix
#ifndef TRTLAB_MEMORY_LOG_PREFIX
#define TRTLAB_MEMORY_LOG_PREFIX "trtlab::memory"
#endif

// version
#define TRTLAB_MEMORY_VERSION                                                                   \
    (TRTLAB_MEMORY_VERSION_MAJOR * 100 + TRTLAB_MEMORY_VERSION_MINOR)

// use this macro to mark implementation-defined types
// gives it more semantics and useful with doxygen
// add PREDEFINED: TRTLAB_IMPL_DEFINED():=implementation_defined
#ifndef TRTLAB_IMPL_DEFINED
#define TRTLAB_IMPL_DEFINED(...) __VA_ARGS__
#endif

// use this macro to mark base class which only purpose is EBO
// gives it more semantics and useful with doxygen
// add PREDEFINED: TRTLAB_EBO():=
#ifndef TRTLAB_EBO
#define TRTLAB_EBO(...) __VA_ARGS__
#endif

#ifndef TRTLAB_ALIAS_TEMPLATE
// defines a template alias
// usage:
// template <typename T>
// TRTLAB_ALIAS_TEMPLATE(bar, foo<T, int>);
// useful for doxygen
#ifdef DOXYGEN
#define TRTLAB_ALIAS_TEMPLATE(Name, ...)                                                        \
    class Name : public __VA_ARGS__                                                                \
    {                                                                                              \
    }
#else
#define TRTLAB_ALIAS_TEMPLATE(Name, ...) using Name = __VA_ARGS__
#endif
#endif

#ifdef DOXYGEN
// dummy definitions of config macros for doxygen

/// The major version number.
/// \ingroup memory core
#define TRTLAB_MEMORY_VERSION_MAJOR 1

/// The minor version number.
/// \ingroup memory core
#define TRTLAB_MEMORY_VERSION_MINOR 1

/// The total version number of the form \c Mmm.
/// \ingroup memory core
#define TRTLAB_MEMORY_VERSION                                                                   \
    (TRTLAB_MEMORY_VERSION_MAJOR * 100 + TRTLAB_MEMORY_VERSION_MINOR)

/// Whether or not the allocation size will be checked,
/// i.e. the \ref bad_allocation_size thrown.
/// \ingroup memory core
#define TRTLAB_MEMORY_CHECK_ALLOCATION_SIZE 1

/// Whether or not the \ref foonathan::memory::default_mutex will be \c std::mutex or \ref foonathan::memory::no_mutex.
/// \ingroup memory core
#define TRTLAB_MEMORY_THREAD_SAFE_REFERENCE 1

/// Whether or not internal assertions in the library are enabled.
/// \ingroup memory core
#define TRTLAB_MEMORY_DEBUG_ASSERT 1

/// Whether or not allocated memory will be filled with special values.
/// \ingroup memory core
#define TRTLAB_MEMORY_DEBUG_FILL 1

/// The size of the fence memory, it has no effect if \ref TRTLAB_MEMORY_DEBUG_FILL is \c false.
/// \note For most allocators, the actual value doesn't matter and they use appropriate defaults to ensure alignment etc.
/// \ingroup memory core
#define TRTLAB_MEMORY_DEBUG_FENCE 1

/// Whether or not leak checking is enabled.
/// \ingroup memory core
#define TRTLAB_MEMORY_DEBUG_LEAK_CHECK 1

/// Whether or not the deallocation functions will check for pointers that were never allocated by an allocator.
/// \ingroup memory core
#define TRTLAB_MEMORY_DEBUG_POINTER_CHECK 1

/// Whether or not the deallocation functions will check for double free errors.
/// This option makes no sense if \ref TRTLAB_MEMORY_DEBUG_POINTER_CHECK is \c false.
/// \ingroup memory core
#define TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECK 1

/// Whether or not everything is in namespace <tt>foonathan::memory</tt>.
/// If \c false, a namespace alias <tt>namespace memory = foonathan::memory</tt> is automatically inserted into each header,
/// allowing to qualify everything with <tt>foonathan::</tt>.
/// \note This option breaks in combination with using <tt>using namespace foonathan;</tt>.
/// \ingroup memory core
#define TRTLAB_MEMORY_NAMESPACE_PREFIX 1

/// The mode of the automatic \ref temporary_stack creation.
/// Set to `2` to enable automatic lifetime management of the per-thread stack through nifty counter.
/// Then all memory will be freed upon program termination automatically.
/// Set to `1` to disable automatic lifetime managment of the per-thread stack,
/// requires managing it through the \ref temporary_stack_initializer.
/// Set to `0` to disable the per-thread stack completely.
/// \ref get_temporary_stack() will abort the program upon call.
/// \ingroup memory allocator
#define TRTLAB_MEMORY_TEMPORARY_STACK_MODE 2
#endif

#endif // TRTLAB_MEMORY_CONFIG_H_INCLUDED
