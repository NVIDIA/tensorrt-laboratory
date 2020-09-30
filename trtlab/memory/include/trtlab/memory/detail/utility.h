// MODIFICATION_MESSAGE

// Modification notes:
// - removed custom move/forward/swap implementations for std definitions

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_DETAIL_UTILITY_HPP
#define TRTLAB_MEMORY_DETAIL_UTILITY_HPP

// implementation of some functions from <utility> to prevent dependencies on it

#include <type_traits>
#include <utility>

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            using std::move;
            using std::forward;

            // ADL aware swap
            template <typename T>
            void adl_swap(T& a, T& b) noexcept
            {
                std::swap(a, b);
            }

// fancier syntax for enable_if
// used as (template) parameter
// also useful for doxygen
// define PREDEFINED: TRTLAB_REQUIRES(x):=
#define TRTLAB_REQUIRES(Expr) typename std::enable_if<(Expr), int>::type = 0

// same as above, but as return type
// also useful for doxygen:
// defined PREDEFINED: TRTLAB_REQUIRES_RET(x,r):=r
#define TRTLAB_REQUIRES_RET(Expr, ...) typename std::enable_if<(Expr), __VA_ARGS__>::type

// fancier syntax for enable_if on non-templated member function
#define TRTLAB_ENABLE_IF(Expr)                                                                     \
    template <typename Dummy = std::true_type, TRTLAB_REQUIRES(Dummy::value && (Expr))>

// fancier syntax for general expression SFINAE
// used as (template) parameter
// also useful for doxygen:
// define PREDEFINED: TRTLAB_SFINAE(x):=
#define TRTLAB_SFINAE(Expr) decltype((Expr), int()) = 0

// avoids code repetition for one-line forwarding functions
#define TRTLAB_AUTO_RETURN(Expr)                                                                   \
    decltype(Expr)                                                                                 \
    {                                                                                              \
        return Expr;                                                                               \
    }

// same as above, but requires certain type
#define TRTLAB_AUTO_RETURN_TYPE(Expr, T)                                                           \
    decltype(Expr)                                                                                 \
    {                                                                                              \
        static_assert(std::is_same<decltype(Expr), T>::value,                                      \
                      #Expr " does not have the return type " #T);                                 \
        return Expr;                                                                               \
    }

            // whether or not a type is an instantiation of a template
            template <template <typename...> class Template, typename T>
            struct is_instantiation_of : std::false_type
            {
            };

            template <template <typename...> class Template, typename... Args>
            struct is_instantiation_of<Template, Template<Args...>> : std::true_type
            {
            };
        } // namespace detail
    }
} // namespace trtlab::memory

#endif //TRTLAB_MEMORY_DETAIL_UTILITY_HPP
