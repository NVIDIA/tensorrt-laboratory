// MODIFICATION_MESSAGE

// Modification notes:
// - Replaced in-library logging feature with glog

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_DETAIL_ASSERT_H_INCLUDED
#define TRTLAB_MEMORY_DETAIL_ASSERT_H_INCLUDED

#include <cstdlib>

#include "../config.h"

#include <glog/logging.h>

#if defined(TRTLAB_MEMORY_ASSERT)
#define TRTLAB_MEMORY_ASSERT(Expr) CHECK(Expr)
#define TRTLAB_MEMORY_ASSERT_MSG(Expr, Msg) CHECK(Expr) << Msg
#define TRTLAB_MEMORY_UNREACHABLE(Msg) LOG(FATAL) << Msg
#define TRTLAB_MEMORY_WARNING(Msg) LOG(WARNING) << Msg
#else
#define TRTLAB_MEMORY_ASSERT(Expr)
#define TRTLAB_MEMORY_ASSERT_MSG(Expr, Msg)
#define TRTLAB_MEMORY_UNREACHABLE(Msg) std::abort()
#define TRTLAB_MEMORY_WARNING(Msg)
#endif

#endif // TRTLAB_MEMORY_DETAIL_ASSERT_H_INCLUDED
