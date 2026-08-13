////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

// __host__ and __device__ as no-ops for the host compiler, so a kernel body can be written once
// and compiled by both.
//
// This is deliberately its own header rather than part of arch/arch.h: arch.h includes nda, and a
// kernel body included from a .cu must not pull nda through nvcc.

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
