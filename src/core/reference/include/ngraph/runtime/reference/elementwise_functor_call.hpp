// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename U, typename Functor>
void elementwise_functor_call(const T* input, U* output, size_t count, Functor functor) {
    for (size_t i = 0; i < count; i++) {
        output[i] = static_cast<U>(functor(input[i]));
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
