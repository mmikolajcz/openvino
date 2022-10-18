// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "ngraph/runtime/reference/elementwise_functor_call.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename U>
void is_inf(const T* input, U* output, size_t count, const ov::op::v10::IsInf::Attributes& attributes) {
    if (attributes.detect_negative && attributes.detect_positive) {
        runtime::reference::elementwise_functor_call(input, output, count, [](T x) -> U {
            return (static_cast<U>(std::isinf(x)));
        });
    } else if (!attributes.detect_negative && attributes.detect_positive) {
        runtime::reference::elementwise_functor_call(input, output, count, [](T x) -> U {
            return (static_cast<U>(x == std::numeric_limits<T>::infinity()));
        });
    } else if (attributes.detect_negative && !attributes.detect_positive) {
        runtime::reference::elementwise_functor_call(input, output, count, [](T x) -> U {
            return (static_cast<U>(x == -std::numeric_limits<T>::infinity()));
        });
    } else {
        std::memset(output, 0, count);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
