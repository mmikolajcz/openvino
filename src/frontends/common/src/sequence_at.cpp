// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_at.hpp"

namespace ov {
namespace frontend {

SequenceAt::SequenceAt(const Output<Node>& input_sequence, const Output<Node>& position)
    : FrameworkNode({input_sequence, position}, 1) {}

void SequenceAt::validate_and_infer_types() {
    // Deferred sequence placeholder resolved later by a transformation. Skip the
    // base FrameworkNode descriptor check, which fails when the op lives inside a
    // Loop body that re-validates with changed input shapes (e.g. [?] -> []).
    set_output_type(0, element::dynamic, PartialShape::dynamic());
}

std::shared_ptr<Node> SequenceAt::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 2, "SequenceAt requires 2 inputs");
    return std::make_shared<SequenceAt>(inputs[0], inputs[1]);
}

}  // namespace frontend
}  // namespace ov
