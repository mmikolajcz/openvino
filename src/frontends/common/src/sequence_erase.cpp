// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_erase.hpp"

namespace ov {
namespace frontend {

SequenceErase::SequenceErase(const Output<Node>& input_sequence) : FrameworkNode({input_sequence}, 1) {}

SequenceErase::SequenceErase(const Output<Node>& input_sequence, const Output<Node>& position)
    : FrameworkNode({input_sequence, position}, 1) {}

void SequenceErase::validate_and_infer_types() {
    // Deferred sequence placeholder resolved later by a transformation. Skip the
    // base FrameworkNode descriptor check, which fails when the op lives inside a
    // Loop body that re-validates with changed input shapes (e.g. [?] -> []).
    set_output_type(0, element::dynamic, PartialShape::dynamic());
}

std::shared_ptr<Node> SequenceErase::clone_with_new_inputs(const OutputVector& inputs) const {
    if (inputs.size() == 1) {
        return std::make_shared<SequenceErase>(inputs[0]);
    } else if (inputs.size() == 2) {
        return std::make_shared<SequenceErase>(inputs[0], inputs[1]);
    }
    OPENVINO_THROW("SequenceErase requires 1 or 2 inputs");
}

}  // namespace frontend
}  // namespace ov
