// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_insert.hpp"

namespace ov {
namespace frontend {

SequenceInsert::SequenceInsert(const Output<Node>& input_sequence, const Output<Node>& tensor)
    : FrameworkNode({input_sequence, tensor}, 1) {}

SequenceInsert::SequenceInsert(const Output<Node>& input_sequence,
                               const Output<Node>& tensor,
                               const Output<Node>& position)
    : FrameworkNode({input_sequence, tensor, position}, 1) {}

void SequenceInsert::validate_and_infer_types() {
    // Deferred sequence placeholder resolved later by a transformation. Skip the
    // base FrameworkNode descriptor check, which fails when the op lives inside a
    // Loop body that re-validates with changed input shapes (e.g. [?] -> []).
    set_output_type(0, element::dynamic, PartialShape::dynamic());
}

std::shared_ptr<Node> SequenceInsert::clone_with_new_inputs(const OutputVector& inputs) const {
    if (inputs.size() == 2) {
        return std::make_shared<SequenceInsert>(inputs[0], inputs[1]);
    } else if (inputs.size() == 3) {
        return std::make_shared<SequenceInsert>(inputs[0], inputs[1], inputs[2]);
    }
    OPENVINO_THROW("SequenceInsert requires 2 or 3 inputs");
}

}  // namespace frontend
}  // namespace ov
