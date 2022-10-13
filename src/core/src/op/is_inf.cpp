// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_inf.hpp"
#include "ngraph/runtime/reference/is_inf.hpp"

#include "itt.hpp"

namespace ov {
op::v10::IsInf::IsInf(const Output<Node>& data, const Attributes& attributes)
    : op::Op{{data}},
      m_attributes{attributes} {
    constructor_validate_and_infer_types();
}

bool op::v10::IsInf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_IsInf_visit_attributes);
    visitor.on_attribute("detect_negative", m_attributes.detect_negative);
    visitor.on_attribute("detect_positive", m_attributes.detect_positive);
    return true;
}

void op::v10::IsInf::validate_and_infer_types() {
    OV_OP_SCOPE(v10_IsInf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_real(),
                          "The element type of the input tensor must be a floating point number.");
    set_output_type(0, element::boolean, get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v10::IsInf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_IsInf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v10::IsInf>(new_args.at(0), this->get_attributes());
}

namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& input,
                   const HostTensorPtr& output,
                   const op::v10::IsInf::Attributes& attributes) {
    ov::runtime::reference::is_inf(input->get_data_ptr<ET>(),
                                   output->get_data_ptr<element::Type_t::boolean>(),
                                   input->get_shape());
}

bool evaluate_exec(const HostTensorPtr& input,
                   const HostTensorPtr& output,
                   const op::v10::IsInf::Attributes& attributes) {
    bool rc = true;
    switch (input->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_equal, bf16, input, output, attributes);
        NGRAPH_TYPE_CASE(evaluate_equal, f16, input, output, attributes);
        NGRAPH_TYPE_CASE(evaluate_equal, f32, input, output, attributes);
        NGRAPH_TYPE_CASE(evaluate_equal, f64, input, output, attributes);
    default:
        rc = false;
        break;
    }
    return rc;
    }
}
}  // namespace ov
