// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/model.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_cummax(const NodeContext& context) {
    // aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
    // aten.cummax.default(Tensor self, int dim) -> (Tensor values, Tensor indices)
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto dim = context.const_input<int64_t>(1);

    auto dim_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));
    auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));

    Output<Node> shape, rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true, element::i64);
    auto axis = normalize_axis(context, dim_const, rank);
    auto axis_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(axis, zero));
    auto scan_len = context.mark_node(std::make_shared<v8::Gather>(shape, axis, zero));
    auto positions = context.mark_node(std::make_shared<v4::Range>(zero, scan_len, one, element::i64));

    // Initial indices: position along `dim`, broadcast to the input shape.
    auto rank_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(rank, zero));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one, rank_1d));
    auto scan_len_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(scan_len, zero));
    auto pos_shape = context.mark_node(std::make_shared<v3::ScatterUpdate>(ones, axis_1d, scan_len_1d, zero));
    auto positions_nd = context.mark_node(std::make_shared<v1::Reshape>(positions, pos_shape, false));
    auto init_idx = context.mark_node(std::make_shared<v3::Broadcast>(positions_nd, shape));

    // Hillis-Steele inclusive scan: the iteration with shift k combines each element with the one k positions
    // earlier, so ceil(log2(S)) full-tensor iterations replace S sequential ones. The first k elements are
    // combined with themselves, which is a no-op for any dtype.
    auto body_val = std::make_shared<v0::Parameter>(x.get_element_type(), PartialShape::dynamic());
    auto body_idx = std::make_shared<v0::Parameter>(element::i64, PartialShape::dynamic());
    auto body_k = std::make_shared<v0::Parameter>(element::i64, Shape{});
    auto body_positions = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1});
    auto body_len = std::make_shared<v0::Parameter>(element::i64, Shape{});

    auto has_prev = std::make_shared<v1::GreaterEqual>(body_positions, body_k);
    auto prev_positions =
        std::make_shared<v1::Select>(has_prev, std::make_shared<v1::Subtract>(body_positions, body_k), body_positions);
    auto body_axis = v0::Constant::create(element::i64, Shape{}, {dim});
    auto prev_val = std::make_shared<v8::Gather>(body_val, prev_positions, body_axis);
    auto prev_idx = std::make_shared<v8::Gather>(body_idx, prev_positions, body_axis);

    // torch.cummax orders NaN above everything and takes the latest index on ties (NaN ties included).
    Output<Node> take_cur = std::make_shared<v1::GreaterEqual>(body_val, prev_val);
    const auto& elem_type = x.get_element_type();
    if (elem_type.is_dynamic() || elem_type.is_real()) {
        // x != x is the NaN test; unlike IsNaN it also accepts integer types (element type may be unknown here).
        take_cur = std::make_shared<v1::LogicalOr>(take_cur, std::make_shared<v1::NotEqual>(body_val, body_val));
    }
    auto new_val = std::make_shared<v1::Select>(take_cur, body_val, prev_val);
    auto new_idx = std::make_shared<v1::Select>(take_cur, body_idx, prev_idx);

    auto next_k = std::make_shared<v1::Multiply>(body_k, v0::Constant::create(element::i64, Shape{}, {2}));
    auto body_cond = std::make_shared<v1::Less>(next_k, body_len);

    auto body = std::make_shared<Model>(OutputVector{body_cond, new_val, new_idx, next_k},
                                         ParameterVector{body_val, body_idx, body_k, body_positions, body_len});

    // 64 iterations cover any int64 length; the body condition stops the loop after ceil(log2(S)) of them.
    auto trip_count = context.mark_node(v0::Constant::create(element::i64, Shape{}, {64}));
    auto exec_cond = context.mark_node(v0::Constant::create(element::boolean, Shape{}, {true}));
    auto loop = std::make_shared<v5::Loop>(trip_count, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(v5::Loop::SpecialBodyPorts{-1, 0});
    loop->set_merged_input(body_val, x, new_val);
    loop->set_merged_input(body_idx, init_idx, new_idx);
    loop->set_merged_input(body_k, one, next_k);
    loop->set_invariant_input(body_positions, positions);
    loop->set_invariant_input(body_len, scan_len);
    auto values = loop->get_iter_value(new_val, -1);
    auto indices = loop->get_iter_value(new_idx, -1);
    context.mark_node(loop);

    return {values, indices};
}

}  // namespace ov::frontend::pytorch::op
