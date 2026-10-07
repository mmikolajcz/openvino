# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCumMax(PytorchLayerTest):
    def _prepare_input(self, input_dtype="float32", fixed_values=None):
        if fixed_values is not None:
            return (np.array(fixed_values, dtype=input_dtype),)
        return (self.random.randn(2, 3, 4).astype(input_dtype),)

    def create_model(self, dim):
        import torch

        class aten_cummax(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                values, indices = torch.cummax(x, self.dim)
                return values, indices

        return aten_cummax(dim), "aten::cummax"

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32", "int64"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_cummax(self, dim, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version,
                    kwargs_to_prepare_input={"input_dtype": input_dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cummax_ties(self, ie_device, precision, ir_version):
        # Repeated values along the scan dim: torch.cummax returns the latest index on ties.
        self._test(*self.create_model(1), ie_device, precision, ir_version,
                    kwargs_to_prepare_input={"input_dtype": "int32", "fixed_values": [[1, 3, 3, 2, 3, 0]]})

    @pytest.mark.parametrize("length", [1, 2, 3, 5, 8, 37])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cummax_lengths(self, length, ie_device, precision, ir_version):
        values = np.random.default_rng(length).integers(-5, 5, (2, length))
        self._test(*self.create_model(1), ie_device, precision, ir_version,
                    kwargs_to_prepare_input={"input_dtype": "int64", "fixed_values": values})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cummax_nan(self, ie_device, precision, ir_version):
        nan = float("nan")
        self._test(*self.create_model(1), ie_device, precision, ir_version,
                    kwargs_to_prepare_input={"input_dtype": "float32",
                                             "fixed_values": [[1.0, nan, 2.0, 3.0, nan], [2.0, 2.0, nan, nan, 5.0]]})
