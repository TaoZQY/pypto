# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for distributed Python code generation."""

import ast
import inspect
import json
import re
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.runtime import _distributed_support as support
from pypto.runtime.distributed_runner import _load_orch_entry


class TestDistributedCodegen:
    """Test distributed Python codegen on outlined hierarchy programs."""

    def test_chip_sub_worker_and_orchestrator(self):
        """HOST orchestrator calling CHIP orchestrator → CHIP worker dispatches via _submit_chip."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Verify imports
        assert "from simpler.task_interface import " in code
        assert "TaskArgs" in code and "TensorArgType" in code
        assert "from pypto.runtime.tensor_arg import make_tensor_arg" in code

        # Verify function definition
        assert "def host_orch" in code
        assert "orch, _args, config" in code

        # Verify call-site lowering: CHIP orchestrator → _submit_chip (the
        # comm-less dispatch passes None and _submit_chip resolves the chip;
        # it forwards to submit_next_level when DFX is off).
        assert "_submit_chip" in code
        assert 'callables["chip_orch"]' in code
        assert "TaskArgs()" in code

    def test_renamed_host_orch_marks_entry(self):
        """Host orchestrator under any name gets the runtime entry marker.

        Regression for issue #1678: the runtime resolves the dispatch entry by
        the ``_pypto_distributed_entry`` marker, not by function name, so a
        renamed ``@pl.jit.host`` orchestrator must carry the marker.
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def moe_ep_l3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x)
                return y

        program = passes.convert_to_ssa()(Input)
        code = codegen.DistributedCodegen().generate(program)

        assert "def moe_ep_l3(" in code
        assert "moe_ep_l3._pypto_distributed_entry = True" in code
        # The marker must follow the function definition it tags.
        assert code.index("def moe_ep_l3(") < code.index("moe_ep_l3._pypto_distributed_entry")

    def test_sub_worker_submit_sub(self):
        """HOST worker (SubWorker) produces submit_sub call."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.verify(x)
                return x

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # HOST worker (level 3) → submit_sub
        assert "submit_sub" in code
        assert 'sub_ids["verify"]' in code

    def test_chip_and_sub_worker_combined(self):
        """Program with both CHIP orchestrator (→ chip worker) and HOST SubWorker."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                f: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, b)
                self.verify(f)
                return f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "_submit_chip" in code
        assert "submit_sub" in code
        assert "TensorArgType.INPUT" in code

    def test_for_loop_codegen(self):
        """ForStmt in function body produces Python for loop."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def worker(x: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def orch_with_loop(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 4):
                    self.worker(x)
                return x

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "for " in code
        assert "in range(" in code
        assert "submit_sub" in code
        assert 'sub_ids["worker"]' in code

    def test_python_imports(self):
        """Generated code contains required Python imports."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def simple_worker(x: pl.Tensor[[64], pl.FP32]):
                pass

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "from simpler.task_interface import " in code
        assert "TaskArgs" in code and "TensorArgType" in code
        assert "from pypto.runtime.tensor_arg import make_tensor_arg" in code

    def test_tensor_arg_type_tags(self):
        """Parameter directions map to correct TensorArgType tags."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b, f)
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, b, f)
                return out

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "TensorArgType.INPUT" in code
        assert "TensorArgType.OUTPUT_EXISTING" in code

    def test_bool_constants(self):
        """Boolean constants use Python True/False, not C++ true/false."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def worker(x: pl.Tensor[[64], pl.FP32]):
                pass

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Python uses True/False, not true/false
        assert "true" not in code.lower() or "True" in code or "False" in code

    def test_sub_worker_pure_python_body(self):
        """HOST Worker with pure Python body is captured without DSL parsing."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[128, 128], pl.FP32]):
                import torch  # noqa: PLC0415

                expected = torch.full((128, 128), 5.0, dtype=torch.float32)
                assert torch.allclose(f, expected)  # pyright: ignore[reportArgumentType]

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                self.verify(x)
                return x

        # Should not raise — pure Python body is skipped during DSL parsing
        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "submit_sub" in code
        assert 'sub_ids["verify"]' in code

    def test_sub_worker_body_inlined_in_ir(self):
        """SubWorker body is captured as an InlineStmt on the IR Function."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.verify(x)
                return x

        verify_fn = Input.get_function("verify")
        assert verify_fn is not None
        assert isinstance(verify_fn.body, ir.InlineStmt)
        assert verify_fn.body.language == ir.InlineLanguage.Python
        assert isinstance(verify_fn.body.body, str)
        # A concrete (`pass`) body is NOT a runtime-bound callback.
        assert verify_fn.requires_runtime_binding is False

    def test_abstract_sub_worker_is_runtime_bound(self):
        """A SubWorker declared with a `...` body is an abstract callback."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        sample_fn = Input.get_function("sample")
        assert sample_fn is not None
        assert sample_fn.requires_runtime_binding is True
        # Abstract body carries no captured source text.
        assert isinstance(sample_fn.body, ir.InlineStmt)
        assert sample_fn.body.body == ""

    def test_abstract_sub_worker_round_trips_as_ellipsis(self):
        """An abstract SubWorker prints as `...` and reparses to the same flag."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        printed = str(Input)
        assert "def sample(" in printed
        assert "..." in printed

        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Input, reparsed)
        sample_fn = reparsed.get_function("sample")
        assert sample_fn is not None
        assert sample_fn.requires_runtime_binding is True

    def test_abstract_sub_worker_emits_guard_stub(self):
        """Codegen emits a raising guard for an abstract SubWorker module."""
        from pypto.backend.pto_backend import _emit_sub_worker_module  # noqa: PLC0415

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        sample_fn = Input.get_function("sample")
        assert sample_fn is not None
        module = _emit_sub_worker_module(sample_fn)
        assert "def sample(args):" in module
        assert "raise RuntimeError" in module
        assert "prepare(callbacks=" in module
        # Generated stub must be syntactically valid Python.
        compile(module, "<sample>", "exec")

    def test_abstract_sub_worker_survives_pto_serialization(self):
        """requires_runtime_binding round-trips through binary .pto serialization."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        restored = ir.deserialize(ir.serialize(Input))
        assert isinstance(restored, ir.Program)
        sample_fn = restored.get_function("sample")
        assert sample_fn is not None
        assert sample_fn.requires_runtime_binding is True
        ir.assert_structural_equal(Input, restored)

    def test_create_tensor_emits_shared_torch_zeros(self):
        """tensor.create in HOST orchestrator emits torch.zeros(...).share_memory_()."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, buf)
                return result

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                buf: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, buf)
                return result

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # torch.zeros with share_memory_() emitted
        assert "torch.zeros(" in code
        assert "torch.float32" in code
        assert ".share_memory_()" in code
        assert "import torch" in code

    def test_create_tensor_shared_zeros_for_multiple_tensors(self):
        """Multiple tensor.create calls each emit torch.zeros(...).share_memory_()."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_add(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_sub(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.sub(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch_add(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_add(a, b, f)
                return out

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch_sub(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_sub(a, b, f)
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def reduce_sum(
                sum_ab: pl.Tensor[[64], pl.FP32],
                diff_ab: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return f

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                sum_ab: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                diff_ab: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                out_sum: pl.Tensor[[64], pl.FP32] = self.chip_orch_add(a, b, sum_ab)
                out_diff: pl.Tensor[[64], pl.FP32] = self.chip_orch_sub(a, b, diff_ab)
                out_f: pl.Tensor[[64], pl.FP32] = self.reduce_sum(out_sum, out_diff, f)
                return out_f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Two torch.zeros().share_memory_() calls
        assert code.count("torch.zeros(") == 2
        assert code.count(".share_memory_()") == 2
        # Plain host tensors use the worker-aware address-free wire helper.
        assert re.search(r'add_chip_tensor_arg\([^,]+, orch\._worker, tensors\["a', code)
        assert re.search(r'add_chip_tensor_arg\([^,]+, orch\._worker, tensors\["b', code)

    def test_host_orch_create_tensor_hoisted_to_alloc_intermediates(self):
        """HOST-orch tensor.create lifts to _alloc_intermediates(tensors).

        The simpler L3 runtime forks subworker / chip-worker child processes
        inside w.init(); POSIX shared memory created after fork is invisible
        to inherited children. Intermediate tensors created via
        pl.create_tensor in the HOST orchestrator body must therefore be
        allocated *before* w.init() — codegen splits them into a separate
        _alloc_intermediates(tensors) function that the runtime invokes
        pre-init.
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, buf)
                return result

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                buf: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, buf)
                return result

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        alloc_idx = code.find("def _alloc_intermediates(tensors, world_size=1):")
        host_idx = code.find("def host_orch(")
        assert alloc_idx >= 0, f"Missing _alloc_intermediates in:\n{code}"
        assert host_idx >= 0, f"Missing host_orch in:\n{code}"
        assert alloc_idx < host_idx, "_alloc_intermediates must precede host_orch"

        alloc_block = code[alloc_idx:host_idx]
        host_block = code[host_idx:]

        # Allocation lives in _alloc_intermediates only. SSA renames the local
        # so match by structure rather than the literal source name.
        assert "torch.zeros((64,), dtype=torch.float32).share_memory_()" in alloc_block
        match = re.search(r'tensors\["([^"]+)"\] = torch\.zeros\(', alloc_block)
        assert match is not None, f"No tensors[...] = torch.zeros(...) in:\n{alloc_block}"
        hoisted_name = match.group(1)

        # host_orch must NOT re-allocate the hoisted tensor — but it must
        # still pass it via `tensors["<name>"]` to the chip orchestrator.
        assert "torch.zeros(" not in host_block
        assert f'tensors["{hoisted_name}"]' in host_block

    def test_alloc_intermediates_emitted_when_no_creates(self):
        """HOST orchestrator without tensor.create still gets an empty alloc fn.

        Keeping the symbol present simplifies the runtime contract: it can
        unconditionally call _alloc_intermediates(tensors) before w.init().
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "def _alloc_intermediates(tensors, world_size=1):" in code
        # Body is just `pass` since there are no allocations to hoist.
        alloc_idx = code.find("def _alloc_intermediates(tensors, world_size=1):")
        host_idx = code.find("def host_orch(")
        alloc_block = code[alloc_idx:host_idx]
        assert "    pass" in alloc_block

    def test_tuple_return_pl_tuple(self):
        """Tuple-return worker (pl.Tuple) populates per-element tensors aliases."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                return out_s, out_d

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                s, d = self.chip_orch(a, b, out_s, out_d)
                return s, d

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Each tuple element should get its own tensors[...] alias
        assert code.count('tensors["') >= 2
        # _submit_chip dispatch emitted for chip_orch (comm-less, chip resolved at dispatch)
        assert "_submit_chip" in code
        # Two OUTPUT_EXISTING args for the two Out params
        assert code.count("TensorArgType.OUTPUT_EXISTING") == 2

    def test_tuple_return_builtin_tuple(self):
        """Tuple-return worker (builtin tuple[...]) also produces per-element aliases."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                return out_s, out_d

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                s, d = self.chip_orch(a, b, out_s, out_d)
                return s, d

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Must produce identical structure as the pl.Tuple variant
        assert code.count('tensors["') >= 2
        assert "_submit_chip" in code
        assert code.count("TensorArgType.OUTPUT_EXISTING") == 2


class TestSubWorkerSourceGeneration:
    """Test _emit_sub_worker_module for correct param names and imports."""

    def test_sub_worker_source_param_names_match_signature(self):
        """_user_* function params come from the IR function params."""
        from pypto.backend.pto_backend import _emit_sub_worker_module  # noqa: PLC0415

        @pl.program
        class P:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                assert f is not None

        verify_fn = P.get_function("verify")
        assert verify_fn is not None
        source = _emit_sub_worker_module(verify_fn)
        param_name = verify_fn.params[0].name_hint
        assert f"def _user_verify({param_name}):" in source
        assert f"{param_name} = _tensor_from_continuous(args[0])" in source
        assert f"_user_verify({param_name})" in source

    def test_sub_worker_source_imports_torch(self):
        """Generated SubWorker source includes import torch."""
        from pypto.backend.pto_backend import _emit_sub_worker_module  # noqa: PLC0415

        @pl.program
        class P:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def worker(x: pl.Tensor[[64], pl.FP32]):
                pass

        worker_fn = P.get_function("worker")
        assert worker_fn is not None
        source = _emit_sub_worker_module(worker_fn)
        assert "import torch" in source


class TestChipTaskCollection:
    """Next-level program extraction must follow Submit callees, not just Call."""

    def test_collect_chip_funcs_includes_submit_callee(self):
        """Regression for issue #1707: a kernel reached only via ``pl.submit``.

        Orchestration submits ``down_seed`` through ``pl.submit`` inside a
        ``pl.manual_scope``. The submitted InCore callee must appear in the
        collected chip program, or distributed codegen drops it and fails with
        ``function 'down_seed' not found after validation``.
        """
        from pypto.backend.pto_backend import _collect_chip_task_functions  # noqa: PLC0415

        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def down_seed(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def decode_fwd(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, _a_tid = pl.submit(self.down_seed, x)
                return a

        orch = Input.get_function("decode_fwd")
        assert orch is not None
        chip_funcs = _collect_chip_task_functions(orch, Input)
        names = {f.name for f in chip_funcs}
        assert "down_seed" in names, names


def test_chip_tensor_transfer_slots_ignore_scalar_parameters():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self,
            data: pl.Tensor[[8], pl.INT32],
            n: pl.Scalar[pl.INT32],
            gate: pl.InOut[pl.Tensor[[2], pl.INT32]],
        ):
            value = pl.tensor.read(gate, [0])
            pl.tensor.write(gate, [1], value + n)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.Tensor[[8], pl.INT32],
            n: pl.Scalar[pl.INT32],
            gate: pl.InOut[pl.Tensor[[2], pl.INT32]],
        ):
            self.chip(data, n, gate)

    code = codegen.DistributedCodegen().generate(passes.convert_to_ssa()(Program))
    assert re.search(r'add_chip_tensor_arg\([^\n]+TensorArgType.INPUT, callables, "chip", 0\)', code)
    assert re.search(r'add_chip_tensor_arg\([^\n]+TensorArgType.INOUT, callables, "chip", 1\)', code)
    assert code.count(".add_scalar(") == 1
    compile(code, "host_orch.py", "exec")


def test_c3a_cluster_entry_selects_cluster_imports_and_signature():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(self, x: pl.Tensor[[8], pl.FP32]):
            return x

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def cluster(self, x: pl.Tensor[[8], pl.FP32]):
            self.node(x, worker=0)
            return x

    code = codegen.DistributedCodegen().generate(passes.convert_to_ssa()(Program))
    module = ast.parse(code)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    entry = next(node for node in functions if node.name == "cluster")
    keywords = {arg.arg for arg in entry.args.kwonlyargs}
    assert {"tensors", "callables", "nodes", "node_sizes", "node_bindings", "_keep", "_retain"} <= keywords
    assert "world_size" not in keywords
    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)]
    runtime_import = next(node for node in imports if node.module == "simpler.task_interface")
    assert {"TaskArgs", "TensorArgType"} <= {alias.name for alias in runtime_import.names}
    imported = {alias.name for node in imports for alias in node.names}
    # The selected highest entry determines module imports. Reusing HOST's
    # helper imports would silently retain the wrong argument-routing policy.
    assert not {"make_tensor_arg", "add_chip_tensor_arg", "_submit_chip"} & imported
    assert "cluster._pypto_distributed_entry = True" in code
    assert "def node(" not in code
    compile(code, "cluster_orch.py", "exec")


def test_c3a_cluster_generation_does_not_change_reused_host_codegen():
    @pl.program
    class Host:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(self, x: pl.Tensor[[8], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, x: pl.Tensor[[8], pl.FP32]):
            self.chip(x, worker=0)
            return x

    @pl.program
    class Cluster:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(self, x: pl.Tensor[[8], pl.FP32]):
            return x

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def cluster(self, x: pl.Tensor[[8], pl.FP32]):
            self.node(x, worker=0)
            return x

    generator = codegen.DistributedCodegen()
    host = passes.convert_to_ssa()(Host)
    before = generator.generate(host)
    generator.generate(passes.convert_to_ssa()(Cluster))
    after = generator.generate(host)
    assert after == before
    assert 'callables["chip"]' in after and "_submit_chip(" in after
    assert "nodes[" not in after and "_io_arg(" not in after


@pytest.fixture
def c3b_embedded_program():
    """Keep the complete hierarchy available while selecting one HOST body."""

    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(self, x: pl.Tensor[[8], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def observer(x: pl.Tensor[[8], pl.FP32]):
            pass

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(self, x: pl.Tensor[[8], pl.FP32]):
            self.chip(x, worker=0)

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def cluster(self, x: pl.Tensor[[8], pl.FP32]):
            self.node(x, worker=0)

    program = passes.convert_to_ssa()(Program)
    # Exercise existing native preconditions before the missing private API,
    # so an invalid hierarchy cannot masquerade as the intended RED failure.
    ordinary = codegen.DistributedCodegen().generate(program)
    compile(ordinary, "cluster_orch.py", "exec")
    assert "def cluster(" in ordinary
    return program


@pytest.fixture
def c3b_multiple_host_program():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(self, x: pl.Tensor[[8], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def first(self, x: pl.Tensor[[8], pl.FP32]):
            self.chip(x, worker=0)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def second(self, x: pl.Tensor[[8], pl.FP32]):
            self.chip(x, worker=0)

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def cluster(self, x: pl.Tensor[[8], pl.FP32]):
            self.first(x, worker=0)
            self.second(x, worker=1)

    program = passes.convert_to_ssa()(Program)
    compile(codegen.DistributedCodegen().generate(program), "cluster_orch.py", "exec")
    return program


def test_c3b_embedded_unique_host_has_direct_native_entry(c3b_embedded_program):
    code = codegen.DistributedCodegen()._generate_embedded(c3b_embedded_program)
    module = ast.parse(code)
    entry = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "node")
    assert [arg.arg for arg in entry.args.args] == ["orch", "args", "config"]
    assert not entry.args.posonlyargs and not entry.args.kwonlyargs
    assert entry.args.vararg is None and entry.args.kwarg is None
    assert not entry.args.defaults and not entry.args.kw_defaults
    # The selected entry owns the lowered dispatch body; it is not a wrapper
    # that delegates to a reconstructed program or another orchestration entry.
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_submit_chip"
        for node in ast.walk(entry)
    )
    assert "node._pypto_distributed_entry = True" in code
    assert "def cluster(" not in code
    compile(code, "embedded_node.py", "exec")


def test_c3b_embedded_explicit_host_selection_keeps_full_program(c3b_multiple_host_program):
    code = codegen.DistributedCodegen()._generate_embedded(c3b_multiple_host_program, entry_name="second")
    functions = [node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)]
    assert "second" in functions
    assert not {"first", "cluster", "chip"} & set(functions)
    assert "second._pypto_distributed_entry = True" in code
    compile(code, "embedded_second.py", "exec")


def test_c3b_embedded_ambiguous_host_requires_entry_name(c3b_multiple_host_program):
    with pytest.raises(ValueError, match=r"(?i)ambiguous.*HOST|HOST.*ambiguous"):
        codegen.DistributedCodegen()._generate_embedded(c3b_multiple_host_program, entry_name=None)


@pytest.mark.parametrize("entry_name", ["cluster", "observer", "missing"])
def test_c3b_embedded_rejects_non_host_orchestrator_names(c3b_embedded_program, entry_name):
    with pytest.raises(ValueError, match=r"(?i)HOST.*[Oo]rchestrator|[Oo]rchestrator.*HOST"):
        codegen.DistributedCodegen()._generate_embedded(c3b_embedded_program, entry_name=entry_name)


def test_c3b_embedded_requires_host_orchestrator():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def cluster(self, x: pl.Tensor[[8], pl.FP32]):
            return x

    program = passes.convert_to_ssa()(Program)
    compile(codegen.DistributedCodegen().generate(program), "cluster_orch.py", "exec")
    with pytest.raises(ValueError, match=r"(?i)HOST.*[Oo]rchestrator|[Oo]rchestrator.*HOST"):
        codegen.DistributedCodegen()._generate_embedded(program)


def test_c3b_embedded_mode_does_not_change_reused_standalone_host_codegen(c3b_embedded_program):
    @pl.program
    class Host:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(self, x: pl.Tensor[[8], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, x: pl.Tensor[[8], pl.FP32]):
            self.chip(x, worker=0)

    host = passes.convert_to_ssa()(Host)
    generator = codegen.DistributedCodegen()
    before = generator.generate(host)
    generator._generate_embedded(c3b_embedded_program, entry_name="node")
    after = generator.generate(host)
    assert after == before
    assert 'callables["chip"]' in after and "_submit_chip(" in after
    assert "orch, _args, config, *" in after


def _c3b_api_module(source):
    """Expose API state/marker logic while recording only runtime dispatch.

    The full-source subprocess tests load the actual relative support module.
    These API-only tests retain their recording dispatch fakes, so remove that
    optional import and its aliases, but preserve real builtins/types imports,
    fresh entry aliases and all entry/marker code.
    """
    module = ast.parse(source)
    optional_modules = {
        "simpler.task_interface",
        "pypto.runtime.tensor_arg",
        "pypto.runtime.distributed_runner",
    }
    helper_names = {"make_tensor_arg", "add_chip_tensor_arg", "_submit_chip"}
    module.body = [
        node
        for node in module.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (
                node.module in optional_modules
                or node.level == 1
                and any(alias.name == "_distributed_support" for alias in node.names)
            )
            or isinstance(node, ast.Import)
            and all(alias.name == "torch" for alias in node.names)
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in helper_names for target in node.targets)
        )
    ]
    return module


@pytest.mark.parametrize("entry_name", ["args", "orch", "config", "getattr", "globals"])
def test_c3b_embedded_state_lookup_preserves_valid_entry_names(entry_name):
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, x: pl.Tensor[[8], pl.FP32]):
        return x
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def {entry_name}(self, x: pl.Tensor[[8], pl.FP32]):
        self.chip(x, worker=0)
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster(self, x: pl.Tensor[[8], pl.FP32]):
        self.{entry_name}(x, worker=0)
""")
    program = passes.convert_to_ssa()(program)
    compile(codegen.DistributedCodegen().generate(program), "cluster_orch.py", "exec")
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name=entry_name)
    module = _c3b_api_module(source)
    tensor, chip, args, config = object(), object(), object(), object()
    orch = SimpleNamespace(_worker=object())
    frame = SimpleNamespace(tensors={"x__ssa_v0": tensor}, keepalive=[])
    calls, beginnings = [], []

    def begin_invocation(actual_orch, actual_args, actual_config):
        beginnings.append((actual_orch, actual_args, actual_config))
        return frame

    def append_tensor(task_args, worker, value, tag, callables, name, index):
        assert worker is orch._worker
        assert callables == {"chip": chip} and name == "chip"
        assert (value, tag) == ((tensor, "input") if index == 0 else (prepared, "existing"))
        assert index == len(task_args.tensors) and index in (0, 1)
        task_args.tensors.append((value, tag))

    def submit_chip(actual_orch, callable_id, task_args, actual_config, worker):
        calls.append((actual_orch, callable_id, task_args.tensors, actual_config, worker))

    namespace = {
        "TaskArgs": lambda: SimpleNamespace(tensors=[]),
        "TensorArgType": SimpleNamespace(INPUT="input", OUTPUT_EXISTING="existing"),
        "add_chip_tensor_arg": append_tensor,
        "_submit_chip": submit_chip,
    }
    exec(compile(module, "embedded_entry.py", "exec"), namespace)
    entry = namespace[entry_name]
    # Even this recording-only ABI fixture must prepare the output storage
    # declared for the CHIP callee's inferred, unused tensor return. Preserve
    # the original return-x body and its actual emitted output packing.
    storage_rows = [
        row for row in entry._pypto_embedded_layout["prefork_requirements"] if row[0] == "tensor_storage"
    ]
    assert len(storage_rows) == 1 and storage_rows[0][2:] == ("fp32", (8,))
    prepared = object()
    frame.tensors[storage_rows[0][1]] = prepared
    entry._pypto_embedded_state = SimpleNamespace(
        begin_invocation=begin_invocation,
        callables={"chip": chip},
        sub_ids={},
        world_size=1,
        domain_provider=None,
    )
    # Bind state on the actual function, not on an ABI argument or a fake
    # lookup helper. Names matching ABI locals and Python builtins stay valid.
    entry(orch, args, config)
    assert beginnings == [(orch, args, config)]
    assert calls == [(orch, chip, [(tensor, "input"), (prepared, "existing")], config, 0)]


def test_c3b_embedded_entry_marker_remains_unique_for_runtime_loader(c3b_embedded_program, tmp_path):
    source = codegen.DistributedCodegen()._generate_embedded(c3b_embedded_program, entry_name="node")
    module = _c3b_api_module(source)
    orch_dir = tmp_path / "orchestration"
    orch_dir.mkdir()
    # Use the real module loader and marker discovery without normalizing its
    # candidates. A second module-level alias to the marked function must not
    # make the generated module look like it contains two entry functions.
    (orch_dir / "host_orch.py").write_text(ast.unparse(module) + "\n", encoding="utf-8")
    entry, alloc = _load_orch_entry(tmp_path)
    loaded_module = inspect.getmodule(entry)
    assert loaded_module is not None and entry is loaded_module.node
    assert entry.__name__ == "node" and entry._pypto_distributed_entry is True
    assert list(inspect.signature(entry).parameters) == ["orch", "args", "config"]
    assert alloc is loaded_module._alloc_intermediates


def test_c3b_embedded_mixed_formals_declare_immutable_dense_layout():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT8],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
            enabled: pl.Scalar[pl.BOOL],
            limit: pl.Scalar[pl.INDEX],
            gain: pl.Scalar[pl.FP32],
        ):
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT8],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
            enabled: pl.Scalar[pl.BOOL],
            limit: pl.Scalar[pl.INDEX],
            gain: pl.Scalar[pl.FP32],
        ):
            self.chip(x, count, out, enabled, limit, gain)

    program = passes.convert_to_ssa()(Program)
    compile(codegen.DistributedCodegen().generate(program), "host_orch.py", "exec")
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    module = ast.parse(source)
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "node"
            and target.attr == "_pypto_embedded_layout"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, "Embedded entry must declare its immutable wire layout"
    factory = assignments[0].value
    mapping_proxy_names = {
        alias.asname or alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module == "types"
        for alias in node.names
        if alias.name == "MappingProxyType"
    }
    assert isinstance(factory, ast.Call) and isinstance(factory.func, ast.Name)
    assert factory.func.id in mapping_proxy_names
    # Tensor/scalar pools are independently dense, despite interleaved formals.
    # Explicit Out storage remains caller supplied and adds no implicit slot.
    assert ast.literal_eval(factory.args[0]) == {
        "version": 1,
        "tensor_slots": (
            (0, "x__ssa_v0", 0, "fp32", "In", (8,)),
            (2, "out__ssa_v0", 1, "fp32", "Out", (8,)),
        ),
        "scalar_slots": (
            (1, "count__ssa_v0", 0, "int8"),
            (3, "enabled__ssa_v0", 1, "bool"),
            (4, "limit__ssa_v0", 2, "index"),
            (5, "gain__ssa_v0", 3, "fp32"),
        ),
        "window_slots": (),
        "implicit_outputs": (),
        "prefork_requirements": (),
    }
    compile(source, "embedded_node.py", "exec")


def _c3b_normalize_return_fixture(program):
    program = passes.normalize_return_order()(passes.convert_to_ssa()(program))
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.ReturnParamsExplicit)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    return program


def _c3b_tensor_return_program(case):
    tensor = "pl.Tensor[[8], pl.FP32]"
    params = f"x: {tensor}, count: pl.Scalar[pl.INT32]"
    arguments, body = "x, count", "        return x\n"
    returns = tensor
    if case in {"unrelated_out", "returned_out", "ssa_out_orchestration"}:
        params = f"x: {tensor}, out: pl.Out[{tensor}], count: pl.Scalar[pl.INT32]"
        arguments = "x, out, count"
        if case == "returned_out":
            body = "        return out\n"
        elif case == "ssa_out_orchestration":
            body = "        result = self.chip(x, out, worker=0)\n        return result\n"
    elif case == "internal":
        body = (
            "        scratch = pl.create_tensor([8], dtype=pl.FP32)\n"
            "        result = self.chip(x, scratch, worker=0)\n        return result\n"
        )
    elif case in {"pair_call", "unannotated_pair_call", "pair_submit"}:
        params = f"x: {tensor}, y: {tensor}, count: pl.Scalar[pl.INT32]"
        arguments, body, returns = "x, y, count", "        return x, y\n", f"tuple[{tensor}, {tensor}]"
    if case in {"pair_call", "unannotated_pair_call"}:
        launch = f"        left, right = self.node({arguments}, worker=0)\n"
    elif case == "pair_submit":
        launch = (
            f"        (left, right), task = pl.submit(self.node, {arguments}, worker=0)\n"
            "        consumed, next_task = pl.submit(self.sink, left, right, worker=1, deps=[task])\n"
        )
    else:
        launch = f"        returned = self.node({arguments}, worker=0)\n"
    return_annotation = "" if case.startswith("unannotated_") else f" -> {returns}"
    program = pl.parse_program(
        "@pl.program\nclass Program:\n"
        "    @pl.function(type=pl.FunctionType.Orchestration)\n"
        f"    def chip(self, x: {tensor}, out: pl.Out[{tensor}]) -> {tensor}:\n"
        "        alias = out\n        return alias\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        f"    def node(self, {params}){return_annotation}:\n"
        f"{body}"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        f"    def sink(self, x: {tensor}, out: pl.InOut[{tensor}]) -> {tensor}:\n"
        "        return out\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        f"    def cluster(self, x: {tensor}, y: {tensor}, out: pl.Out[{tensor}],\n"
        "                count: pl.Scalar[pl.INT32]):\n"
        f"{launch}        return 0\n"
    )
    return _c3b_normalize_return_fixture(program)


def _c3b_record_return_caller(source):
    class TaskArgs:
        def __init__(self):
            self.tensors, self.scalars, self.deps = [], [], []

        def add_tensor(self, tensor, tag):
            assert not self.scalars, "Return tensors must precede every scalar slot"
            self.tensors.append((tensor, tag))

        def add_scalar(self, value):
            self.scalars.append(value)

        def add_dep_wait(self, task):
            self.deps.append(task)

    module = _c3b_api_module(source)
    tensors = {
        node.slice.value: object()
        for node in ast.walk(module)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "tensors"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    tensors["count__ssa_v0"] = 9
    calls = []

    def submit_next_level(handle, args, config, *, worker):
        task = object()
        calls.append(SimpleNamespace(handle=handle, args=args, config=config, worker=worker, task=task))
        return task

    namespace = {
        "TaskArgs": TaskArgs,
        "TensorArgType": SimpleNamespace(INPUT="input", OUTPUT_EXISTING="existing", INOUT="inout"),
        "_io_arg": lambda binding, tensor, direction: tensor,
    }
    exec(compile(module, "cluster_return.py", "exec"), namespace)
    # All storage is preseeded to test native packing/aliasing independently
    # of future I0 allocation. The real-wire tests use actual Simpler TaskArgs.
    namespace["cluster"](
        SimpleNamespace(submit_next_level=submit_next_level),
        None,
        object(),
        tensors=tensors,
        callables={"node": [101, 103], "sink": [107, 109]},
        nodes=[7, 13],
        node_sizes=[1, 1],
        node_bindings=[object(), object()],
        _keep=[],
        _retain=False,
    )
    return tensors, calls


def _c3b_final_return(function):
    body = function.body
    result = body if isinstance(body, ir.ReturnStmt) else body.stmts[-1]
    assert isinstance(result, ir.ReturnStmt)
    return result


@pytest.mark.parametrize(
    "case",
    [
        "input",
        "unannotated_input",
        "unrelated_out",
        "returned_out",
        "ssa_out_orchestration",
        "internal",
        "pair_call",
        "unannotated_pair_call",
        "pair_submit",
    ],
)
def test_c3b_tensor_return_layout_matches_caller_slots_and_aliases(case):
    program = _c3b_tensor_return_program(case)
    node = program.functions[program.get_global_var("node")]
    chip = program.functions[program.get_global_var("chip")]
    # These hierarchy functions are excluded by the actual normalization and
    # ReturnParamsExplicit verifier domain. Its PASS is not proof that their
    # SSA aliases became literal params: the CHIP Orchestration alias survives.
    assert node.func_type == pl.FunctionType.Opaque
    assert chip.func_type == pl.FunctionType.Orchestration
    assert _c3b_final_return(chip).value[0] is not chip.params[1]
    if case in {"input", "unannotated_input", "unrelated_out", "returned_out"}:
        expected_param = 1 if case == "returned_out" else 0
        assert _c3b_final_return(node).value[0] is node.params[expected_param]

    caller = codegen.DistributedCodegen().generate(program)
    tensors, calls = _c3b_record_return_caller(caller)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    module = ast.parse(source)
    layout = next(
        ast.literal_eval(statement.value.args[0])
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_pypto_embedded_layout"
            for target in statement.targets
        )
    )
    alias = case in {"returned_out", "ssa_out_orchestration"}
    pair = case in {"pair_call", "unannotated_pair_call", "pair_submit"}
    formal_tensors = 2 if pair or case in {"unrelated_out", "returned_out", "ssa_out_orchestration"} else 1
    output_count = 0 if alias else 2 if pair else 1
    assert layout["implicit_outputs"] == tuple(
        (index, formal_tensors + index, "fp32", (8,)) for index in range(output_count)
    )
    if output_count:
        assert ("return_copy_subworker", 1) in layout["prefork_requirements"]
    else:
        assert layout["prefork_requirements"] == ()
    assert len(calls[0].args.tensors) == formal_tensors + output_count
    assert calls[0].args.scalars == [9]
    assert calls[0].args.tensors[0] == (tensors["x__ssa_v0"], "input")
    if case in {"unrelated_out", "returned_out", "ssa_out_orchestration"}:
        assert calls[0].args.tensors[1] == (tensors["out__ssa_v0"], "existing")
    if alias:
        assert tensors["returned__ssa_v0"] is tensors["out__ssa_v0"]
    elif pair:
        assert tensors["left__ssa_v0"] is calls[0].args.tensors[formal_tensors][0]
        assert tensors["right__ssa_v0"] is calls[0].args.tensors[formal_tensors + 1][0]
    else:
        assert tensors["returned__ssa_v0"] is calls[0].args.tensors[formal_tensors][0]
        assert tensors["returned__ssa_v0"] is not tensors["x__ssa_v0"]
    if case == "pair_submit":
        assert len(calls) == 2
        assert calls[1].args.deps == [calls[0].task]
        assert [value for value, _ in calls[1].args.tensors] == [
            tensors["left__ssa_v0"],
            tensors["right__ssa_v0"],
        ]


@pytest.mark.parametrize("annotated", [True, False])
def test_c3b_consumed_scalar_return_has_user_diagnostic_and_unused_return_is_allowed(annotated):
    template = """
@pl.program
class Program:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, count: pl.Scalar[pl.INT32]){return_annotation}:
        return count
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster(self, count: pl.Scalar[pl.INT32]):
{launch}
        return 0
"""
    template = template.replace("{return_annotation}", " -> pl.Scalar[pl.INT32]" if annotated else "")
    unused = _c3b_normalize_return_fixture(
        pl.parse_program(template.replace("{launch}", "        self.node(count)"))
    )
    compile(codegen.DistributedCodegen().generate(unused), "unused_scalar.py", "exec")
    consumed = _c3b_normalize_return_fixture(
        pl.parse_program(template.replace("{launch}", "        result = self.node(count)"))
    )
    with pytest.raises(ValueError, match=r"(?i)scalar.*return.*not supported|unsupported.*scalar.*return"):
        codegen.DistributedCodegen().generate(consumed)


def test_c3b_embedded_internal_result_schedules_chip_then_return_copy_and_retains_args():
    program = _c3b_tensor_return_program("internal")
    compile(codegen.DistributedCodegen().generate(program), "internal_cluster.py", "exec")
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    module = _c3b_api_module(source)

    class TaskArgs:
        def __init__(self):
            self.tensors, self.scalars = [], []

        def add_tensor(self, tensor, direction):
            self.tensors.append((tensor, direction))

        def add_scalar(self, scalar):
            self.scalars.append(scalar)

        def tensor_count(self):
            return len(self.tensors)

        def scalar_count(self):
            return len(self.scalars)

        def tensor(self, index):
            return self.tensors[index][0]

        def scalar(self, index):
            return self.scalars[index]

    x, scratch, destination = (torch.zeros((8,), dtype=torch.float32).share_memory_() for _ in range(3))
    chip_handle, copy_handle, worker, config = object(), object(), object(), object()
    incoming = TaskArgs()
    incoming.add_tensor(x, "input")
    incoming.add_tensor(destination, "existing")
    incoming.add_scalar(9)
    submissions = []

    def submit_sub(handle, args):
        assert handle is copy_handle
        submissions.append(("copy", args))

    orch = SimpleNamespace(_worker=worker, submit_sub=submit_sub)

    def append_chip_tensor(args, actual_worker, tensor, direction, callables, name, index):
        assert actual_worker is worker and callables == {"chip": chip_handle}
        assert name == "chip" and index == len(args.tensors)
        args.add_tensor(tensor, direction)

    def submit_chip(actual_orch, handle, args, actual_config, rank):
        assert actual_orch is orch and handle is chip_handle
        assert actual_config is config and rank == 0
        submissions.append(("chip", args))

    def convert(actual_worker, tensor):
        assert actual_worker is worker
        return tensor

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Embedded invocation allocated tensor storage after prefork preparation")

    namespace = {
        "TaskArgs": TaskArgs,
        "TensorArgType": SimpleNamespace(INPUT="input", OUTPUT_EXISTING="existing", INOUT="inout"),
        "make_tensor_arg": convert,
        "add_chip_tensor_arg": append_chip_tensor,
        "_submit_chip": submit_chip,
        "torch": SimpleNamespace(zeros=unexpected_allocation, empty=unexpected_allocation),
    }
    exec(compile(module, "internal_embedded.py", "exec"), namespace)
    entry = namespace["node"]
    state = support._EmbeddedEntryState(
        entry._pypto_embedded_layout,
        callables={"chip": chip_handle},
        sub_ids={},
        world_size=1,
        prepared_tensors={"scratch__ssa_v0": scratch},
        domain_provider=None,
        return_copy_callable=copy_handle,
    )
    entry._pypto_embedded_state = state
    # Recording fakes prove the emitted internal CHIP body and return source.
    # The separate real CPU test proves canonical SUB dependency inference.
    entry(orch, incoming, config)
    assert [kind for kind, _ in submissions] == ["chip", "copy"]
    chip_args, copy_args = (args for _, args in submissions)
    assert chip_args.tensors == [(x, "input"), (scratch, "existing")]
    assert copy_args.tensors == [(scratch, "input"), (destination, "existing")]
    assert state._frame.tensors["x__ssa_v0"] is x
    assert state._frame.tensors["scratch__ssa_v0"] is scratch
    assert destination is not scratch and destination is not x
    assert any(value is chip_args for value in state._frame.keepalive)
    assert any(value is copy_args for value in state._frame.keepalive)
    state.complete_invocation()
    assert state._frame is None
    state.close()


def _c3b_domain_program(mode="distinct"):
    right_dtype = "INT32" if mode == "builtin" else "FP32"
    local_param = ", local: pld.DistributedTensor[[4], pl.FP32]" if mode == "local" else ""
    local_arg = ", local" if mode == "local" else ""
    local_body = (
        "        local_buffer = pld.alloc_window_buffer(16)\n"
        "        local = pld.window(local_buffer, [4], dtype=pl.FP32)\n"
        if mode == "local"
        else ""
    )
    right_body = (
        "        right = left\n"
        if mode == "repeated"
        else "        right_buffer = pld.alloc_window_buffer(16)\n"
        f"        right = pld.window(right_buffer, [4], dtype=pl.{right_dtype})\n"
    )
    worker = 1 if mode == "repeated" else 3
    program = pl.parse_program(
        f"""
@pl.program
class Domains:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],
             left: pld.DistributedTensor[[4], pl.FP32], right: pld.DistributedTensor[[4], pl.{right_dtype}]"""
        + local_param
        + f""",
             out: pl.Out[pl.Tensor[[4], pl.FP32]], left_ctx: pld.CommCtx):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],
             left: pld.DistributedTensor[[4], pl.FP32], right: pld.DistributedTensor[[4], pl.{right_dtype}],
             out: pl.Out[pl.Tensor[[4], pl.FP32]], left_ctx: pld.CommCtx):
"""
        + local_body
        + f"        self.chip(x, count, left, right{local_arg}, out, left_ctx, worker={worker})\n"
        + "        return 0\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        "    def root(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],\n"
        "             out: pl.Out[pl.Tensor[[4], pl.FP32]]):\n"
        "        left_buffer = pld.alloc_window_buffer(16)\n"
        "        left = pld.window(left_buffer, [4], dtype=pl.FP32)\n"
        + right_body
        + "        original = pld.system.get_comm_ctx(left)\n"
        "        self.node(x, count, left, right, out, original, worker=0)\n"
        "        return 0\n"
    )
    for transform in (
        passes.convert_to_ssa,
        passes.synthesize_allreduce_signals,
        passes.materialize_comm_domain_scopes,
        passes.lower_host_tensor_collectives,
        passes.derive_call_directions,
        passes.materialize_dist_tensor_ctx,
        passes.normalize_return_order,
    ):
        program = transform()(program)
    if mode == "builtin":
        # Codegen fixture for an already-lowered builtin. Public inherited
        # collective lowering still needs a separate pass checkpoint, and a
        # pre-materialization builtin currently loses its result-type WB during
        # materialization roundtrip. Construct the correctly typed call only
        # AFTER the real C2b pipeline; never relax its verification instruments.
        node = program.functions[program.get_global_var("node")]
        assert isinstance(node.body, ir.SeqStmts)
        signal = node.params[3]
        builtin = ir.Call(
            ir.get_op("builtin.tensor.barrier"),
            [signal],
            {},
            {"worker": ir.ConstInt(3, pl.INDEX, node.span), "arg_directions": [ir.ArgDirection.InOut]},
            signal.type,
            node.span,
        )
        body = ir.SeqStmts(
            [*node.body.stmts[:-1], ir.EvalStmt(builtin, node.span), node.body.stmts[-1]], node.span
        )
        replacement = ir.Function(
            node.name,
            list(zip(node.params, node.param_directions)),
            node.return_types,
            body,
            node.span,
            type=node.func_type,
            level=node.level,
            role=node.role,
            attrs=node.attrs,
            requires_runtime_binding=node.requires_runtime_binding,
            ir_stage=node.ir_stage,
        )
        program = ir.Program(
            [replacement if function is node else function for function in program.functions.values()],
            program.name,
            program.span,
        )
        program = passes.normalize_return_order()(program)
        ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program, format=False)))
        ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.CommDomainScopesMaterialized,
        passes.IRProperty.DistTensorCtxMaterialized,
        passes.IRProperty.ReturnParamsExplicit,
        passes.IRProperty.TypeChecked,
    ):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    compile(codegen.DistributedCodegen().generate(program), "owned_domain_cluster.py", "exec")
    node = program.functions[program.get_global_var("node")]
    chip = program.functions[program.get_global_var("chip")]
    inherited = [
        param.type.window_buffer for param in node.params if isinstance(param.type, ir.DistributedTensorType)
    ]
    assert len(inherited) == 2
    assert (inherited[0] is inherited[1]) == (mode == "repeated")
    assert chip.params[2].type.window_buffer is inherited[0]
    assert chip.params[3].type.window_buffer is inherited[1]
    if mode == "local":
        local_window = chip.params[4].type.window_buffer
        assert local_window is not inherited[0]
    return program


def _c3b_embedded_layout(source):
    module = ast.parse(source)
    return next(
        ast.literal_eval(statement.value.args[0])
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_pypto_embedded_layout"
            for target in statement.targets
        )
    )


def test_c3b_inherited_window_layout_uses_final_ctx_pairs_and_dense_pool_ordinals():
    program = _c3b_domain_program()
    node = program.functions[program.get_global_var("node")]
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    layout = _c3b_embedded_layout(source)
    # C2b appends one ctx for EACH DT, after the original explicit ctx. The
    # original name deliberately collides with a generated ctx base name.
    assert len(node.params) == 8
    assert all(isinstance(param.type, ir.CommCtxType) for param in node.params[5:])
    assert layout["tensor_slots"] == (
        (0, node.params[0].name_hint, 0, "fp32", "In", (4,)),
        (4, node.params[4].name_hint, 1, "fp32", "Out", (4,)),
    )
    assert layout["scalar_slots"] == (
        (1, node.params[1].name_hint, 0, "int32"),
        (5, node.params[5].name_hint, 1, "comm_ctx"),
        (6, node.params[6].name_hint, 2, "comm_ctx"),
        (7, node.params[7].name_hint, 3, "comm_ctx"),
    )
    assert layout["window_slots"] == tuple(
        (
            position,
            node.params[position].name_hint,
            ctx_position,
            scalar_index,
            node.params[position].type.window_buffer.name_hint,
            "fp32",
            "In",
            (4,),
        )
        for position, ctx_position, scalar_index in ((2, 6, 2), (3, 7, 3))
    )
    assert ("inherited_domain_tokens", 3) in layout["prefork_requirements"]
    assert layout["implicit_outputs"] == ()


@pytest.mark.parametrize("mode", ["repeated", "distinct", "local", "builtin"])
def test_c3b_embedded_inherited_windows_select_target_context_and_share_domain_token(mode, monkeypatch):
    program = _c3b_domain_program(mode)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    layout = _c3b_embedded_layout(source)
    factory = getattr(support, "_EmbeddedNodeState", None)
    assert callable(factory), "Missing canonical prefork node/domain token state"
    tokens = tuple(torch.zeros((1,), dtype=torch.int32).share_memory_() for _ in range(2))
    node_state = factory(local_worker_ids=(1, 3), prepared_domain_tokens=(tokens,))
    worker = 1 if mode == "repeated" else 3
    x, out = (torch.zeros((4,), dtype=torch.float32).share_memory_() for _ in range(2))
    resolutions, views, submissions, provider_events = [], [], [], []
    descriptors = {}

    class NamedBuffer:
        def __init__(self, domain, actual_worker, name):
            self.key = (domain, actual_worker, name)
            descriptors[self.key] = object()

        def tensor(self, *, shapes, dtype):
            assert shapes == (4,) and dtype in ("fp32", "int32")
            views.append(self.key)
            return descriptors[self.key]

    def contexts(domain_name, names, *, local=False):
        return {
            actual_worker: SimpleNamespace(
                domain_rank=rank,
                domain_size=12,
                device_ctx=(20000 if local else 10000) + actual_worker,
                buffers={name: NamedBuffer(domain_name, actual_worker, name) for name in names},
            )
            for actual_worker, rank in ((1, 4), (3, 9))
        }

    class Domain:
        # Strict recording boundary matches GlobalCommDomainView's actual API;
        # it supplies no local_view and makes no claim of a physical VMM domain.
        committed = True
        domain_id, generation = 801, 5

        def __init__(self, names):
            self.name = "global"
            self.contexts = contexts(self.name, names)

        def __getitem__(self, actual_worker):
            return self.contexts[actual_worker]

    class LocalDomain:
        # CommDomainHandle's local API has allocation_id, not global identity.
        allocation_id = 601

        def __init__(self, name, names):
            self.name, self.workers = name, (worker,)
            self.contexts = contexts("local", names, local=True)

        def __getitem__(self, actual_worker):
            return self.contexts[actual_worker]

        def __enter__(self):
            provider_events.append("enter")
            return self

        def __exit__(self, *args):
            provider_events.append("exit")

    inherited_names = {row[4] for row in layout["window_slots"]}
    global_domain = Domain(inherited_names)

    class TaskArgs:
        def __init__(self):
            self.tensors, self.scalars = [], []

        def add_tensor(self, value, direction):
            assert not self.scalars
            self.tensors.append((value, direction))

        def add_scalar(self, value):
            self.scalars.append(value)

        def tensor_count(self):
            return len(self.tensors)

        def scalar_count(self):
            return len(self.scalars)

        def tensor(self, index):
            return self.tensors[index][0]

        def scalar(self, index):
            return self.scalars[index]

    incoming = TaskArgs()
    incoming.add_tensor(x, "input")
    incoming.add_tensor(out, "existing")
    for value in (9, 801, 801, 801):
        incoming.add_scalar(value)
    chip_handle, builtin_handle, actual_worker = object(), object(), object()
    config = SimpleNamespace(aicpu_thread_num=2)
    callables = {"chip": chip_handle}
    if mode == "builtin":
        callables["builtin.tensor.barrier__fp32"] = builtin_handle

    def get_domain(domain_id):
        resolutions.append(domain_id)
        assert domain_id == 801
        return global_domain

    def provider(**kwargs):
        assert kwargs["workers"] == [worker]
        assert kwargs["window_size"] >= 16
        assert len(kwargs["buffers"]) == 1 and kwargs["buffers"][0].count == 16
        domain = LocalDomain(kwargs["name"], [kwargs["buffers"][0].name])
        assert not hasattr(domain, "domain_id") and not hasattr(domain, "generation")
        return domain

    def convert(bound_worker, value):
        assert bound_worker is actual_worker
        return value

    def append(args, bound_worker, value, direction, actual_callables, name, index):
        assert bound_worker is actual_worker and actual_callables is callables and name == "chip"
        args.add_tensor(value, direction)

    def submit(orch, handle, args, actual_config, target):
        assert handle is chip_handle and actual_config is config and target == worker
        submissions.append(args)

    def submit_builtin(handle, args, actual_config, *, worker):
        assert handle is builtin_handle and worker == 3
        assert actual_config.aicpu_thread_num == config.aicpu_thread_num
        submissions.append(args)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("First invocation allocated Torch storage instead of using prefork tokens")

    prepared = {}
    local_rows = [row for row in layout["prefork_requirements"] if row[0] == "local_domain_tokens"]
    assert bool(local_rows) == (mode == "local")
    for row in local_rows:
        prepared[row[1]] = tuple(torch.zeros((1,), dtype=torch.int32).share_memory_() for _ in range(4))
    namespace = {
        "TaskArgs": TaskArgs,
        "TensorArgType": SimpleNamespace(INPUT="input", OUTPUT_EXISTING="existing", INOUT="inout"),
        "DataType": SimpleNamespace(FLOAT32="fp32", INT32="int32"),
        "CallConfig": SimpleNamespace,
        "CommBufferSpec": lambda **kwargs: SimpleNamespace(**kwargs),
        "make_tensor_arg": convert,
        "add_chip_tensor_arg": append,
        "_submit_chip": submit,
        "torch": SimpleNamespace(zeros=unexpected_allocation, empty=unexpected_allocation),
    }
    exec(compile(_c3b_api_module(source), "embedded_domains.py", "exec"), namespace)
    entry = namespace["node"]
    state = support._EmbeddedEntryState(
        entry._pypto_embedded_layout,
        callables=callables,
        sub_ids={},
        world_size=4,
        prepared_tensors=prepared,
        domain_provider=provider,
        node_state=node_state,
    )
    entry._pypto_embedded_state = state
    monkeypatch.setattr(torch, "zeros", unexpected_allocation)
    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    entry(
        SimpleNamespace(
            _worker=actual_worker, get_global_domain=get_domain, submit_next_level=submit_builtin
        ),
        incoming,
        config,
    )
    assert resolutions == [801] and len(submissions) == (2 if mode == "builtin" else 1)
    sent = submissions[0]
    assert sent.tensors[0] == (x, "input")
    assert all(key[1] == worker for key in views)
    assert [value for value, _ in sent.tensors[1:3]] == [
        descriptors[("global", worker, row[4])] for row in layout["window_slots"]
    ]
    business_count = 5 if mode == "local" else 4
    assert sent.tensors[business_count - 1] == (out, "existing")
    order_tokens = sent.tensors[business_count:]
    assert len(order_tokens) == (2 if mode == "local" else 1)
    assert all(direction == "inout" for _, direction in order_tokens)
    assert any(value is tokens[(1, 3).index(worker)] for value, _ in order_tokens)
    expected_scalars = [9, 10000 + worker, 10000 + worker, 10000 + worker]
    if mode == "local":
        assert provider_events == ["enter", "exit"]
        assert any(value is prepared[local_rows[0][1]][worker] for value, _ in order_tokens)
        expected_scalars.append(20000 + worker)
    else:
        assert provider_events == []
    assert sent.scalars == expected_scalars
    if mode == "builtin":
        builtin_args = submissions[1]
        assert builtin_args.scalars == [12, 10003]
        assert len(builtin_args.tensors) == 2
        assert builtin_args.tensors[0] == (descriptors[("global", 3, layout["window_slots"][1][4])], "inout")
        assert builtin_args.tensors[1][0] is order_tokens[0][0]
    assert any(value is sent for value in state._frame.keepalive)
    state.complete_invocation()
    state.close()
    node_state.retire_domain(801, 5)
    node_state.close()


def _c3b_descriptor_program(mode):
    """Keep descriptor operations inside a complete, verified hierarchy."""
    dynamic = mode in {"bare", "affine", "dynamic_storage"}
    dim = "M * 2" if mode == "affine" else "M"
    shape = f"[2, {dim}, 4]" if dynamic else "[2, 3, 4]"
    chip_shape = "[M, 4]" if dynamic else "[6, 4]"
    if mode in {"bare", "affine"}:
        body = (
            "        sliced = pl.tensor.slice(x, [1, M, 4], [1, 0, 0], drop_dims=[0])\n"
            "        target = pl.tensor.slice(out, [1, M, 4], [1, 0, 0], drop_dims=[0])\n"
            "        self.chip(sliced, target, worker=0)\n"
        )
    elif mode in {"view", "reshape"}:
        body = (
            f"        viewed = pl.tensor.{mode}(x, [6, 4])\n"
            f"        target = pl.tensor.{mode}(out, [6, 4])\n"
            "        self.chip(viewed, target, worker=0)\n"
        )
    elif mode == "assemble":
        body = "        written = pl.tensor.assemble(out, x, [0, 0, 0])\n"
    else:
        assert mode == "dynamic_storage"
        body = "        scratch = pl.create_tensor([M, 4], dtype=pl.FP32)\n"
    program = pl.parse_program(
        ('M = pl.dynamic("M")\n' if dynamic else "") + "@pl.program\nclass Descriptors:\n"
        "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n"
        f"    def chip(self, x: pl.Tensor[{chip_shape}, pl.FP32],\n"
        f"             out: pl.Out[pl.Tensor[{chip_shape}, pl.FP32]]):\n"
        "        return 0\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        f"    def node(self, x: pl.Tensor[{shape}, pl.FP32],\n"
        f"             out: pl.Out[pl.Tensor[{shape}, pl.FP32]]):\n" + body + "        return 0\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        f"    def root(self, x: pl.Tensor[{shape}, pl.FP32],\n"
        f"             out: pl.Out[pl.Tensor[{shape}, pl.FP32]]):\n"
        "        self.node(x, out, worker=0)\n"
        "        return 0\n"
    )
    program = _c3b_normalize_return_fixture(program)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    compile(codegen.DistributedCodegen().generate(program), "descriptor_cluster.py", "exec")
    return program


@pytest.mark.parametrize("mode", ["bare", "affine"])
def test_c3b_descriptor_dynamic_formals_use_serializable_shape_expressions(mode):
    program = _c3b_descriptor_program(mode)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "dynamic_descriptor_entry.py", "exec")
    layout = _c3b_embedded_layout(source)
    # Private v1 retains integer dimensions. Dynamic dimensions use a closed,
    # immutable opcode tuple, which C3c can persist without eval or IR objects.
    dimension = ("var", "M")
    if mode == "affine":
        dimension = ("mul", dimension, 2)
    assert layout["tensor_slots"] == (
        (0, "x__ssa_v0", 0, "fp32", "In", (2, dimension, 4)),
        (1, "out__ssa_v0", 1, "fp32", "Out", (2, dimension, 4)),
    )
    assert layout["prefork_requirements"] == ()
    assert layout["implicit_outputs"] == ()
    assert ".shape[" not in source
    assert ".shapes[1]" in source
    module = ast.parse(source)
    entry = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "node")
    bound = next(
        index
        for index, statement in enumerate(entry.body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "M" for target in statement.targets)
    )
    dispatch = next(
        index
        for index, statement in enumerate(entry.body)
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_submit_chip"
            for node in ast.walk(statement)
        )
    )
    assert bound < dispatch


@pytest.mark.parametrize("mode", ["view", "reshape"])
def test_c3b_descriptor_contiguous_shape_views_generate(mode):
    program = _c3b_descriptor_program(mode)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "shape_descriptor_entry.py", "exec")
    assert _c3b_embedded_layout(source)["prefork_requirements"] == ()
    assert 'tensors["viewed__ssa_v0"]' in source
    assert "_submit_chip(" in source


@pytest.mark.parametrize(
    ("mode", "diagnostic"),
    [
        ("assemble", r"(?i)(embedded|descriptor).*assemble.*not supported"),
        ("dynamic_storage", r"(?i)embedded.*(static|constant|prefork).*shape"),
    ],
)
def test_c3b_descriptor_unsupported_operations_fail_before_submission(mode, diagnostic):
    program = _c3b_descriptor_program(mode)
    # These are legal native operations. The embedded implementation must give
    # its own explicit capability diagnostic before loading or invoking a node.
    with pytest.raises(ValueError, match=diagnostic):
        codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")


def _c3b_finish_dt_fixture(program, *, convert=True):
    transforms = [passes.convert_to_ssa] if convert else []
    transforms += [
        passes.synthesize_allreduce_signals,
        passes.materialize_comm_domain_scopes,
        passes.lower_host_tensor_collectives,
        passes.derive_call_directions,
        passes.materialize_dist_tensor_ctx,
        passes.normalize_return_order,
    ]
    for transform in transforms:
        program = transform()(program)
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.CommDomainScopesMaterialized,
        passes.IRProperty.DistTensorCtxMaterialized,
        passes.IRProperty.TypeChecked,
        passes.IRProperty.ReturnParamsExplicit,
    ):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program, format=False)))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    compile(codegen.DistributedCodegen().generate(program), "dt_owned_cluster.py", "exec")
    return program


def _c3b_dt_geometry_program():
    program = pl.parse_program("""
@pl.program
class Geometry:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, left: pld.DistributedTensor[[2, 3], pl.FP32],
             right: pld.DistributedTensor[[2, 3], pl.FP32]):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, left: pld.DistributedTensor[[2, 3], pl.FP32],
             right: pld.DistributedTensor[[2, 3], pl.FP32]):
        self.chip(left, right, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        storage = pld.alloc_window_buffer(128)
        data = pld.window(storage, [2, 3], dtype=pl.FP32)
        self.node(data, data, worker=0)
        return 0
""")
    program = passes.convert_to_ssa()(program)
    root = program.functions[program.get_global_var("root")]
    base = root.body.stmts[0].var
    replacements = {}
    # An already-lowered geometry fixture, not a claim that C2b computes slice
    # MemRefs. Both offsets are relative to this owned named slot's byte0.
    for name in ("node", "chip"):
        function = program.functions[program.get_global_var(name)]
        params = [
            ir.Var(
                old.name_hint,
                ir.DistributedTensorType(
                    [2, 3],
                    pl.FP32,
                    ir.MemRef(base, offset, 44, old.span),
                    ir.TensorView(stride=[8, 1], layout=ir.TensorLayout.ND),
                ),
                old.span,
            )
            for old, offset in zip(function.params, (36, 68))
        ]
        body = function.body
        if name == "node":
            previous = body.stmts[0].expr
            launch = ir.Call(
                previous.op, params, previous.kwargs, previous.attrs, previous.type, previous.span
            )
            body = ir.SeqStmts([ir.EvalStmt(launch, previous.span), body.stmts[-1]], function.span)
        replacements[name] = ir.Function(
            name,
            list(zip(params, function.param_directions)),
            function.return_types,
            body,
            function.span,
            type=function.func_type,
            level=function.level,
            role=function.role,
            attrs=function.attrs,
        )
    program = ir.Program(
        [replacements.get(function.name, function) for function in program.functions.values()],
        program.name,
        program.span,
    )
    program = _c3b_finish_dt_fixture(program, convert=False)
    node = program.functions[program.get_global_var("node")]
    chip = program.functions[program.get_global_var("chip")]
    for index, offset in enumerate((36, 68)):
        actual = node.params[index].type
        assert (
            actual.window_buffer is node.params[0].type.window_buffer is chip.params[index].type.window_buffer
        )
        assert actual.memref.byte_offset_.value == offset
        assert actual.memref.base_ is actual.window_buffer.base
        assert [stride.value for stride in actual.tensor_view.stride] == [8, 1]
    return program


def _c3b_record_dt_geometry(source):
    """Record generated Python geometry; real native descriptors are tested separately."""
    layout = _c3b_embedded_layout(source)
    records, calls = [], []
    token = object()

    class TaskArgs:
        def __init__(self):
            self.tensors, self.scalars = [], []

        def add_tensor(self, value, tag):
            self.tensors.append((value, tag))

        def add_scalar(self, value):
            self.scalars.append(value)

    class NamedBuffer:
        def tensor(self, **geometry):
            records.append(geometry)
            return geometry

    context = SimpleNamespace(
        buffers={row[4]: NamedBuffer() for row in layout["window_slots"]},
        device_ctx=12001,
        domain_size=12,
        domain_rank=9,
    )

    class Domain:
        def __getitem__(self, worker):
            assert worker == 1
            return context

    domain = Domain()
    tensors = {name: domain for _, name, _, dtype in layout["scalar_slots"] if dtype == "comm_ctx"}
    frame = support._EmbeddedFrame(tensors, object())
    frame.ordering_tokens = lambda handles, rank: (token,)
    namespace = {
        "TaskArgs": TaskArgs,
        "TensorArgType": SimpleNamespace(INPUT="input", OUTPUT_EXISTING="existing", INOUT="inout"),
        "DataType": SimpleNamespace(FLOAT32="fp32"),
        "make_tensor_arg": lambda worker, value: value,
        "_submit_chip": lambda orch, handle, args, config, worker: calls.append(args),
    }
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.ImportFrom) and statement.module is None:
            namespace.update({alias.asname or alias.name: support for alias in statement.names})
    exec(compile(_c3b_api_module(source), "dt_geometry_entry.py", "exec"), namespace)
    state = SimpleNamespace(
        begin_invocation=lambda *args: frame,
        callables={"chip": object()},
        sub_ids={},
        world_size=2,
        domain_provider=None,
    )
    namespace["node"]._pypto_embedded_state = state
    namespace["node"](SimpleNamespace(_worker=object()), object(), object())
    return records, calls


def test_c3b_dt_prepared_geometry_preserves_slot_origin_strides_and_alias_token():
    program = _c3b_dt_geometry_program()
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    records, calls = _c3b_record_dt_geometry(source)
    assert records == [
        {"shapes": (2, 3), "dtype": "fp32", "strides": (8, 1), "byte_offset": offset} for offset in (36, 68)
    ]
    assert len(calls) == 1 and len(calls[0].tensors) == 3
    assert calls[0].scalars == [12001, 12001]
    layout = _c3b_embedded_layout(source)
    assert layout["tensor_slots"] == layout["implicit_outputs"] == ()


@pytest.mark.parametrize("kind", ["call", "submit"])
def test_c3b_dt_return_tuple_aliases_reuse_selected_ctx_without_payload_slots(kind):
    launch = (
        "        packed = self.stage(left, right, explicit, worker=1)\n"
        if kind == "call"
        else "        (returned_right, returned_left, repeated_right), task = "
        "pl.submit(self.stage, left, right, explicit, worker=1)\n"
        "        packed = (returned_right, returned_left, repeated_right)\n"
    )
    consume = (
        "        self.sink(selected, ctx, worker=1)\n"
        if kind == "call"
        else "        unused, next_task = pl.submit(self.sink, selected, ctx, worker=1, deps=[task])\n"
    )
    program = pl.parse_program(
        "@pl.program\nclass Metadata:\n"
        "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n"
        "    def stage(self, left: pld.DistributedTensor[[4], pl.FP32],\n"
        "              right: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx) -> "
        "tuple[pld.DistributedTensor[[4], pl.FP32], pld.DistributedTensor[[4], pl.FP32], "
        "pld.DistributedTensor[[4], pl.FP32]]:\n"
        "        return right, left, right\n"
        "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n"
        "    def sink(self, data: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx) -> "
        "pld.DistributedTensor[[4], pl.FP32]:\n"
        "        return data\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        "    def node(self, left: pld.DistributedTensor[[4], pl.FP32],\n"
        "             right: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx):\n"
        + launch
        + "        alias = packed\n        selected = alias[2]\n"
        "        ctx = pld.system.get_comm_ctx(selected)\n" + consume + "        return 0\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        "    def root(self):\n"
        "        left_storage = pld.alloc_window_buffer(16)\n"
        "        left = pld.window(left_storage, [4], dtype=pl.FP32)\n"
        "        right_storage = pld.alloc_window_buffer(16)\n"
        "        right = pld.window(right_storage, [4], dtype=pl.FP32)\n"
        "        explicit = pld.system.get_comm_ctx(left)\n"
        "        self.node(left, right, explicit, worker=0)\n        return 0\n"
    )
    program = _c3b_finish_dt_fixture(program)
    node = program.functions[program.get_global_var("node")]
    # C2b tracks metadata provenance. Return arity stays three: it adds no ctx
    # result wire values and every DT result still refers to its original slot.
    stage = program.functions[program.get_global_var("stage")]
    assert len(_c3b_final_return(stage).value) == 3
    assert all(isinstance(value.type, ir.DistributedTensorType) for value in _c3b_final_return(stage).value)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "dt_metadata_entry.py", "exec")
    layout = _c3b_embedded_layout(source)
    assert layout["tensor_slots"] == layout["implicit_outputs"] == ()
    assert len(layout["scalar_slots"]) == 3
    assert [param.name_hint for param in node.params[-2:]] == [row[1] for row in layout["scalar_slots"][-2:]]
    assert not any(
        row[0] in {"tensor_storage", "return_copy_subworker"} for row in layout["prefork_requirements"]
    )
    assert 'tensors["packed' not in source and 'tensors["selected' not in source
    # HOST retains runtime queries. C2b can reuse the right input's suffix or,
    # after a reconstructed metadata tuple, synthesize a query of that exact DT.
    calls = [
        statement.value if isinstance(statement, ir.AssignStmt) else getattr(statement, "expr", None)
        for statement in node.body.stmts
    ]
    sink = next(call for call in calls if isinstance(call, (ir.Call, ir.Submit)) and call.op.name == "sink")
    if sink.args[-1] is not node.params[-1]:
        definitions = [
            statement.value
            for statement in node.body.stmts
            if isinstance(statement, ir.AssignStmt) and statement.var is sink.args[-1]
        ]
        assert len(definitions) == 1, ir.python_print(program, format=False)
        definition = definitions[0]
        assert isinstance(definition, ir.Call) and definition.op.name == "pld.system.get_comm_ctx"
        assert len(definition.args) == 1 and definition.args[0] is sink.args[0]
        assert definition.args[0].type.window_buffer is node.params[1].type.window_buffer
    query = next(
        call for call in calls if isinstance(call, ir.Call) and call.op.name == "pld.system.get_comm_ctx"
    )
    assert query.args[0].type.window_buffer is node.params[1].type.window_buffer
    if kind == "submit":
        assert sink.deps and sink.type.types[-1].dtype == pl.TASK_ID


def test_c3b_dt_public_nonzero_slice_has_geometry_or_early_user_diagnostic():
    program = pl.parse_program("""
@pl.program
class PublicSlice:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[2, 3], pl.FP32]):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, data: pld.DistributedTensor[[4, 8], pl.FP32]):
        selected = pl.tensor.slice(data, [2, 3], [1, 1])
        self.chip(selected, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        storage = pld.alloc_window_buffer(128)
        data = pld.window(storage, [4, 8], dtype=pl.FP32)
        self.node(data, worker=0)
        return 0
""")
    program = _c3b_finish_dt_fixture(program)
    try:
        source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    except ValueError as error:
        assert re.search(r"(?i)embedded.*(distributed|window).*slice.*(geometry|not supported)", str(error))
        return
    records, calls = _c3b_record_dt_geometry(source)
    assert records == [{"shapes": (2, 3), "dtype": "fp32", "strides": (8, 1), "byte_offset": 36}]
    assert len(calls) == 1


def test_c3b_dt_ctx_tuple_projection_keeps_original_ctx_distinct_from_window_ctx():
    program = pl.parse_program("""
@pl.program
class ContextTuple:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def sink(self, data: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, left: pld.DistributedTensor[[4], pl.FP32],
             right: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx):
        packed = (left, right, explicit)
        alias = packed
        chosen_ctx = alias[2]
        self.sink(left, chosen_ctx, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        left_storage = pld.alloc_window_buffer(16)
        left = pld.window(left_storage, [4], dtype=pl.FP32)
        right_storage = pld.alloc_window_buffer(16)
        right = pld.window(right_storage, [4], dtype=pl.FP32)
        explicit = pld.system.get_comm_ctx(right)
        self.node(left, right, explicit, worker=0)
        return 0
""")
    program = _c3b_finish_dt_fixture(program)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "ctx_tuple_entry.py", "exec")
    layout = _c3b_embedded_layout(source)
    assert layout["tensor_slots"] == layout["implicit_outputs"] == ()
    assert [row[2] for row in layout["scalar_slots"]] == [0, 1, 2]
    assert layout["scalar_slots"][0][1] == "explicit__ssa_v0"
    assert 'tensors["packed' not in source and 'tensors["chosen_ctx' not in source


def test_c3b_dt_default_pipeline_plain_window_normalizes_only_whole_slot_origin():
    program = pl.parse_program("""
@pl.program
class DefaultWindow:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, data: pld.DistributedTensor[[4], pl.FP32]):
        selected = data
        self.chip(selected, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        storage = pld.alloc_window_buffer(16)
        data = pld.window(storage, [4], dtype=pl.FP32)
        self.node(data, worker=0)
        return 0
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        # Actual production strategy; attach no MemRef or TensorView manually.
        program = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    finally:
        backend.reset_for_testing()
    node = program.functions[program.get_global_var("node")]
    tensor = node.params[0].type
    assert tensor.memref.base_ is not tensor.window_buffer.base
    assert tensor.memref.byte_offset_.value == 0 and tensor.memref.size_ == 16
    assert tensor.window_buffer.size.value == 16
    assert [dim.value for dim in tensor.shape] == [4]
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program, format=False)))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    compile(codegen.DistributedCodegen().generate(program), "default_cluster.py", "exec")
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "default_plain_window_entry.py", "exec")
    assert _c3b_embedded_layout(source)["tensor_slots"] == ()


def _c3b_control_flow_program(mode):
    bodies = {
        "scalar": """        count = seed
        for i in pl.range(2):
            self.scalar(count, worker=1)
            count = count + pl.const(1, pl.INT32)
        self.scalar(count, worker=1)
""",
        "zero": """        count = seed
        for i in pl.range(0):
            self.scalar(count, worker=1)
            count = count + pl.const(1, pl.INT32)
        self.scalar(count, worker=1)
""",
        "nested": """        count = seed
        for i in pl.range(2):
            inner = count
            for j in pl.range(2):
                self.scalar(inner, worker=1)
                inner = inner + pl.const(1, pl.INT32)
            count = inner
        self.scalar(count, worker=1)
""",
        "swap": """        for i, (a, da, p, ca, b, db, q, cb) in pl.range(
            2, init_values=(x, data, seed, explicit, y, data, other, explicit)):
            self.mixed(a, p, da, ca, b, q, db, cb, worker=1)
            a_out, da_out, p_out, ca_out, b_out, db_out, q_out, cb_out = pl.yield_(b, da, q, ca, a, db, p, cb)
        self.mixed(a_out, p_out, da_out, ca_out, b_out, q_out, db_out, cb_out, worker=1)
""",
        "metadata_for": """        for i, (carry,) in pl.range(2, init_values=(data,)):
            result = pl.yield_(carry)
        self.window(result, worker=1)
""",
        "if_mixed": """        if flag:
            chosen, chosen_window, count = pl.yield_(x, data, seed)
        else:
            chosen, chosen_window, count = pl.yield_(y, data, seed + pl.const(1, pl.INT32))
        self.observe(chosen, count, chosen_window, explicit, worker=1)
""",
        "if_metadata": """        if flag:
            chosen_window = pl.yield_(data)
        else:
            chosen_window = pl.yield_(data)
        self.window(chosen_window, worker=1)
""",
        "if_ctx": """        if flag:
            result = pl.yield_(explicit)
        else:
            result = pl.yield_(explicit)
        self.observe(x, seed, data, result, worker=1)
""",
        "ambiguous_ctx_for": """        for i, (ctx,) in pl.range(2, init_values=(explicit,)):
            self.observe(x, seed, data, ctx, worker=1)
            next_ctx = pld.system.get_comm_ctx(data)
            result = pl.yield_(next_ctx)
""",
        "ambiguous_ctx_if": """        next_ctx = pld.system.get_comm_ctx(data)
        if flag:
            result = pl.yield_(explicit)
        else:
            result = pl.yield_(next_ctx)
        self.observe(x, seed, data, result, worker=1)
""",
    }
    helper = (
        "    def scalar(self, count: pl.Scalar[pl.INT32]):\n"
        if mode in {"scalar", "zero", "nested"}
        else "    def mixed(self, a: pl.Tensor[[4], pl.FP32], p: pl.Scalar[pl.INT32],\n"
        "              da: pld.DistributedTensor[[4], pl.FP32], ca: pld.CommCtx,\n"
        "              b: pl.Tensor[[4], pl.FP32], q: pl.Scalar[pl.INT32],\n"
        "              db: pld.DistributedTensor[[4], pl.FP32], cb: pld.CommCtx):\n"
        if mode == "swap"
        else "    def window(self, data: pld.DistributedTensor[[4], pl.FP32]):\n"
        if mode in {"metadata_for", "if_metadata"}
        else "    def observe(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],\n"
        "                data: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx):\n"
    )
    program = pl.parse_program(
        "@pl.program\nclass ControlFlow:\n"
        "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n" + helper + "        return 0\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        "    def node(self, x: pl.Tensor[[4], pl.FP32], y: pl.Tensor[[4], pl.FP32],\n"
        "             seed: pl.Scalar[pl.INT32], other: pl.Scalar[pl.INT32], flag: pl.Scalar[pl.BOOL],\n"
        "             data: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx):\n"
        + bodies[mode]
        + "        return 0\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        "    def root(self, x: pl.Tensor[[4], pl.FP32], y: pl.Tensor[[4], pl.FP32],\n"
        "             seed: pl.Scalar[pl.INT32], other: pl.Scalar[pl.INT32], flag: pl.Scalar[pl.BOOL]):\n"
        "        storage = pld.alloc_window_buffer(16)\n"
        "        data = pld.window(storage, [4], dtype=pl.FP32)\n"
        "        explicit = pld.system.get_comm_ctx(data)\n"
        "        self.node(x, y, seed, other, flag, data, explicit, worker=0)\n"
        "        return 0\n"
    )
    return _c3b_finish_dt_fixture(program)


@pytest.mark.parametrize(
    "mode", ["scalar", "zero", "nested", "swap", "metadata_for", "if_mixed", "if_metadata", "if_ctx"]
)
def test_c3b_embedded_control_flow_preserves_live_ssa_and_metadata_positions(mode):
    program = _c3b_control_flow_program(mode)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "embedded_control_flow.py", "exec")
    layout = _c3b_embedded_layout(source)
    x, y = (torch.zeros((4,), dtype=torch.float32).share_memory_() for _ in range(2))
    assert x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr()
    window, token = object(), object()
    ctx_token = object() if mode in {"swap", "if_ctx"} else token

    class TaskArgs:
        def __init__(self):
            self.tensors, self.scalars = [], []

        def add_tensor(self, value, tag):
            self.tensors.append((value, tag))

        def add_scalar(self, value):
            self.scalars.append(value)

    class Domain:
        def __init__(self, device_ctx):
            self.device_ctx = device_ctx

        def __getitem__(self, worker):
            assert worker == 1
            return SimpleNamespace(
                device_ctx=self.device_ctx,
                domain_size=12,
                domain_rank=9,
                buffers={
                    row[4]: SimpleNamespace(tensor=lambda **kwargs: window) for row in layout["window_slots"]
                }
                if self.device_ctx == 12001
                else {},
            )

    domain = Domain(12001)
    ctx_domain = Domain(12002) if mode in {"swap", "if_ctx"} else domain
    tensors = {"x__ssa_v0": x, "y__ssa_v0": y, "seed__ssa_v0": 5, "other__ssa_v0": 9}
    tensors.update(
        {
            row[1]: ctx_domain if row[0] == 6 else domain
            for row in layout["scalar_slots"]
            if row[3] == "comm_ctx"
        }
    )

    def ordering_tokens(handles, rank):
        assert rank == 1
        by_domain = {id(domain): token, id(ctx_domain): ctx_token}
        seen, selected = set(), []
        for handle in handles:
            if id(handle) not in seen:
                seen.add(id(handle))
                selected.append(by_domain[id(handle)])
        return tuple(selected)

    frame = support._EmbeddedFrame(tensors, object())
    frame.ordering_tokens = ordering_tokens
    calls = []
    namespace = {
        "TaskArgs": TaskArgs,
        "TensorArgType": SimpleNamespace(INPUT="input", OUTPUT_EXISTING="existing", INOUT="inout"),
        "DataType": SimpleNamespace(FLOAT32="fp32"),
        "make_tensor_arg": lambda worker, value: value,
        "add_chip_tensor_arg": lambda args, worker, value, tag, *rest: args.add_tensor(value, tag),
        "_submit_chip": lambda orch, handle, args, config, worker: calls.append(args),
    }
    exec(compile(_c3b_api_module(source), "control_flow_recording.py", "exec"), namespace)
    state = SimpleNamespace(
        begin_invocation=lambda *args: frame,
        callables={name: object() for name in ("scalar", "mixed", "observe", "window")},
        sub_ids={},
        world_size=2,
        domain_provider=None,
    )
    namespace["node"]._pypto_embedded_state = state
    for flag in (True, False) if mode.startswith("if_") else (True,):
        tensors["flag__ssa_v0"] = flag
        calls.clear()
        namespace["node"](SimpleNamespace(_worker=object()), object(), object())
        if mode in {"scalar", "zero", "nested"}:
            expected = {"scalar": [5, 6, 7], "zero": [5], "nested": [5, 6, 7, 8, 9]}[mode]
            assert [args.scalars for args in calls] == [[value] for value in expected]
            assert all(not args.tensors for args in calls)
        elif mode == "swap":
            assert len(calls) == 3
            for args, (left, p, right, q) in zip(calls, ((x, 5, y, 9), (y, 9, x, 5), (x, 5, y, 9))):
                assert args.tensors == [
                    (left, "input"),
                    (window, "input"),
                    (right, "input"),
                    (window, "input"),
                    (token, "inout"),
                    (ctx_token, "inout"),
                ]
                assert ctx_token is not token
                assert args.scalars == [p, 12002, q, 12002, 12001, 12001]
        elif mode == "if_ctx":
            assert len(calls) == 1
            assert calls[0].tensors == [
                (x, "input"),
                (window, "input"),
                (token, "inout"),
                (ctx_token, "inout"),
            ]
            assert ctx_token is not token
            assert calls[0].scalars == [5, 12002, 12001]
        elif mode == "if_mixed":
            assert len(calls) == 1
            assert calls[0].tensors == [((x if flag else y), "input"), (window, "input"), (token, "inout")]
            assert calls[0].scalars == [5 if flag else 6, 12001, 12001]
        else:
            assert len(calls) == 1
            assert calls[0].tensors == [(window, "input"), (token, "inout")]
            assert calls[0].scalars == [12001]


@pytest.mark.parametrize("mode", ["ambiguous_ctx_for", "ambiguous_ctx_if"])
def test_c3b_embedded_control_flow_rejects_unresolved_ctx_before_dispatch(mode):
    program = _c3b_control_flow_program(mode)
    with pytest.raises(ValueError, match=r"(?i)embedded.*CommCtx.*(formal|query|carry|phi|ambiguous)"):
        codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")


def _c3c_emit_entry_metadata(program, root, *, local_device_count=2):
    """Persist actual native HOST metadata without asserting CHIP compilation.

    C3b fixtures intentionally keep Opaque CHIP callees to test native HOST
    semantics. The recursive backend tests separately use real Orchestration
    CHIP functions; do not silently retag these fixtures for PTO generation.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        files = {"orchestration/cluster_orch.py": codegen.DistributedCodegen().generate(program)}
        for function in program.functions.values():
            if function.level == ir.Level.HOST and function.role == ir.Role.Orchestrator:
                files[f"next_levels/{function.name}/orchestration/host_orch.py"] = (
                    codegen.DistributedCodegen()._generate_embedded(program, entry_name=function.name)
                )
    finally:
        backend.reset_for_testing()
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    config = ir.ClusterConfig(
        [
            ir.ClusterNodeSpec("node-a", list(range(local_device_count))),
            ir.ClusterNodeSpec("node-b", list(range(local_device_count, local_device_count * 2))),
        ],
        "/env/bin/python",
    )
    ir.DistributedCompiledProgram(program, str(root), cluster_config=config)
    return json.loads((root / "distributed_meta.json").read_text())


def test_c3c_cold_window_geometry_is_versioned_beside_unchanged_native_layout(tmp_path):
    # Keep the actual lowered offsets/strides fixture: neither the eight-field
    # v1 row nor a packed shape is sufficient to reconstruct these descriptors.
    program = _c3b_dt_geometry_program()
    native = _c3b_embedded_layout(codegen.DistributedCodegen()._generate_embedded(program, entry_name="node"))
    metadata = _c3c_emit_entry_metadata(program, tmp_path)
    entry = metadata["l3_entries"]["entries"][0]
    assert entry["entry_name"] == "node"
    assert entry["layout"] == json.loads(json.dumps(native))
    assert native["version"] == 1 and all(len(row) == 8 for row in native["window_slots"])
    assert native["tensor_slots"] == () and len(native["scalar_slots"]) == 2
    assert [row[0] for row in native["window_slots"]] == [0, 1]
    assert [row[2:4] for row in native["window_slots"]] == [(2, 0), (3, 1)]
    assert entry["view_geometry"] == {
        "version": 1,
        "window_slots": [
            {"formal_index": 0, "strides": [8, 1], "byte_offset": 36},
            {"formal_index": 1, "strides": [8, 1], "byte_offset": 68},
        ],
    }
    # Prefix CommCtx and DT suffix positions remain explicit in layout, and
    # restoring this record does not require the original native Program.
    restored = ir.DistributedCompiledProgram.from_dir(tmp_path)
    assert restored.program is None and restored.entry_kind == "cluster"


@pytest.mark.parametrize("mutation", ["missing_view", "duplicate_view", "bool_offset", "bad_stride"])
def test_c3c_cold_window_geometry_rejects_incomplete_or_malformed_views(tmp_path, mutation):
    metadata = _c3c_emit_entry_metadata(_c3b_dt_geometry_program(), tmp_path)
    views = metadata["l3_entries"]["entries"][0]["view_geometry"]["window_slots"]
    if mutation == "missing_view":
        views.pop()
    elif mutation == "duplicate_view":
        views[1]["formal_index"] = views[0]["formal_index"]
    elif mutation == "bool_offset":
        views[0]["byte_offset"] = True
    else:
        views[0]["strides"] = [8]
    (tmp_path / "distributed_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"distributed_meta\.json.*(view|geometry|l3_entries)"):
        ir.DistributedCompiledProgram.from_dir(tmp_path)


def test_c3c_cold_metadata_keeps_real_prefork_return_copy_and_storage_requirements(tmp_path):
    program = _c3b_tensor_return_program("internal")
    native = _c3b_embedded_layout(codegen.DistributedCodegen()._generate_embedded(program, entry_name="node"))
    metadata = _c3c_emit_entry_metadata(program, tmp_path)
    entry = metadata["l3_entries"]["entries"][0]
    assert entry["layout"] == json.loads(json.dumps(native))
    requirements = entry["layout"]["prefork_requirements"]
    assert any(row[0] == "tensor_storage" for row in requirements)
    assert any(row[0] == "return_copy_subworker" for row in requirements)
    assert entry["return_copy_callable"] == {
        "module": "orchestration._distributed_support",
        "qualname": "_copy_return_tensor",
    }
    assert entry["minimum_sub_workers"] >= 1
    assert entry["layout"]["implicit_outputs"]


@pytest.mark.parametrize("mode", ["distinct", "local"])
def test_c3c_cold_metadata_keeps_original_ctx_prefix_dt_suffix_and_domain_tokens(tmp_path, mode):
    program = _c3b_domain_program(mode)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    native = _c3b_embedded_layout(source)
    metadata = _c3c_emit_entry_metadata(program, tmp_path, local_device_count=4)
    entry = metadata["l3_entries"]["entries"][0]
    assert entry["layout"] == json.loads(json.dumps(native))
    assert [row[0] for row in native["scalar_slots"]] == [1, 5, 6, 7]
    assert [row[2:4] for row in native["window_slots"]] == [(6, 2), (7, 3)]
    assert ("inherited_domain_tokens", 3) in native["prefork_requirements"]
    assert entry["local_device_count"] == 4
    if mode == "local":
        assert any(row[0] == "local_domain_tokens" for row in native["prefork_requirements"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
