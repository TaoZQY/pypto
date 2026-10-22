# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for pl.parse() and pl.loads() text parsing functions."""

import ast
import os
import tempfile

import pypto
import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics import ParserError, ParserTypeError
from pypto.language.parser.diagnostics.renderer import ErrorRenderer


def _span_begin(err: ParserError) -> tuple[int, int]:
    """Return (begin_line, begin_column) from a ParserError span.

    ``ParserError.span`` is stored as a dict of extracted coordinates (see
    ``diagnostics/exceptions.py``). Both coordinates are 1-indexed; use
    :func:`_span_char_index` to slice a source line with the column.
    """
    sp = err.span
    assert sp is not None, "expected a span on the parser error"
    begin_line, begin_column = sp["begin_line"], sp["begin_column"]
    assert isinstance(begin_line, int) and isinstance(begin_column, int)
    return begin_line, begin_column


def _span_char_index(begin_column: int) -> int:
    """Convert a 1-indexed span column to a 0-based index into the source line."""
    return begin_column - 1


def _assert_caret_on_line(err: ParserError, expected_substring: str) -> None:
    """Assert the rendered caret (^) sits under a line containing ``expected_substring``.

    The renderer prints each source line immediately before its caret row, so
    the line above the caret is the one the caret points at.
    """
    rendered = ErrorRenderer(use_color=False).render(err)
    rows = rendered.split("\n")
    caret_idx = next(i for i, r in enumerate(rows) if "^" in r)
    source_row = rows[caret_idx - 1]
    assert expected_substring in source_row, (
        f"caret points at {source_row!r}, expected a line containing "
        f"{expected_substring!r}\n--- full render ---\n{rendered}"
    )


class TestParse:
    """Tests for pl.parse() function."""

    def test_parse_simple_function_with_import(self):
        """Test parsing simple function with import statement."""
        code = """
import pypto.language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_one"
        assert len(func.params) == 1
        assert len(func.return_types) == 1

    def test_parse_simple_function_without_import(self):
        """Test parsing simple function without import statement (auto-injected)."""
        code = """
@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_one"
        assert len(func.params) == 1
        assert len(func.return_types) == 1

    def test_parse_multiple_params(self):
        """Test parsing function with multiple parameters."""
        code = """
@pl.function
def add_three(
    x: pl.Tensor[[64], pl.FP32],
    y: pl.Tensor[[64], pl.FP32],
    z: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    temp: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
    result: pl.Tensor[[64], pl.FP32] = pl.add(temp, z)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_three"
        assert len(func.params) == 3

    def test_parse_with_for_loop(self):
        """Test parsing function with for loop control flow."""
        code = """
@pl.function
def sum_loop(x: pl.Tensor[[10], pl.FP32]) -> pl.Tensor[[10], pl.FP32]:
    init_sum: pl.Tensor[[10], pl.FP32] = pl.create_tensor([10], dtype=pl.FP32)
    for i, (running_sum,) in pl.range(5, init_values=(init_sum,)):
        new_sum: pl.Tensor[[10], pl.FP32] = pl.add(running_sum, x)
        result = pl.yield_(new_sum)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "sum_loop"

    def test_parse_with_multiple_statements(self):
        """Test parsing function with multiple statements."""
        code = """
@pl.function
def multi_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    a: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    b: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
    c: pl.Tensor[[64], pl.FP32] = pl.sub(b, 0.5)
    return c
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "multi_op"

    def test_parse_no_function_error(self):
        """Test that parsing code with no function raises ValueError."""
        code = """
x = 42
y = x + 1
"""
        with pytest.raises(ValueError, match="No @pl.function or @pl.program found"):
            pl.parse(code)

    def test_parse_multiple_functions_error(self):
        """Test that parsing code with multiple functions raises ValueError."""
        code = """
@pl.function
def func1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x

@pl.function
def func2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ValueError, match="Multiple functions/programs found"):
            pl.parse(code)

    def test_parse_syntax_error(self):
        """Test that parsing code with syntax error raises SyntaxError."""
        code = """
@pl.function
def bad_syntax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x +
"""
        with pytest.raises(SyntaxError, match="Failed to compile code"):
            pl.parse(code)

    def test_parse_with_custom_filename(self):
        """Test that custom filename is used in error reporting."""
        code = """
@pl.function
def bad_func(x):
    return x
"""
        with pytest.raises(pl.parser.ParserError):
            pl.parse(code, filename="custom_file.py")

    def test_parse_from_import_variant(self):
        """Test parsing with 'from pypto import language as pl' variant."""
        code = """
from pypto import language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_one"

    def test_parse_with_different_dtypes(self):
        """Test parsing function with various data types."""
        code = """
@pl.function
def cast_op(
    fp16: pl.Tensor[[64], pl.FP16],
    fp32: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(
        pl.cast(fp16, target_type=pl.FP32), fp32
    )
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "cast_op"
        assert len(func.params) == 2


class TestLoad:
    """Tests for pl.loads() function."""

    def test_load_simple_function(self):
        """Test loading function from a file."""
        code = """
import pypto.language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.loads(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "add_one"
            assert len(func.params) == 1
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_load_function_without_import(self):
        """Test loading function without import (auto-injected)."""
        code = """
@pl.function
def multiply(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
    result: pl.Tensor[[32, 32], pl.FP32] = pl.mul(x, 2.0)
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.loads(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "multiply"
        finally:
            os.unlink(temp_path)

    def test_load_file_not_found(self):
        """Test that loading non-existent file raises OSError."""
        with pytest.raises(OSError):
            pl.loads("/non/existent/path/file.py")

    def test_load_complex_function(self):
        """Test loading a complex function with control flow."""
        code = """
import pypto.language as pl

@pl.function
def complex_op(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP32]:
    # Multiple operations
    temp1: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
    temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, 2.0)
    result: pl.Tensor[[64, 128], pl.FP32] = pl.sub(temp2, x)
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.loads(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "complex_op"
            assert len(func.params) == 2
        finally:
            os.unlink(temp_path)

    def test_load_preserves_filename_in_errors(self):
        """Test that errors reference the correct file path."""
        code = """
@pl.function
def bad_func(x):
    return x
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        with pytest.raises(pl.parser.ParserError):
            pl.loads(temp_path)
        os.unlink(temp_path)


class TestIntegration:
    """Integration tests for parse/load with existing decorator."""

    def test_decorator_and_parse_produce_same_result(self):
        """Test that @pl.function decorator and pl.parse produce equivalent results."""

        # Using decorator
        @pl.function
        def func_decorator(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        # Using parse
        code = """
@pl.function
def func_parse(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func_parsed = pl.parse(code)

        # Both should be ir.Function objects with same structure
        assert isinstance(func_decorator, ir.Function)
        assert isinstance(func_parsed, ir.Function)
        assert len(func_decorator.params) == len(func_parsed.params)
        assert len(func_decorator.return_types) == len(func_parsed.return_types)

    def test_serialization_of_parsed_function(self):
        """Test that parsed functions can be serialized and deserialized."""
        code = """
@pl.function
def serializable(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        func = pl.parse(code)

        # Should be able to serialize
        data = pypto.ir.serialize(func)
        assert len(data) > 0

        # Should be able to deserialize
        restored = pypto.ir.deserialize(data)
        assert isinstance(restored, ir.Function)
        assert restored.name == "serializable"


class TestParseProgram:
    """Tests for pl.parse_program() function."""

    def test_parse_simple_program_with_import(self):
        """Test parsing simple program with import statement."""
        code = """
import pypto.language as pl

@pl.program
class SimpleProgram:
    @pl.function
    def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert program.name == "SimpleProgram"
        assert len(program.functions) == 1

    def test_parse_simple_program_without_import(self):
        """Test parsing simple program without import statement (auto-injected)."""
        code = """
@pl.program
class SimpleProgram:
    @pl.function
    def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert program.name == "SimpleProgram"
        assert len(program.functions) == 1

    def test_parse_program_with_multiple_functions(self):
        """Test parsing program with multiple functions."""
        code = """
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
        return result

    @pl.function
    def cube(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        x_sq: pl.Tensor[[1], pl.INT32] = self.square(x)
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x_sq)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert program.name == "MathOps"
        assert len(program.functions) == 2

    def test_parse_program_with_cross_function_calls(self):
        """Test parsing program with cross-function calls."""
        code = """
@pl.program
class CallTest:
    @pl.function
    def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, 2)
        return result

    @pl.function
    def caller(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = self.helper(x)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert len(program.functions) == 2

        # Verify self parameter was stripped
        caller_func = program.get_function("caller")
        assert caller_func is not None
        assert len(caller_func.params) == 1
        assert caller_func.params[0].name_hint == "x"

    def test_parse_program_no_program_error(self):
        """Test that code without @pl.program raises error."""
        code = """
@pl.function
def standalone(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ValueError, match="Expected @pl.program but found @pl.function"):
            pl.parse_program(code)

    def test_parse_program_multiple_programs_error(self):
        """Test that multiple @pl.program classes raises error."""
        code = """
@pl.program
class Program1:
    @pl.function
    def func1(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        return x

@pl.program
class Program2:
    @pl.function
    def func2(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        return x
"""
        with pytest.raises(ValueError, match="Multiple functions/programs found"):
            pl.parse_program(code)

    def test_parse_program_syntax_error(self):
        """Test that syntax errors are properly reported."""
        code = """
@pl.program
class BadSyntax:
    @pl.function
    def broken(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        return x +
"""
        with pytest.raises(SyntaxError):
            pl.parse_program(code)


class TestLoadProgram:
    """Tests for pl.loads_program() function."""

    def test_load_program_from_file(self):
        """Test loading program from a file."""

        code = """
import pypto.language as pl

@pl.program
class FileProgram:
    @pl.function
    def add(self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
        return result
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            program = pl.loads_program(temp_path)
            assert isinstance(program, ir.Program)
            assert program.name == "FileProgram"
            assert len(program.functions) == 1
        finally:
            os.unlink(temp_path)

    def test_load_program_file_not_found(self):
        """Test that load_program raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            pl.loads_program("nonexistent_file.py")

    def test_parse_function_with_scalar_param(self):
        """Test parsing function with scalar parameter from text."""
        code = """
@pl.function
def add_scalar(x: pl.Tensor[[64], pl.FP32], scalar: pl.Scalar[pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, scalar)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert len(func.params) == 2
        assert isinstance(func.params[1].type, ir.ScalarType)
        assert func.params[1].type.dtype == pl.FP32


class TestExecErrorDiagnostics:
    """Tests for exec-time error diagnostics in parse()."""

    def test_exec_runtime_error_becomes_parser_syntax_error(self):
        """Verify that a runtime error during exec() produces a ParserSyntaxError with span."""
        from pypto.language.parser.diagnostics import ParserSyntaxError  # noqa: PLC0415

        # pl.FunctionType exists but FunctionType.BadType does not — caught by pre-validator
        code = """
@pl.function(type=pl.FunctionType.BadType)
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ParserSyntaxError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        assert err.span is not None
        assert "BadType" in err.message or "FunctionType" in err.message

    def test_exec_runtime_error_span_has_correct_filename(self):
        """Verify that ParserSyntaxError from exec error has the expected filename in span."""
        from pypto.language.parser.diagnostics import ParserSyntaxError  # noqa: PLC0415

        code = """
@pl.function(type=pl.FunctionType.BadType)
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ParserSyntaxError) as exc_info:
            pl.parse(code, filename="test_file.py")
        err = exc_info.value
        assert err.span is not None
        assert err.span["filename"] == "test_file.py"

    def test_exec_runtime_error_includes_source_lines(self):
        """Verify that ParserSyntaxError from exec error includes source lines for context."""
        from pypto.language.parser.diagnostics import ParserSyntaxError  # noqa: PLC0415

        code = """
@pl.function(type=pl.FunctionType.BadType)
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ParserSyntaxError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        assert err.source_lines is not None
        assert len(err.source_lines) > 0

    def test_exec_error_column_points_to_bad_attribute(self):
        """Column in span points to the start of the bad attribute (not column 0)."""
        from pypto.language.parser.diagnostics import ParserSyntaxError  # noqa: PLC0415

        code = """
@pl.function(type=pl.FunctionType.BadType)
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ParserSyntaxError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        assert err.span is not None
        span = err.span
        column = span["column"]
        assert isinstance(column, int) and column > 0  # Not column 0 — points at 'BadType'

    def test_exec_error_column_points_at_non_ascii_attribute(self):
        """The bad-attribute column must survive a non-ASCII identifier.

        ``ast`` reports ``end_col_offset`` in UTF-8 bytes, so the attribute's
        *byte* length is what locates its start. Subtracting the character
        length instead lands inside the identifier whenever it contains
        non-ASCII text, which Python permits in identifiers.
        """
        from pypto.language.parser.diagnostics import ParserSyntaxError  # noqa: PLC0415

        code = """
@pl.function(type=pl.FunctionType.Bad_é_Type)
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ParserSyntaxError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        assert err.span is not None
        span = err.span
        begin_line = span["begin_line"]
        begin_column = span["begin_column"]
        assert isinstance(begin_line, int) and isinstance(begin_column, int)
        line = code.split("\n")[begin_line - 1]
        assert line[begin_column - 1 :].startswith("Bad_é_Type"), (
            f"column {begin_column} does not land on the attribute in {line!r}"
        )

    def test_exec_error_hint_lists_valid_values(self):
        """Hint message lists the valid enum values."""
        from pypto.language.parser.diagnostics import ParserSyntaxError  # noqa: PLC0415

        code = """
@pl.function(type=pl.FunctionType.BadType)
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ParserSyntaxError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        assert err.hint is not None
        assert "AIC" in err.hint  # Valid value should be listed
        assert "AIV" in err.hint


class TestScalarRangeRoundTrip:
    """Tests for round-trip (print -> parse) of pl.range() with Scalar arguments."""

    def test_scalar_stop_roundtrip(self):
        """Test round-trip: pl.range(n) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = Before.as_python()
        # Verify the printed output contains the Scalar parameter in range
        assert "pl.Scalar[pl.INT64]" in printed
        assert "pl.range(" in printed
        assert "n" in printed

        # Re-parse the printed output
        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_start_stop_step_roundtrip(self):
        """Test round-trip: pl.range(0, n, s) where n, s are Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                n: pl.Scalar[pl.INT64],
                s: pl.Scalar[pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, n, s):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = Before.as_python()
        assert "pl.range(" in printed
        assert "n" in printed
        assert "s" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_expression_roundtrip(self):
        """Test round-trip: pl.range(n * 2) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n * 2):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = Before.as_python()
        assert "n * 2" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_range_with_iter_args_roundtrip(self):
        """Test round-trip: pl.range(n, init_values=(...)) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(n, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        printed = Before.as_python()
        assert "pl.range(" in printed
        assert "init_values=" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_parallel_roundtrip(self):
        """Test round-trip: pl.parallel(n) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.parallel(n):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = Before.as_python()
        assert "pl.parallel(" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_range_text_parse(self):
        """Test parsing Scalar range directly from text string."""
        code = """
@pl.function
def scalar_range_func(
    n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.range(n):
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return y
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "scalar_range_func"
        assert len(func.params) == 2
        assert isinstance(func.params[0].type, ir.ScalarType)

    def test_scalar_range_complex_expression_text_parse(self):
        """Test parsing Scalar range with complex expression from text string."""
        code = """
@pl.function
def complex_range(
    n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.range(0, n * 2 + 1, 1):
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return y
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "complex_range"


class TestErrorCaretAlignment:
    """Diagnostic caret must align with the span when parsing ``<string>`` sources.

    ``pl.parse`` (and therefore ``@pl.jit``, which renders specialized DSL source
    and parses it as a ``<string>``) yields spans in module/file coordinates.
    ``error.source_lines`` must be indexed the same way or the rendered caret
    drifts by the entity's ``line_offset`` — pointing several lines past the real
    error. Regression for issue #1558.
    """

    def test_parse_error_caret_aligns_with_span_deps_shape(self):
        """A rejected ``pl.at(deps=...)`` shape points the caret at the ``deps=``
        argument on the ``with pl.at(...)`` line, not lines past it (issue #1558).

        The ``with pl.at(...)`` is written on a single collapsed line the way
        ``@pl.jit``'s ``ast.unparse`` emits it.
        """
        code = """
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[128], pl.FP16]) -> pl.Tensor[[128], pl.FP32]:
        out = pl.create_tensor([128], dtype=pl.FP32)
        with pl.manual_scope():
            silu_tids = pl.array.create(8, pl.TASK_ID)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='seed') as seed_tid:
                tmp = pl.create_tensor([128], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='proj', deps=[seed_tid] + [silu_tids[i] for i in range(4)]) as down_tid:
                a0 = pl.create_tensor([128], dtype=pl.FP32)
        return out
"""
        with pytest.raises(ParserError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        begin_line, begin_column = _span_begin(err)

        # Module indexing: the span's line must resolve to the with pl.at line.
        source_lines = err.source_lines
        assert source_lines is not None
        error_line = source_lines[begin_line - 1]
        assert "deps=" in error_line
        # The column must point exactly at the rejected `[seed_tid] + ...` value.
        assert error_line[_span_char_index(begin_column) :].startswith("[seed_tid]")

        # The rendered caret must sit under that same line.
        _assert_caret_on_line(err, expected_substring="deps=")

    def test_parse_error_caret_aligns_with_span_generic(self):
        """A generic parse error in a ``<string>`` program points the caret at the
        real offending line (guards the general renderer alignment bug)."""
        code = """
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[128], pl.FP16]) -> pl.Tensor[[128], pl.FP32]:
        a = pl.create_tensor([128], dtype=pl.FP32)
        b = pl.create_tensor([128], dtype=pl.FP32)
        c = pl.this_op_does_not_exist(a, b)
        return c
"""
        with pytest.raises(ParserError) as exc_info:
            pl.parse(code)
        err = exc_info.value
        begin_line, begin_column = _span_begin(err)

        source_lines = err.source_lines
        assert source_lines is not None
        error_line = source_lines[begin_line - 1]
        assert "this_op_does_not_exist" in error_line
        assert error_line[_span_char_index(begin_column) :].startswith("pl.this_op_does_not_exist")

        _assert_caret_on_line(err, expected_substring="this_op_does_not_exist")


@pytest.mark.parametrize("extent", ["8", "N"])
@pytest.mark.parametrize("axis", [0, -2])
@pytest.mark.parametrize("arguments", ["x, {axis}", "x, axis={axis}", "tensor=x, axis={axis}"])
def test_tensor_dim_parameter_shape_roundtrips(extent, axis, arguments):
    """Dependent annotations are parsed as IR, never evaluated as Python."""
    dim_arguments = arguments.replace("{axis}", f"pl.const({axis}, pl.INDEX)")
    source = f"""
N = pl.dynamic("N")

@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[{extent}, 32], pl.FP32],
             view: pl.Tensor[[pl.tensor.dim({dim_arguments}) * 2, 16], pl.FP32]):
        return view
"""
    before = pl.parse(source)
    printed = ir.python_print(before)
    assert "pl.tensor.dim(" in printed
    after = pl.parse(printed)
    ir.assert_structural_equal(before, after)


@pytest.mark.parametrize(
    "setup, extent",
    [
        ("", "8"),
        ("", "rows"),
        ("n = pl.tensor.dim(x, 0)", "n * 3"),
        ("n = pl.tensor.dim(x, 0)\n    n = rows", "n * 2"),
    ],
)
def test_tensor_dim_annotation_keeps_unproven_inferred_shape(setup, extent):
    """An annotation cannot replace a different or unproven runtime extent."""
    source = f"""
@pl.function(type=pl.FunctionType.InCore)
def main(x: pl.Tensor[[8, 16], pl.FP32], rows: pl.Scalar[pl.INDEX]):
    {setup}
    view: pl.Tensor[[pl.tensor.dim(x, 0) * 2, 16], pl.FP32] = pl.reshape(x, [{extent}, 16])
    return view
"""
    func = pl.parse(source)
    assert isinstance(func, ir.Function)
    assert isinstance(func.body, ir.SeqStmts)
    assign = next(s for s in func.body.stmts if isinstance(s, ir.AssignStmt) and s.var.name_hint == "view")
    ir.assert_structural_equal(assign.var.type, assign.value.type)


@pytest.mark.parametrize("extent", ["8", "N"])
def test_tensor_dim_annotation_recovers_flattened_read(extent):
    """A direct dimension temporary proves the declarative shape unchanged."""
    source = f"""
N = pl.dynamic("N")
@pl.function(type=pl.FunctionType.InCore)
def main(x: pl.Tensor[[{extent}, 16], pl.FP32]):
    n = pl.tensor.dim(x, 0)
    view: pl.Tensor[[pl.tensor.dim(x, 0) * 2, 16], pl.FP32] = pl.reshape(x, [n * 2, 16])
    return view
"""
    before = pl.parse(source)
    printed = ir.python_print(before)
    assert "view: pl.Tensor[[pl.tensor.dim(" in printed
    ir.assert_structural_equal(before, pl.parse('N = pl.dynamic("N")\n' + printed))


def _window_identity_source(order=("chip", "host", "owner"), *, key='"shared"', size="size"):
    """Build explicit dump text without depending on the materialization pass.

    The inherited types deliberately retain different local layouts/storage.
    The owner descriptor alone binds its base and dynamic byte size.
    """
    child_type = (
        "pld.DistributedTensor[[2, 3], pl.FP16, "
        "pl.TensorView(stride=[1, 2], layout=pl.TensorLayout.ND), "
        f'pl.MemRef("buf", 8, 12), pl._dump.window_ref({key})]'
    )
    owner_type = f"pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref({key})]"
    methods = {}
    for name, level in (("chip", "CHIP"), ("host", "HOST")):
        methods[name] = f"""
    @pl.function(level=pl.Level.{level}, role=pl.Role.Orchestrator)
    def {name}(self, x: {child_type}, n: pl.Scalar[pl.INDEX]) -> {child_type}:
        size = n * 8
        return x
"""
    methods["owner"] = f"""
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def owner(self, n: pl.Scalar[pl.INDEX]):
        with pl._dump.comm_domain(name="global", devices=[], slots=[
            pl._dump.window_buffer({key}, base="buf", size={size},
                                   load_from_host=True, store_to_host=True)
        ]):
            size = n * 4
            buf = pld.alloc_window_buffer(size)
            data: {owner_type} = pld.window(buf, [4], dtype=pl.FP32)
            return data
"""
    return "@pl.program\nclass ExplicitWindowIdentity:\n" + "\n".join(methods[name] for name in order)


def _window_function(program, name):
    return program.functions[program.get_global_var(name)]


def _window_statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _window_statements(child)
    elif isinstance(stmt, ir.ScopeStmt):
        yield from _window_statements(stmt.body)


def _window_owner(program):
    owner = _window_function(program, "owner")
    scope = next(s for s in _window_statements(owner.body) if isinstance(s, ir.CommDomainScopeStmt))
    return owner, scope, scope.slots[0]


@pytest.mark.parametrize(
    "order", [("chip", "host", "owner"), ("owner", "host", "chip"), ("host", "owner", "chip")]
)
@pytest.mark.parametrize("key", ['"shared"', "7"])
def test_program_window_identity_across_function_order(order, key):
    """Forward references and three levels share only the explicit window ID."""
    program = pl.parse_program(_window_identity_source(order, key=key))
    _, scope, slot = _window_owner(program)
    assert slot.load_from_host is True
    assert slot.store_to_host is True
    for name in ("chip", "host"):
        func = _window_function(program, name)
        assert func.params[0].type.window_buffer is slot
        assert func.return_types[0].window_buffer is slot
        returned = next(s for s in _window_statements(func.body) if isinstance(s, ir.ReturnStmt))
        assert returned.value[0] is func.params[0]
        assert returned.value[0].type.window_buffer is slot
        assert not any(isinstance(s, ir.CommDomainScopeStmt) for s in _window_statements(func.body))
    data = next(
        s
        for s in _window_statements(scope.body)
        if isinstance(s, ir.AssignStmt) and s.var.name_hint == "data"
    )
    assert data.var.type.window_buffer is slot


@pytest.mark.parametrize("size", ["size", "n * 4"])
def test_program_window_identity_uses_owner_local_base_and_dynamic_size(size):
    """A child's same-spelling locals must never resolve owner metadata."""
    program = pl.parse_program(_window_identity_source(size=size))
    owner, scope, slot = _window_owner(program)
    assignments = {s.var.name_hint: s for s in _window_statements(scope.body) if isinstance(s, ir.AssignStmt)}
    assert slot.base is assignments["buf"].var
    assert assignments["buf"].value.args[0] is assignments["size"].var
    owner_size = assignments["size"].value
    assert isinstance(owner_size, ir.Mul)
    assert owner_size.left is owner.params[0]
    assert owner_size.right.value == 4
    if size == "size":
        assert slot.size is assignments["size"].var
    else:
        assert isinstance(slot.size, ir.Mul)
        assert slot.size.left is owner.params[0]
        assert slot.size.right.value == 4
    for name in ("chip", "host"):
        child = _window_function(program, name)
        assert child.params[1] is not owner.params[0]
        child_size = next(s.var for s in _window_statements(child.body) if isinstance(s, ir.AssignStmt))
        assert slot.size is not child_size


def test_program_window_identity_preserves_function_local_storage_and_views():
    """Identical base names grant no cross-function MemRef allocation identity."""
    program = pl.parse_program(_window_identity_source())
    _, _, slot = _window_owner(program)
    child_types = [_window_function(program, name).params[0].type for name in ("chip", "host")]
    for ty in child_types:
        assert ty.window_buffer is slot
        assert [dim.value for dim in ty.shape] == [2, 3]
        assert ty.dtype == pl.FP16
        assert ty.memref.byte_offset_.value == 8
        assert ty.memref.size_ == 12
        assert ty.memref.base_.name_hint == "buf"
        assert ty.memref.base_ is not slot.base
        assert [stride.value for stride in ty.tensor_view.stride] == [1, 2]
        assert ty.tensor_view.layout == ir.TensorLayout.ND
    assert child_types[0].memref is not child_types[1].memref
    assert child_types[0].memref.base_ is not child_types[1].memref.base_


@pytest.mark.parametrize("kind", ["call", "submit"])
def test_program_window_identity_rebinds_call_submit_and_result_types(kind):
    """Deferred windows in signatures also reach launch results and projections."""
    ty = 'pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("shared")]'
    dispatch = (
        "result = self.chip(x, worker=0)"
        if kind == "call"
        else "result, tid = pl.submit(self.chip, x, worker=0, deps=[dep])"
    )
    source = f"""
@pl.program
class WindowLaunch:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, x: {ty}) -> {ty}:
        return x
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self, x: {ty}, dep: pl.Scalar[pl.TASK_ID]) -> {ty}:
        with pl.manual_scope():
            {dispatch}
        return result
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def owner(self):
        with pl._dump.comm_domain(name="global", devices=[], slots=[
            pl._dump.window_buffer("shared", base="buf", size=16)
        ]):
            buf = pld.alloc_window_buffer(16)
            data: {ty} = pld.window(buf, [4], dtype=pl.FP32)
            return data
"""
    program = pl.parse_program(source)
    _, _, slot = _window_owner(program)
    host = _window_function(program, "host")
    assignments = [s for s in _window_statements(host.body) if isinstance(s, ir.AssignStmt)]
    launch = next(s.value for s in assignments if isinstance(s.value, (ir.Call, ir.Submit)))
    assert launch.args[0] is host.params[0]
    assert launch.args[0].type.window_buffer is slot
    assert host.return_types[0].window_buffer is slot
    if kind == "call":
        assert launch.type.window_buffer is slot
    else:
        assert isinstance(launch, ir.Submit)
        assert launch.type.types[0].window_buffer is slot
        assert launch.type.types[1].dtype == pl.TASK_ID
        assert len(launch.deps) == 1 and launch.deps[0] is host.params[1]
        projection = next(s for s in assignments if isinstance(s.value, ir.TupleGetItemExpr))
        assert projection.var.type.window_buffer is slot
        assert projection.value.type.window_buffer is slot
    returned = next(s for s in _window_statements(host.body) if isinstance(s, ir.ReturnStmt))
    assert returned.value[0].type.window_buffer is slot


def test_program_window_identity_rejects_unknown_id():
    source = _window_identity_source().replace(
        'pl._dump.window_ref("shared")', 'pl._dump.window_ref("missing")'
    )
    with pytest.raises(ParserTypeError, match="Undefined window_buffer ID.*missing"):
        pl.parse_program(source)


def test_program_window_identity_rejects_conflicting_cross_function_descriptors():
    source = """
@pl.program
class ConflictingWindows:
    @pl.function
    def first(self, x: pld.DistributedTensor[[4], pl.FP32,
        pl._dump.window_buffer("shared", base="first_buf", size=16)]):
        return x
    @pl.function
    def second(self, x: pld.DistributedTensor[[4], pl.FP32,
        pl._dump.window_buffer("shared", base="second_buf", size=32)]):
        return x
"""
    with pytest.raises(ParserTypeError, match="Conflicting window_buffer descriptors.*shared"):
        pl.parse_program(source)


def test_program_window_identity_does_not_leak_between_parses():
    first = pl.parse_program(_window_identity_source())
    second = pl.parse_program(_window_identity_source())
    _, _, first_slot = _window_owner(first)
    _, _, second_slot = _window_owner(second)
    assert first_slot is not second_slot
    assert first_slot.base is not second_slot.base
    assert _window_function(first, "chip").params[0].type.window_buffer is first_slot
    assert _window_function(second, "chip").params[0].type.window_buffer is second_slot
    # A successful previous parse cannot supply a missing owner to this one.
    source = """
@pl.program
class MissingOwner:
    @pl.function
    def child(self, x: pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("shared")]):
        return x
"""
    with pytest.raises(ParserTypeError, match="Undefined window_buffer ID.*shared"):
        pl.parse_program(source)


@pytest.mark.parametrize("level", ["HOST", "CLUSTER_0"])
def test_program_window_identity_ignores_opaque_subworker_body(level):
    """Private dump spellings inside callback Python are opaque source text."""
    source = (
        _window_identity_source(("owner",))
        + f"""
    @pl.function(level=pl.Level.{level}, role=pl.Role.SubWorker)
    def callback(value: pl.Scalar[pl.INT64]):
        ignored = pl._dump.window_buffer("shared", base="not_ir", size=123)
        other = pl._dump.window_buffer("shared", base="also_not_ir", size=456)
        return {{"value": value, "ignored": ignored, "other": other}}
"""
    )
    program = pl.parse_program(source)
    _, _, slot = _window_owner(program)
    assert slot.base.name_hint == "buf"
    callback = _window_function(program, "callback")
    inline = next(s for s in _window_statements(callback.body) if isinstance(s, ir.InlineStmt))
    expected_body = "\n".join(
        [
            'ignored = pl._dump.window_buffer("shared", base="not_ir", size=123)',
            'other = pl._dump.window_buffer("shared", base="also_not_ir", size=456)',
            'return {"value": value, "ignored": ignored, "other": other}',
        ]
    )
    # Source capture uses ast.unparse, whose quote choice is not semantic.
    # Compare all callback statements, literal fields and the return payload.
    assert ast.dump(ast.parse(inline.body)) == ast.dump(ast.parse(expected_body))


def _window_owner_size_query_source(*, owner_first):
    ty_a = 'pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("window_a")]'
    ty_b = 'pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("window_b")]'
    size = "pld.system.rank(pld.system.get_comm_ctx(x)) + 16"
    methods = {
        "child": f"""
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def a_child(self, y: {ty_b}) -> {ty_b}:
        return y
""",
        "owner_b": f"""
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def m_owner_b(self, x: {ty_a}):
        with pl._dump.comm_domain(name="local", devices=[], slots=[
            pl._dump.window_buffer("window_b", base="b_buf", size={size})
        ]):
            b_buf = pld.alloc_window_buffer({size})
            y: {ty_b} = pld.window(b_buf, [4], dtype=pl.FP32)
""",
        "owner_a": f"""
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def z_owner_a(self):
        with pl._dump.comm_domain(name="global", devices=[], slots=[
            pl._dump.window_buffer("window_a", base="a_buf", size=16)
        ]):
            a_buf = pld.alloc_window_buffer(16)
            x: {ty_a} = pld.window(a_buf, [4], dtype=pl.FP32)
""",
    }
    order = ("owner_a", "owner_b", "child") if owner_first else ("child", "owner_b", "owner_a")
    return "@pl.program\nclass OwnerSizeQuery:\n" + "\n".join(methods[name] for name in order)


def _window_size_query_ctx(expr):
    assert isinstance(expr, ir.Add)
    promoted = expr.left
    assert isinstance(promoted, ir.Cast) and promoted.type.dtype == pl.INDEX
    rank = promoted.operand
    assert isinstance(rank, ir.Call) and rank.op.name == ir.get_op("pld.system.rank").name
    assert rank.type.dtype == pl.INT32
    ctx = rank.args[0]
    assert isinstance(ctx, ir.Call) and ctx.op.name == ir.get_op("pld.system.get_comm_ctx").name
    return ctx


@pytest.mark.parametrize("owner_first", [True, False])
def test_program_window_identity_owner_size_query_keeps_rebound_parameter(owner_first):
    """A local owner's size query must bind its rewritten inherited formal."""
    program = pl.parse_program(_window_owner_size_query_source(owner_first=owner_first))
    owner_a = _window_function(program, "z_owner_a")
    owner_b = _window_function(program, "m_owner_b")
    scope_a = next(s for s in _window_statements(owner_a.body) if isinstance(s, ir.CommDomainScopeStmt))
    scope_b = next(s for s in _window_statements(owner_b.body) if isinstance(s, ir.CommDomainScopeStmt))
    slot_a, slot_b = scope_a.slots[0], scope_b.slots[0]
    assert owner_b.params[0].type.window_buffer is slot_a
    assert _window_function(program, "a_child").params[0].type.window_buffer is slot_b
    allocation = next(
        s
        for s in _window_statements(scope_b.body)
        if isinstance(s, ir.AssignStmt) and s.var.name_hint == "b_buf"
    )
    assert slot_b.base is allocation.var
    # Both queries are legal HOST scalar expressions. Verify the complete
    # source premise before distinguishing metadata and body def-use identity.
    properties = passes.IRPropertySet()
    for prop in (passes.IRProperty.TypeChecked, passes.IRProperty.AssignTypeSymmetry):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []

    ctx = _window_size_query_ctx(slot_b.size)
    alloc_ctx = _window_size_query_ctx(allocation.value.args[0])
    assert alloc_ctx.args[0] is owner_b.params[0]
    assert ctx.args[0] is owner_b.params[0]


def test_program_window_identity_manual_descriptor_size_rebinds_without_own_placeholder():
    """An eager manual descriptor still depends on its owner's rewritten Var.

    B has no owning scope and defines its full descriptor in the return
    annotation after x binds. Therefore B is already actual, never a registry
    placeholder, when the later child signature refers to it. A remains forward.
    """
    ty_a = 'pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("window_a")]'
    size = "pld.system.rank(pld.system.get_comm_ctx(x)) + 16"
    ty_b = (
        f'pld.DistributedTensor[[4], pl.FP32, pl._dump.window_buffer("window_b", base="b_buf", size={size})]'
    )
    ref_b = 'pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("window_b")]'
    source = f"""
@pl.program
class ManualOwnerSizeQuery:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def m_owner_b(self, x: {ty_a}) -> {ty_b}:
        b_buf = pld.alloc_window_buffer({size})
        y: {ty_b} = pld.window(b_buf, [4], dtype=pl.FP32)
        return y
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def a_child(self, y: {ref_b}) -> {ref_b}:
        return y
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def z_owner_a(self):
        with pl._dump.comm_domain(name="global", devices=[], slots=[
            pl._dump.window_buffer("window_a", base="a_buf", size=16)
        ]):
            a_buf = pld.alloc_window_buffer(16)
            x: {ty_a} = pld.window(a_buf, [4], dtype=pl.FP32)
"""
    program = pl.parse_program(source)
    owner = _window_function(program, "m_owner_b")
    slot_b = owner.return_types[0].window_buffer
    assert _window_function(program, "a_child").params[0].type.window_buffer is slot_b
    assert not any(isinstance(s, ir.CommDomainScopeStmt) for s in _window_statements(owner.body))
    assignments = {s.var.name_hint: s for s in _window_statements(owner.body) if isinstance(s, ir.AssignStmt)}
    assert assignments["y"].var.type.window_buffer is slot_b
    assert slot_b.base is assignments["b_buf"].var
    properties = passes.IRPropertySet()
    for prop in (passes.IRProperty.TypeChecked, passes.IRProperty.AssignTypeSymmetry):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    alloc_ctx = _window_size_query_ctx(assignments["b_buf"].value.args[0])
    assert alloc_ctx.args[0] is owner.params[0]
    assert _window_size_query_ctx(slot_b.size).args[0] is owner.params[0]


def test_standalone_window_identity_preserves_full_descriptor_annotations():
    """The program registry must not change the standalone annotation surface."""
    source = """
@pl.function
def standalone(x: pld.DistributedTensor[[4], pl.FP32,
    pl._dump.window_buffer(7, base="buf", size=16, load_from_host=True)]
) -> pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref(7)]:
    return x
"""
    func = pl.parse(source)
    assert isinstance(func, ir.Function)
    slot = func.params[0].type.window_buffer
    assert slot is func.return_types[0].window_buffer
    assert slot.base.name_hint == "buf"
    assert slot.size.value == 16
    assert slot.load_from_host is True
    assert slot.store_to_host is False
    returned = next(s for s in _window_statements(func.body) if isinstance(s, ir.ReturnStmt))
    assert returned.value[0] is func.params[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
