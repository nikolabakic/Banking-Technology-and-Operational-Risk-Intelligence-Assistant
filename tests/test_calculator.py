from __future__ import annotations

from decimal import Decimal

import pytest

from bankscope.tools import CALCULATOR_TOOL, CalculatorError, calculate, evaluate_expression
from bankscope.tools.calculator import MAX_EXPRESSION_LENGTH


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("0.1 + 0.2", "0.3"),
        ("-2 ** 2", "-4"),
        ("2 ** -3", "0.125"),
        ("7 // 2", "3"),
        ("7 % 4", "3"),
        ("-7 // 2", "-4"),
        ("-7 % 4", "1"),
    ],
)
def test_calculate_uses_decimal_arithmetic_and_precedence(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


def test_evaluate_expression_exposes_decimal_result() -> None:
    assert evaluate_expression("1 / 8") == Decimal("0.125")


@pytest.mark.parametrize("operator", ["/", "//", "%"])
def test_calculate_rejects_zero_divisors(operator: str) -> None:
    with pytest.raises(CalculatorError, match="Division by zero"):
        calculate(f"1 {operator} 0")


def test_calculate_limits_integer_exponents() -> None:
    with pytest.raises(CalculatorError, match="cannot exceed 100"):
        calculate("2 ** 101")

    with pytest.raises(CalculatorError, match="must be an integer"):
        calculate("2 ** 0.5")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "unknown_name + 1",
        "[1, 2][0]",
        "True + 1",
        "1 << 2",
    ],
)
def test_calculate_rejects_non_arithmetic_syntax(expression: str) -> None:
    with pytest.raises(CalculatorError, match="Unsupported|numeric literals"):
        calculate(expression)


def test_calculate_rejects_oversized_and_overly_complex_expressions() -> None:
    with pytest.raises(CalculatorError, match="cannot exceed"):
        calculate("1" * (MAX_EXPRESSION_LENGTH + 1))

    balanced_expression = "1"
    for _ in range(5):
        balanced_expression = f"({balanced_expression} + {balanced_expression})"
    with pytest.raises(CalculatorError, match="too complex"):
        calculate(balanced_expression)


def test_calculate_rejects_deep_nesting_and_large_magnitudes() -> None:
    with pytest.raises(CalculatorError, match="nesting is too deep"):
        calculate("-" * 20 + "1")

    with pytest.raises(CalculatorError, match="magnitude range"):
        calculate("1e101")


def test_calculate_rejects_empty_expression_and_long_literal() -> None:
    with pytest.raises(CalculatorError, match="cannot be empty"):
        calculate("   ")

    with pytest.raises(CalculatorError, match="literal is too long"):
        calculate("9" * 51)


def test_calculator_tool_schema_is_strict_and_bounded() -> None:
    function = CALCULATOR_TOOL["function"]
    assert isinstance(function, dict)
    assert function["name"] == "calculator"
    assert function["strict"] is True
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["required"] == ["expression"]
    assert parameters["additionalProperties"] is False
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["expression"]["maxLength"] == MAX_EXPRESSION_LENGTH
