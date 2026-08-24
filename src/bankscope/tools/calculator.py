"""A small, deterministic arithmetic tool with no dynamic code execution."""

from __future__ import annotations

import ast
from decimal import (
    ROUND_FLOOR,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    Underflow,
    localcontext,
)

MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 64
MAX_AST_DEPTH = 16
MAX_LITERAL_DIGITS = 50
MAX_ABSOLUTE_POWER = 100
MAX_ADJUSTED_EXPONENT = 100
DECIMAL_PRECISION = 50


class CalculatorError(ValueError):
    """Raised when an expression is invalid, unsupported, or exceeds a safety limit."""


CALCULATOR_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Safely evaluate deterministic arithmetic. Use numeric literals, parentheses, "
            "and the operators +, -, *, /, //, %, or ** only."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                    "minLength": 1,
                    "maxLength": MAX_EXPRESSION_LENGTH,
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}


def _validate_complexity(tree: ast.AST) -> None:
    node_count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise CalculatorError("Expression is too complex.")
        if depth > MAX_AST_DEPTH:
            raise CalculatorError("Expression nesting is too deep.")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _guard_result(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise CalculatorError("Expression produced a non-finite result.")
    if not value.is_zero() and abs(value.adjusted()) > MAX_ADJUSTED_EXPONENT:
        raise CalculatorError("Result is outside the supported magnitude range.")
    return value


class _DecimalEvaluator:
    def __init__(self, source: str) -> None:
        self.source = source

    def evaluate(self, node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        raise CalculatorError(f"Unsupported calculator syntax: {type(node).__name__}.")

    def _constant(self, node: ast.Constant) -> Decimal:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalculatorError("Only numeric literals are supported.")

        literal = ast.get_source_segment(self.source, node) or str(value)
        if sum(character.isdigit() for character in literal) > MAX_LITERAL_DIGITS:
            raise CalculatorError("Numeric literal is too long.")

        try:
            if isinstance(value, int):
                result = Decimal(value)
            else:
                result = Decimal(literal.replace("_", ""))
        except (DecimalException, ValueError) as error:
            raise CalculatorError("Invalid numeric literal.") from error
        return _guard_result(result)

    def _unary(self, node: ast.UnaryOp) -> Decimal:
        operand = self.evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _guard_result(+operand)
        if isinstance(node.op, ast.USub):
            return _guard_result(-operand)
        raise CalculatorError(f"Unsupported unary operator: {type(node.op).__name__}.")

    def _binary(self, node: ast.BinOp) -> Decimal:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        try:
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                self._require_nonzero_divisor(right)
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                self._require_nonzero_divisor(right)
                result = self._floor_quotient(left, right)
            elif isinstance(node.op, ast.Mod):
                self._require_nonzero_divisor(right)
                quotient = self._floor_quotient(left, right)
                result = left - quotient * right
            elif isinstance(node.op, ast.Pow):
                result = self._power(left, right)
            else:
                raise CalculatorError(f"Unsupported binary operator: {type(node.op).__name__}.")
        except CalculatorError:
            raise
        except (DecimalException, ZeroDivisionError) as error:
            raise CalculatorError("Arithmetic operation is outside supported limits.") from error
        return _guard_result(result)

    @staticmethod
    def _require_nonzero_divisor(divisor: Decimal) -> None:
        if divisor.is_zero():
            raise CalculatorError("Division by zero is not allowed.")

    @staticmethod
    def _floor_quotient(dividend: Decimal, divisor: Decimal) -> Decimal:
        return (dividend / divisor).to_integral_value(rounding=ROUND_FLOOR)

    @staticmethod
    def _power(base: Decimal, exponent: Decimal) -> Decimal:
        integral_exponent = exponent.to_integral_value()
        if exponent != integral_exponent:
            raise CalculatorError("Exponent must be an integer.")
        if abs(integral_exponent) > MAX_ABSOLUTE_POWER:
            raise CalculatorError(f"Absolute exponent cannot exceed {MAX_ABSOLUTE_POWER}.")
        if base.is_zero() and integral_exponent < 0:
            raise CalculatorError("Division by zero is not allowed.")
        return base ** int(integral_exponent)


def evaluate_expression(expression: str) -> Decimal:
    """Evaluate one bounded arithmetic expression and return its Decimal result."""

    if not isinstance(expression, str):
        raise CalculatorError("Expression must be a string.")
    source = expression.strip()
    if not source:
        raise CalculatorError("Expression cannot be empty.")
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression cannot exceed {MAX_EXPRESSION_LENGTH} characters.")

    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise CalculatorError("Expression is not valid arithmetic.") from error
    _validate_complexity(tree)

    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.Emax = MAX_ADJUSTED_EXPONENT
        context.Emin = -MAX_ADJUSTED_EXPONENT
        context.traps[DivisionByZero] = True
        context.traps[InvalidOperation] = True
        context.traps[Overflow] = True
        context.traps[Underflow] = True
        try:
            return _guard_result(_DecimalEvaluator(source).evaluate(tree.body))
        except CalculatorError:
            raise
        except DecimalException as error:
            raise CalculatorError("Arithmetic operation is outside supported limits.") from error


def _format_result(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        normalized = value.normalize()
    if -6 <= normalized.adjusted() <= 20:
        return format(normalized, "f")
    return str(normalized)


def calculate(expression: str) -> str:
    """Return a compact, JSON-safe string result for the calculator tool."""

    return _format_result(evaluate_expression(expression))
