"""OLMo/BAR tool-message adapter and non-executing call validation.

The wire format follows Ai2's Dolci tool data and OLMES code-call handler.
Function definitions are OpenAI-shaped JSON; calls are tagged Python literals.
Parsing a call never executes Python or invokes an external tool.
"""

import ast
import copy
import json
import math
import re


CALL_INSTRUCTIONS = (
    "You are a helpful assistant with access to the functions listed below. "
    "Use a provided function when it can answer the user's request. "
    "For a function call, reply only with a <function_calls> block, one call per line, "
    "and close it with </function_calls>. Use named arguments and literal values. "
    "The syntax is <function_calls>function_name(argument_name=\"value\")</function_calls>. "
    "Choose the function and values from the request and the supplied definitions; "
    "the syntax example is not a function to call. "
    "Do not calculate a tool result yourself or invent missing arguments. "
    "If a required argument is missing, ask the user to supply it."
)


def definitions(functions):
    """Validate the finite tool registry while preserving the upstream envelope."""
    if not isinstance(functions, list) or not functions:
        raise ValueError("tool definitions must be a nonempty list")
    by_name = {}
    for item in functions:
        if not isinstance(item, dict) or item.get("type") != "function":
            raise ValueError("tool definition needs the function envelope")
        spec = item.get("function", {})
        if not isinstance(spec, dict):
            raise ValueError("function definition must be an object")
        name = spec.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", name):
            raise ValueError("invalid function name")
        if name in by_name:
            raise ValueError("duplicate function name")
        schema = spec.get("parameters", {})
        if not isinstance(schema, dict):
            raise ValueError("function parameters must be an object schema")
        properties = schema.get("properties")
        if schema.get("type") != "object" or not isinstance(properties, dict):
            raise ValueError("function parameters must be an object schema")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(key, str) and key in properties for key in required):
            raise ValueError("required argument is absent from the schema")
        by_name[name] = spec
    return by_name


def tool_messages(messages, functions):
    """Build model-visible input using only the conversation and available tools."""
    definitions(functions)
    if not messages or any(message.get("role") not in ("user", "assistant", "environment") for message in messages):
        raise ValueError("provide conversation turns without a second system message")
    return [{"role": "system", "content": CALL_INSTRUCTIONS,
             "functions": json.dumps(functions, separators=(",", ":"), ensure_ascii=False)},
            *copy.deepcopy(messages)]


def _function_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _function_name(node.value) + "." + node.attr
    raise ValueError("function name is not a registry identifier")


def _check_value(value, schema):
    if not isinstance(schema, dict):
        raise ValueError("invalid argument schema")
    kind = schema.get("type")
    valid = {"string": lambda: isinstance(value, str),
             "integer": lambda: type(value) is int,
             "number": lambda: type(value) in (int, float) and math.isfinite(value),
             "boolean": lambda: type(value) is bool,
             "null": lambda: value is None,
             "array": lambda: isinstance(value, list),
             "object": lambda: isinstance(value, dict)}
    if kind not in valid or not valid[kind]():
        raise ValueError("argument type does not match its schema")
    if "enum" in schema and not any(type(value) is type(choice) and value == choice for choice in schema["enum"]):
        raise ValueError("argument is outside its enum")
    if kind == "array":
        for item in value:
            _check_value(item, schema["items"])
    if kind == "object":
        properties = schema.get("properties", {})
        if set(value) - set(properties) or set(schema.get("required", [])) - set(value):
            raise ValueError("missing or unknown object argument")
        for name, item in value.items():
            _check_value(item, properties[name])


def parse_calls(text, functions):
    """Reject executable expressions, unknown calls, duplicates and invalid types."""
    registry = definitions(functions)
    start, end = "<function_calls>", "</function_calls>"
    if not isinstance(text, str) or len(text) > 65536 or text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("expected exactly one function-call block")
    prefix, body = text.split(start)
    body, suffix = body.split(end)
    if prefix.strip() or suffix.strip():
        raise ValueError("unexpected text outside the function-call block")
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 16:
        raise ValueError("invalid number of function calls")
    calls = []
    for line in lines:
        try:
            node = ast.parse(line, mode="eval").body
            if not isinstance(node, ast.Call) or node.args:
                raise ValueError("a call with keyword arguments is required")
            name = _function_name(node.func)
            if name not in registry:
                raise ValueError("unknown function")
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg in kwargs:
                    raise ValueError("expanded or duplicate keyword argument")
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            _check_value(kwargs, registry[name]["parameters"])
            calls.append({"name": name, "arguments": kwargs})
        except (SyntaxError, TypeError, KeyError, RecursionError, OverflowError) as error:
            raise ValueError("invalid function-call syntax or schema") from error
    return calls


def validate_reply(text, functions):
    try:
        return {"valid": True, "calls": parse_calls(text, functions)}
    except ValueError as error:
        return {"valid": False, "error": str(error)}
