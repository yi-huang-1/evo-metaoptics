"""
Signature-based validation for MCE interface implementations.

Validates that agent implementations match the required InterfaceSignatures.
"""

import ast
import sys
import inspect
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evo_metaoptics.mce_env.base import InterfaceSignature

_MAX_VALIDATION_FEEDBACK_CHARS = 1600
_MAX_VALIDATION_ERRORS_IN_FEEDBACK = 8
_INTERFACE_ERROR_RE = re.compile(r"^\[(?P<iface>[^\]]+)\]\s*(?P<body>.+)$")
_MISSING_FILE_RE = re.compile(r"File not found:\s*(?P<path>interfaces/[^\s]+\.py)")


@dataclass
class ValidationResult:
    """Result of interface validation."""
    success: bool
    errors: List[str] = field(default_factory=list)
    interfaces: Dict[str, Callable] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.success:
            return f"Validation passed: {len(self.interfaces)} interfaces loaded"
        return f"Validation failed with {len(self.errors)} errors:\n" + "\n".join(f"  - {e}" for e in self.errors)


def validate_interfaces(
    iter_dir: Path,
    signatures: List[InterfaceSignature]
) -> ValidationResult:
    """
    Validate all interface implementations against their signatures.
    
    Args:
        iter_dir: Iteration directory containing interfaces/
        signatures: List of required interface signatures
        
    Returns:
        ValidationResult with success status, errors, and loaded interfaces
    """
    errors = []
    interfaces = {}
    
    iter_dir = Path(iter_dir)
    interfaces_dir = iter_dir / "interfaces"
    
    # Check interfaces directory exists
    if not interfaces_dir.exists():
        return ValidationResult(
            success=False,
            errors=["interfaces/ directory not found. Create it and implement required functions."]
        )
    
    # Check __init__.py exists
    init_file = interfaces_dir / "__init__.py"
    if not init_file.exists():
        errors.append("interfaces/__init__.py not found. Create it to export your functions.")
    
    # Validate each signature from its canonical file.
    for sig in signatures:
        result = _validate_single_interface(iter_dir, sig)
        if result["error"]:
            errors.append(f"[{sig.name}] {result['error']}")
        else:
            interfaces[sig.name] = result["function"]

    # Ensure interfaces/__init__.py exports the required callables.
    exported_interfaces: Dict[str, Callable] = {}
    if init_file.exists():
        try:
            exported_interfaces = load_interfaces_from_init(iter_dir)
        except Exception as exc:
            errors.append(f"Failed to load interfaces/__init__.py exports: {exc}")

    for sig in signatures:
        exported = exported_interfaces.get(sig.name)
        if not callable(exported):
            errors.append(
                f"[{sig.name}] interfaces/__init__.py must export '{sig.name}' "
                f"(for example: from .{sig.name} import {sig.name}; include in __all__)."
            )
            continue

        expected_params = sig.get_expected_params()
        mismatch = _validate_callable_params(exported, expected_params)
        if mismatch is not None:
            errors.append(
                f"[{sig.name}] Exported callable parameter mismatch. {mismatch}"
            )
            continue

        # Prefer exported callable to match runtime import path.
        interfaces[sig.name] = exported
    
    return ValidationResult(
        success=len(errors) == 0,
        errors=errors,
        interfaces=interfaces
    )


def _validate_single_interface(
    iter_dir: Path,
    sig: InterfaceSignature
) -> Dict[str, Any]:
    """
    Validate a single interface implementation.
    
    Checks:
    1. File exists: interfaces/{name}.py
    2. Function exists with correct name
    3. Function signature matches (parameter names)
    4. Function can be imported and called
    
    Returns:
        Dict with 'function' (if valid) or 'error' (if invalid)
    """
    interfaces_dir = iter_dir / "interfaces"
    file_path = interfaces_dir / f"{sig.name}.py"
    
    # 1. Check file exists
    if not file_path.exists():
        return {"error": f"File not found: interfaces/{sig.name}.py", "function": None}
    
    # 2. Parse AST and find function
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in {sig.name}.py: {e}", "function": None}
    except Exception as e:
        return {"error": f"Failed to parse {sig.name}.py: {e}", "function": None}
    
    # Find the function definition
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == sig.name:
            func_node = node
            break
    
    if func_node is None:
        return {"error": f"Function '{sig.name}' not found in interfaces/{sig.name}.py", "function": None}
    
    # 3. Validate parameter names
    expected_params = sig.get_expected_params()
    actual_params = [arg.arg for arg in func_node.args.args]
    
    if actual_params != expected_params:
        return {
            "error": f"Parameter mismatch. Expected: ({', '.join(expected_params)}), Got: ({', '.join(actual_params)})",
            "function": None
        }
    
    # 4. Check for return statement
    has_return = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            has_return = True
            break
    
    if not has_return:
        return {"error": f"Function '{sig.name}' has no return statement with a value", "function": None}
    
    # 5. Try to import and get the function
    try:
        func = _import_function(file_path, sig.name)
    except Exception as e:
        return {"error": f"Import failed: {e}", "function": None}
    
    return {"error": None, "function": func}


def _validate_callable_params(func: Callable, expected_params: List[str]) -> str | None:
    """Validate callable parameter names/order against expected signature."""
    try:
        signature = inspect.signature(func)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Unable to inspect callable signature: {exc}"

    actual_params: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return "Do not use *args/**kwargs for required interfaces."
        actual_params.append(param.name)

    if actual_params != expected_params:
        return (
            f"Expected: ({', '.join(expected_params)}), "
            f"Got: ({', '.join(actual_params)})"
        )

    return None


def _import_function(file_path: Path, func_name: str) -> Callable:
    """
    Dynamically import a function from a file.
    
    Args:
        file_path: Path to the Python file
        func_name: Name of the function to import
        
    Returns:
        The imported function
        
    Raises:
        Exception: If import fails
    """
    module_name = f"interfaces_{func_name}_{id(file_path)}"
    
    # Clean up if already imported
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {e}")
    
    if not hasattr(module, func_name):
        del sys.modules[module_name]
        raise ImportError(f"Module does not have function '{func_name}'")
    
    return getattr(module, func_name)


def load_interfaces_from_init(iter_dir: Path) -> Dict[str, Callable]:
    """
    Load all interfaces from interfaces/__init__.py.
    
    This is used during evaluation to load the complete interface module.
    
    Args:
        iter_dir: Iteration directory
        
    Returns:
        Dict mapping function names to callables
    """
    interfaces_dir = iter_dir / "interfaces"
    init_file = interfaces_dir / "__init__.py"
    
    if not init_file.exists():
        raise FileNotFoundError(f"interfaces/__init__.py not found in {iter_dir}")
    
    module_name = f"interfaces_module_{id(iter_dir)}"
    
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {init_file}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    # Add interfaces directory to path for relative imports
    interfaces_parent = str(interfaces_dir.parent)
    if interfaces_parent not in sys.path:
        sys.path.insert(0, interfaces_parent)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load interfaces module: {e}")
    
    # Get exported names
    if hasattr(module, '__all__'):
        names = module.__all__
    else:
        names = [n for n in dir(module) if not n.startswith('_')]
    
    interfaces = {}
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            interfaces[name] = obj
    
    return interfaces


def format_validation_feedback(result: ValidationResult) -> str:
    """
    Format validation result as feedback for the agent.
    
    Args:
        result: ValidationResult from validate_interfaces
        
    Returns:
        Formatted string for agent prompt
    """
    if result.success:
        return f"All {len(result.interfaces)} interfaces validated successfully."
    
    compact_errors = list(result.errors[:_MAX_VALIDATION_ERRORS_IN_FEEDBACK])
    lines = ["Interface validation failed. Please fix the following issues:", ""]

    for error in compact_errors:
        lines.append(f"- {error}")

    signatures = _build_validation_signatures(compact_errors)
    if signatures:
        lines.extend(["", "Validation signatures:"])
        for signature in signatures:
            lines.append(f"- {signature}")
    
    lines.extend([
        "",
        "Remember:",
        "- Each interface must be in `interfaces/{name}.py`",
        "- Function name must match exactly",
        "- Parameter names must match the signature",
        "- Function must have a return statement",
        "- Export functions in `interfaces/__init__.py`",
    ])
    
    feedback = "\n".join(lines)
    if len(feedback) > _MAX_VALIDATION_FEEDBACK_CHARS:
        feedback = feedback[: _MAX_VALIDATION_FEEDBACK_CHARS - 3].rstrip() + "..."
    return feedback


def _build_validation_signatures(errors: List[str]) -> List[str]:
    signatures: list[str] = []
    seen: set[str] = set()
    for error in errors:
        text = str(error)
        iface = "unknown"
        body = text
        match = _INTERFACE_ERROR_RE.match(text)
        if match:
            iface = match.group("iface").strip() or "unknown"
            body = match.group("body").strip()

        normalized_path = f"/interfaces/{iface}.py"

        missing_match = _MISSING_FILE_RE.search(body)
        if missing_match:
            normalized_path = "/" + missing_match.group("path").strip().lstrip("/")
            candidate = f"missing_file:{normalized_path}"
        elif "Parameter mismatch" in body or "parameter mismatch" in body.lower():
            candidate = f"param_mismatch:{normalized_path}"
        elif "Syntax error" in body:
            candidate = f"syntax_error:{normalized_path}"
        elif "Import failed" in body:
            candidate = f"import_failed:{normalized_path}"
        else:
            candidate = f"validation_error:{normalized_path}"

        if candidate in seen:
            continue
        seen.add(candidate)
        signatures.append(candidate)
    return signatures
