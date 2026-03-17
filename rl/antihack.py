"""
antihack.py - SP3 Anti-Hacking Defenses for RL

Static analysis and security checks for generated CUDA kernels.
Prevents models from "gaming" the reward function by bypassing computation,
hardcoding answers, or spawning unauthorized threads.
"""
import ast

BANNED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "threading", 
    "multiprocessing", "urllib", "requests", "http"
}

# The model must construct and use a custom CUDA extension,
# not just call torch.nn.functional or torch.xxx directly in the forward pass.
# We ban calling `torch.nn.functional` entirely within ModelNew methods.
BANNED_CALL_PREFIXES = [
    "torch.nn.functional",
    "F.",
]


class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in BANNED_IMPORTS:
                self.violations.append(f"Banned import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in BANNED_IMPORTS:
            self.violations.append(f"Banned import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for calls like torch.nn.functional.relu()
        func_name = self._get_full_attribute_name(node.func)
        if func_name:
            for prefix in BANNED_CALL_PREFIXES:
                if func_name.startswith(prefix):
                    self.violations.append(f"Banned function call: {func_name}")
        self.generic_visit(node)
        
    def _get_full_attribute_name(self, node):
        """Helper to extract names like 'torch.nn.functional.relu' from AST."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_full_attribute_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        return None


def check_security(code: str) -> tuple[bool, str]:
    """
    Perform static security analysis on the generated code.
    
    Args:
        code: python source code string containing ModelNew and inline CUDA
        
    Returns:
        (is_safe, error_message)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # If it doesn't parse, it's not a security threat, it's just broken code.
        # We let the sandbox catch the syntax error later so the RL gets the
        # correct compiler error feedback.
        return True, ""
        
    visitor = SecurityASTVisitor()
    visitor.visit(tree)
    
    if visitor.violations:
        return False, "SECURITY VIOLATION: " + "; ".join(visitor.violations)
        
    # Check if a custom CUDA extension is actually defined via load_inline
    if "load_inline(" not in code:
        return False, "SECURITY VIOLATION: Must compile custom CUDA extension using load_inline()"
        
    return True, ""
