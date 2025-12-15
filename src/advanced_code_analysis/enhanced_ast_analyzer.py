"""
Enhanced AST Analysis Engine for Advanced Code Analysis.

This module implements LLM-guided AST analysis that focuses on specific code regions
identified by the LLM, performs targeted error pattern detection, validates function
calls, and tracks data flow for key variables.
"""

import ast
from typing import List, Dict, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import re

from .models import (
    AnalysisResult,
    ReasoningChain,
    ReasoningStep,
    EvidenceChain,
    Conflict,
    ContextWindow
)
from .llm_interface import LLMInterface


@dataclass
class SuspiciousRegion:
    """Represents a code region identified as suspicious by LLM."""
    start_line: int
    end_line: int
    reason: str
    confidence: float
    code_snippet: str
    suggested_focus: List[str] = field(default_factory=list)


@dataclass
class ErrorPattern:
    """Represents a detected error pattern in code."""
    pattern_type: str
    location: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    suggested_fix: str = ""
    code_context: str = ""


@dataclass
class FunctionCallValidation:
    """Result of function call validation."""
    function_name: str
    call_location: str
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    expected_signature: Optional[str] = None
    actual_call: str = ""


@dataclass
class DataFlowTrace:
    """Represents data flow trace for a variable."""
    variable_name: str
    definition_points: List[Tuple[int, str]] = field(default_factory=list)  # (line, context)
    usage_points: List[Tuple[int, str]] = field(default_factory=list)  # (line, context)
    modifications: List[Tuple[int, str]] = field(default_factory=list)  # (line, context)
    flow_path: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)


class EnhancedASTAnalyzer:
    """
    Enhanced AST analyzer that performs LLM-guided targeted analysis.
    
    This analyzer extends traditional AST analysis by:
    1. Focusing on LLM-identified suspicious regions
    2. Detecting common error patterns
    3. Validating function calls
    4. Tracking data flow for key variables
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """Initialize the enhanced AST analyzer."""
        self.llm_interface = llm_interface
        self.error_patterns = self._initialize_error_patterns()
        self.function_signatures = {}  # Cache for function signatures
        self.class_methods = {}  # Cache for class methods
        
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns to detect."""
        return {
            "constant_assignment": {
                "description": "Assignment of constant value instead of variable",
                "pattern": r"(\w+)\s*=\s*(\d+|'[^']*'|\"[^\"]*\"|True|False|None)",
                "severity": "medium",
                "check_function": self._check_constant_assignment
            },
            "wrong_comparison": {
                "description": "Using assignment (=) instead of comparison (==)",
                "pattern": r"if\s+\w+\s*=\s*",
                "severity": "high",
                "check_function": self._check_wrong_comparison
            },
            "undefined_variable": {
                "description": "Usage of potentially undefined variable",
                "pattern": None,  # Requires AST analysis
                "severity": "high",
                "check_function": self._check_undefined_variable
            },
            "type_mismatch": {
                "description": "Potential type mismatch in operations",
                "pattern": None,  # Requires AST analysis
                "severity": "medium",
                "check_function": self._check_type_mismatch
            },
            "unreachable_code": {
                "description": "Code that may never be executed",
                "pattern": None,  # Requires AST analysis
                "severity": "low",
                "check_function": self._check_unreachable_code
            }
        }
    
    async def identify_suspicious_regions(self, bug_location: str, 
                                        target_files: List[str]) -> List[SuspiciousRegion]:
        """
        Identify suspicious code regions based on LLM analysis results.
        
        Args:
            bug_location: Location identified by LLM analysis
            target_files: List of files to analyze
            
        Returns:
            List of suspicious regions for targeted analysis
        """
        suspicious_regions = []
        
        try:
            # Parse bug location to extract file and line information
            file_info = self._parse_bug_location(bug_location)
            
            for file_path in target_files:
                if not file_path.endswith('.py'):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # If specific location mentioned, focus on that area
                    if file_info and file_path.endswith(file_info.get('file', '')):
                        line_num = file_info.get('line', 1)
                        region = SuspiciousRegion(
                            start_line=max(1, line_num - 5),
                            end_line=line_num + 5,
                            reason=f"LLM identified issue near line {line_num}",
                            confidence=0.8,
                            code_snippet=self._extract_code_snippet(code, line_num - 5, line_num + 5),
                            suggested_focus=["assignment", "function_call", "variable_usage"]
                        )
                        suspicious_regions.append(region)
                    else:
                        # General suspicious patterns
                        regions = self._identify_general_suspicious_patterns(code, file_path)
                        suspicious_regions.extend(regions)
                        
                except (IOError, UnicodeDecodeError) as e:
                    continue
            
            return suspicious_regions
            
        except Exception as e:
            # Return empty list on error
            return []
    
    def _parse_bug_location(self, location: str) -> Optional[Dict[str, Any]]:
        """Parse bug location string to extract file and line info."""
        import re
        
        # Try to extract file and line number
        file_match = re.search(r'([^/\\]+\.py)', location)
        line_match = re.search(r'line\s*(\d+)', location, re.IGNORECASE)
        
        if file_match or line_match:
            return {
                'file': file_match.group(1) if file_match else None,
                'line': int(line_match.group(1)) if line_match else None
            }
        return None
    
    def _extract_code_snippet(self, code: str, start_line: int, end_line: int) -> str:
        """Extract code snippet from given line range."""
        lines = code.split('\n')
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        return '\n'.join(lines[start_idx:end_idx])
    
    def _identify_general_suspicious_patterns(self, code: str, file_path: str) -> List[SuspiciousRegion]:
        """Identify general suspicious patterns in code."""
        regions = []
        lines = code.split('\n')
        
        # Look for common suspicious patterns
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Assignment in if conditions
            if re.search(r'if\s+.*=\s*[^=]', line_stripped):
                regions.append(SuspiciousRegion(
                    start_line=max(1, i - 2),
                    end_line=i + 2,
                    reason="Assignment in if condition (possible typo)",
                    confidence=0.7,
                    code_snippet=self._extract_code_snippet(code, i - 2, i + 2),
                    suggested_focus=["comparison_operator"]
                ))
            
            # Constant assignments that might be wrong
            if re.search(r'\w+\s*=\s*\d+\s*$', line_stripped):
                regions.append(SuspiciousRegion(
                    start_line=max(1, i - 1),
                    end_line=i + 1,
                    reason="Constant assignment (might need variable)",
                    confidence=0.5,
                    code_snippet=self._extract_code_snippet(code, i - 1, i + 1),
                    suggested_focus=["variable_assignment"]
                ))
        
        return regions
    
    def analyze_suspicious_regions(
        self, 
        code: str, 
        suspicious_regions: List[SuspiciousRegion],
        context: Optional[ContextWindow] = None
    ) -> List[AnalysisResult]:
        """
        Perform targeted analysis on LLM-identified suspicious regions.
        
        Args:
            code: Complete source code
            suspicious_regions: Regions identified as suspicious by LLM
            context: Additional context information
            
        Returns:
            List of analysis results for each region
        """
        results = []
        
        try:
            # Parse the complete code
            tree = ast.parse(code)
            code_lines = code.split('\n')
            
            # Build function and class caches
            self._build_signature_cache(tree)
            
            for region in suspicious_regions:
                try:
                    # Extract region-specific code
                    region_code = self._extract_region_code(code_lines, region)
                    
                    # Perform targeted analysis on this region
                    region_result = self._analyze_code_region(
                        tree, region, region_code, code_lines, context
                    )
                    
                    if region_result:
                        results.append(region_result)
                        
                except Exception as e:
                    # Create error result for this region
                    error_result = AnalysisResult(
                        bug_location=f"lines {region.start_line}-{region.end_line}",
                        root_cause=f"Analysis error: {str(e)}",
                        fix_suggestion="Manual review required due to analysis error",
                        confidence=0.1,
                        reasoning_chain=ReasoningChain(),
                        supporting_evidence=[f"Error during region analysis: {str(e)}"]
                    )
                    results.append(error_result)
                    
        except SyntaxError as e:
            # Handle syntax errors in the code
            syntax_result = AnalysisResult(
                bug_location=f"line {e.lineno}" if e.lineno else "unknown",
                root_cause=f"Syntax error: {str(e)}",
                fix_suggestion="Fix syntax error before analysis",
                confidence=0.9,
                reasoning_chain=ReasoningChain(),
                supporting_evidence=[f"Python syntax error: {str(e)}"]
            )
            results.append(syntax_result)
            
        return results
    
    def detect_error_patterns(self, code: str, focus_lines: Optional[Set[int]] = None) -> List[ErrorPattern]:
        """
        Detect common error patterns in the code.
        
        Args:
            code: Source code to analyze
            focus_lines: Optional set of line numbers to focus on
            
        Returns:
            List of detected error patterns
        """
        patterns = []
        
        try:
            tree = ast.parse(code)
            code_lines = code.split('\n')
            
            # Check each error pattern
            for pattern_name, pattern_info in self.error_patterns.items():
                try:
                    detected_patterns = pattern_info["check_function"](
                        tree, code_lines, focus_lines
                    )
                    patterns.extend(detected_patterns)
                except Exception as e:
                    # Log pattern check error but continue
                    continue
                    
        except SyntaxError:
            # If code has syntax errors, try to detect them as patterns
            syntax_patterns = self._detect_syntax_error_patterns(code)
            patterns.extend(syntax_patterns)
            
        return patterns
    
    def validate_function_calls(
        self, 
        code: str, 
        target_functions: Optional[List[str]] = None
    ) -> List[FunctionCallValidation]:
        """
        Validate function calls for parameter types and counts.
        
        Args:
            code: Source code to analyze
            target_functions: Optional list of specific functions to validate
            
        Returns:
            List of function call validation results
        """
        validations = []
        
        try:
            tree = ast.parse(code)
            
            # Build function signature cache
            self._build_signature_cache(tree)
            
            # Find all function calls
            call_validator = FunctionCallValidator(
                self.function_signatures, 
                self.class_methods,
                target_functions
            )
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    validation = call_validator.validate_call(node)
                    if validation:
                        validations.append(validation)
                        
        except SyntaxError as e:
            # Create validation error for syntax issues
            error_validation = FunctionCallValidation(
                function_name="syntax_error",
                call_location=f"line {e.lineno}" if e.lineno else "unknown",
                is_valid=False,
                issues=[f"Syntax error prevents validation: {str(e)}"]
            )
            validations.append(error_validation)
            
        return validations
    
    def trace_variable_data_flow(
        self, 
        code: str, 
        key_variables: List[str],
        focus_region: Optional[Tuple[int, int]] = None
    ) -> Dict[str, DataFlowTrace]:
        """
        Trace data flow for key variables identified by LLM.
        
        Args:
            code: Source code to analyze
            key_variables: List of variable names to trace
            focus_region: Optional tuple of (start_line, end_line) to focus on
            
        Returns:
            Dictionary mapping variable names to their data flow traces
        """
        traces = {}
        
        try:
            tree = ast.parse(code)
            code_lines = code.split('\n')
            
            # Create data flow tracer
            tracer = DataFlowTracer(key_variables, focus_region)
            
            # Trace each variable
            for var_name in key_variables:
                trace = tracer.trace_variable(tree, code_lines, var_name)
                traces[var_name] = trace
                
        except SyntaxError as e:
            # Create error traces for syntax issues
            for var_name in key_variables:
                error_trace = DataFlowTrace(
                    variable_name=var_name,
                    potential_issues=[f"Syntax error prevents tracing: {str(e)}"]
                )
                traces[var_name] = error_trace
                
        return traces
    
    def integrate_with_llm_analysis(
        self, 
        code: str, 
        llm_analysis: AnalysisResult,
        context: Optional[ContextWindow] = None
    ) -> AnalysisResult:
        """
        Integrate AST analysis with LLM analysis results.
        
        Args:
            code: Source code being analyzed
            llm_analysis: Initial analysis result from LLM
            context: Additional context information
            
        Returns:
            Enhanced analysis result combining LLM and AST insights
        """
        try:
            # Extract suspicious regions from LLM analysis
            suspicious_regions = self._extract_suspicious_regions_from_llm(llm_analysis)
            
            # Perform targeted AST analysis
            ast_results = self.analyze_suspicious_regions(code, suspicious_regions, context)
            
            # Detect error patterns in the identified regions
            focus_lines = set()
            for region in suspicious_regions:
                focus_lines.update(range(region.start_line, region.end_line + 1))
            
            error_patterns = self.detect_error_patterns(code, focus_lines)
            
            # Validate function calls mentioned in LLM analysis
            target_functions = self._extract_function_names_from_analysis(llm_analysis)
            call_validations = self.validate_function_calls(code, target_functions)
            
            # Trace key variables mentioned in LLM analysis
            key_variables = self._extract_variable_names_from_analysis(llm_analysis)
            data_flows = self.trace_variable_data_flow(code, key_variables)
            
            # Integrate all results
            integrated_result = self._integrate_analysis_results(
                llm_analysis, ast_results, error_patterns, call_validations, data_flows
            )
            
            return integrated_result
            
        except Exception as e:
            # Return original LLM analysis with error note if integration fails
            llm_analysis.supporting_evidence.append(f"AST integration error: {str(e)}")
            llm_analysis.confidence *= 0.8  # Reduce confidence due to integration failure
            return llm_analysis
    
    def _extract_region_code(self, code_lines: List[str], region: SuspiciousRegion) -> str:
        """Extract code for a specific region."""
        start_idx = max(0, region.start_line - 1)
        end_idx = min(len(code_lines), region.end_line)
        return '\n'.join(code_lines[start_idx:end_idx])
    
    def _analyze_code_region(
        self, 
        tree: ast.AST, 
        region: SuspiciousRegion, 
        region_code: str,
        code_lines: List[str],
        context: Optional[ContextWindow]
    ) -> Optional[AnalysisResult]:
        """Analyze a specific code region identified as suspicious."""
        reasoning_chain = ReasoningChain()
        evidence_chain = EvidenceChain()
        
        # Step 1: Analyze the region for error patterns
        step1 = ReasoningStep(
            step_number=1,
            description="Analyzing suspicious region for error patterns",
            input_data={"region": f"lines {region.start_line}-{region.end_line}"},
            confidence=0.8
        )
        
        focus_lines = set(range(region.start_line, region.end_line + 1))
        error_patterns = self.detect_error_patterns('\n'.join(code_lines), focus_lines)
        
        step1.output_data = {"error_patterns": len(error_patterns)}
        step1.evidence = [f"Found {len(error_patterns)} potential error patterns"]
        reasoning_chain.add_step(step1)
        
        # Step 2: Validate function calls in the region
        step2 = ReasoningStep(
            step_number=2,
            description="Validating function calls in region",
            confidence=0.7
        )
        
        call_validations = self.validate_function_calls(region_code)
        invalid_calls = [v for v in call_validations if not v.is_valid]
        
        step2.output_data = {"invalid_calls": len(invalid_calls)}
        step2.evidence = [f"Found {len(invalid_calls)} invalid function calls"]
        reasoning_chain.add_step(step2)
        
        # Step 3: Check for data flow issues
        step3 = ReasoningStep(
            step_number=3,
            description="Analyzing data flow in region",
            confidence=0.6
        )
        
        # Extract variables mentioned in the region
        region_variables = self._extract_variables_from_region(region_code)
        data_flows = self.trace_variable_data_flow('\n'.join(code_lines), region_variables)
        
        flow_issues = []
        for var_name, trace in data_flows.items():
            flow_issues.extend(trace.potential_issues)
        
        step3.output_data = {"flow_issues": len(flow_issues)}
        step3.evidence = [f"Found {len(flow_issues)} data flow issues"]
        reasoning_chain.add_step(step3)
        
        # Determine if this region has significant issues
        total_issues = len(error_patterns) + len(invalid_calls) + len(flow_issues)
        
        if total_issues == 0:
            return None  # No issues found in this region
        
        # Build analysis result
        confidence = reasoning_chain.calculate_overall_confidence()
        
        # Compile evidence
        evidence_items = []
        if error_patterns:
            evidence_items.extend([f"Error pattern: {p.description}" for p in error_patterns])
        if invalid_calls:
            evidence_items.extend([f"Invalid call: {v.function_name}" for v in invalid_calls])
        if flow_issues:
            evidence_items.extend(flow_issues)
        
        # Generate fix suggestion based on findings
        fix_suggestion = self._generate_fix_suggestion(error_patterns, invalid_calls, flow_issues)
        
        result = AnalysisResult(
            bug_location=f"lines {region.start_line}-{region.end_line}",
            root_cause=region.reason,
            fix_suggestion=fix_suggestion,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            supporting_evidence=evidence_items
        )
        
        return result
    
    def _build_signature_cache(self, tree: ast.AST) -> None:
        """Build cache of function signatures and class methods."""
        self.function_signatures.clear()
        self.class_methods.clear()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract function signature
                args = [arg.arg for arg in node.args.args]
                if node.args.vararg:
                    args.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")
                
                self.function_signatures[node.name] = {
                    'args': args,
                    'arg_count': len(node.args.args),
                    'has_varargs': node.args.vararg is not None,
                    'has_kwargs': node.args.kwarg is not None
                }
                
            elif isinstance(node, ast.ClassDef):
                # Extract class methods
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                
                self.class_methods[node.name] = methods
    
    # Error pattern detection methods
    def _check_constant_assignment(self, tree: ast.AST, code_lines: List[str], focus_lines: Optional[Set[int]]) -> List[ErrorPattern]:
        """Check for assignments of constants instead of variables."""
        patterns = []
        
        class ConstantAssignmentChecker(ast.NodeVisitor):
            def visit_Assign(self, node):
                if focus_lines and node.lineno not in focus_lines:
                    return
                
                # Check if assigning a constant to a variable
                if (isinstance(node.value, ast.Constant) and 
                    len(node.targets) == 1 and 
                    isinstance(node.targets[0], ast.Name)):
                    
                    var_name = node.targets[0].id
                    constant_value = node.value.value
                    
                    # Check if this looks suspicious (e.g., assigning a number that might be a variable)
                    if isinstance(constant_value, (int, float)) and constant_value > 1:
                        pattern = ErrorPattern(
                            pattern_type="constant_assignment",
                            location=f"line {node.lineno}",
                            description=f"Assignment of constant {constant_value} to {var_name}",
                            severity="medium",
                            confidence=0.6,
                            suggested_fix=f"Verify if {constant_value} should be a variable reference",
                            code_context=code_lines[node.lineno - 1] if node.lineno <= len(code_lines) else ""
                        )
                        patterns.append(pattern)
                
                self.generic_visit(node)
        
        checker = ConstantAssignmentChecker()
        checker.visit(tree)
        return patterns
    
    def _check_wrong_comparison(self, tree: ast.AST, code_lines: List[str], focus_lines: Optional[Set[int]]) -> List[ErrorPattern]:
        """Check for assignment operators used instead of comparison."""
        patterns = []
        
        # This is better detected through regex on source code
        for i, line in enumerate(code_lines, 1):
            if focus_lines and i not in focus_lines:
                continue
                
            # Look for if statements with assignment (more precise pattern)
            # Match: if variable = value (but not if variable == value)
            if re.search(r'if\s+[^=]*\w+\s*=\s*[^=]', line.strip()):
                pattern = ErrorPattern(
                    pattern_type="wrong_comparison",
                    location=f"line {i}",
                    description="Possible assignment (=) instead of comparison (==) in if statement",
                    severity="high",
                    confidence=0.8,
                    suggested_fix="Change = to == for comparison",
                    code_context=line.strip()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _check_undefined_variable(self, tree: ast.AST, code_lines: List[str], focus_lines: Optional[Set[int]]) -> List[ErrorPattern]:
        """Check for usage of potentially undefined variables."""
        patterns = []
        defined_vars = set()
        
        class UndefinedVariableChecker(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack = [set()]  # Stack of variable scopes
                
            def visit_FunctionDef(self, node):
                # New function scope
                self.scope_stack.append(set())
                # Add function parameters to scope
                for arg in node.args.args:
                    self.scope_stack[-1].add(arg.arg)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_Assign(self, node):
                if focus_lines and node.lineno not in focus_lines:
                    return
                
                # Add assigned variables to current scope
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.scope_stack[-1].add(target.id)
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if focus_lines and node.lineno not in focus_lines:
                    return
                
                if isinstance(node.ctx, ast.Load):  # Variable is being read
                    # Check if variable is defined in any scope
                    is_defined = any(node.id in scope for scope in self.scope_stack)
                    
                    if not is_defined and node.id not in ['True', 'False', 'None']:
                        pattern = ErrorPattern(
                            pattern_type="undefined_variable",
                            location=f"line {node.lineno}",
                            description=f"Variable '{node.id}' may be undefined",
                            severity="high",
                            confidence=0.7,
                            suggested_fix=f"Define variable '{node.id}' before use",
                            code_context=code_lines[node.lineno - 1] if node.lineno <= len(code_lines) else ""
                        )
                        patterns.append(pattern)
        
        checker = UndefinedVariableChecker()
        checker.visit(tree)
        return patterns
    
    def _check_type_mismatch(self, tree: ast.AST, code_lines: List[str], focus_lines: Optional[Set[int]]) -> List[ErrorPattern]:
        """Check for potential type mismatches."""
        patterns = []
        
        class TypeMismatchChecker(ast.NodeVisitor):
            def visit_BinOp(self, node):
                if focus_lines and node.lineno not in focus_lines:
                    return
                
                # Check for string + number operations
                if isinstance(node.op, ast.Add):
                    left_is_str = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                    right_is_num = isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                    
                    if left_is_str and right_is_num:
                        pattern = ErrorPattern(
                            pattern_type="type_mismatch",
                            location=f"line {node.lineno}",
                            description="Potential type mismatch: string + number",
                            severity="medium",
                            confidence=0.8,
                            suggested_fix="Convert number to string or use proper concatenation",
                            code_context=code_lines[node.lineno - 1] if node.lineno <= len(code_lines) else ""
                        )
                        patterns.append(pattern)
                
                self.generic_visit(node)
        
        checker = TypeMismatchChecker()
        checker.visit(tree)
        return patterns
    
    def _check_unreachable_code(self, tree: ast.AST, code_lines: List[str], focus_lines: Optional[Set[int]]) -> List[ErrorPattern]:
        """Check for potentially unreachable code."""
        patterns = []
        
        class UnreachableCodeChecker(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if not node.body:
                    return
                
                # Check for code after return statements
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return):
                        # Check if there are more statements after this return
                        if i < len(node.body) - 1:
                            next_stmt = node.body[i + 1]
                            if not focus_lines or next_stmt.lineno in focus_lines:
                                pattern = ErrorPattern(
                                    pattern_type="unreachable_code",
                                    location=f"line {next_stmt.lineno}",
                                    description="Code after return statement may be unreachable",
                                    severity="low",
                                    confidence=0.6,
                                    suggested_fix="Remove unreachable code or restructure logic",
                                    code_context=code_lines[next_stmt.lineno - 1] if next_stmt.lineno <= len(code_lines) else ""
                                )
                                patterns.append(pattern)
                
                self.generic_visit(node)
        
        checker = UnreachableCodeChecker()
        checker.visit(tree)
        return patterns
    
    def _detect_syntax_error_patterns(self, code: str) -> List[ErrorPattern]:
        """Detect syntax error patterns when AST parsing fails."""
        patterns = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            pattern = ErrorPattern(
                pattern_type="syntax_error",
                location=f"line {e.lineno}" if e.lineno else "unknown",
                description=f"Syntax error: {e.msg}",
                severity="critical",
                confidence=0.9,
                suggested_fix="Fix syntax error",
                code_context=e.text or ""
            )
            patterns.append(pattern)
        
        return patterns
    
    def _extract_suspicious_regions_from_llm(self, llm_analysis: AnalysisResult) -> List[SuspiciousRegion]:
        """Extract suspicious regions from LLM analysis result."""
        regions = []
        
        # Try to extract line numbers from bug_location
        location_text = llm_analysis.bug_location
        
        # Look for line number patterns
        line_matches = re.findall(r'line\s*(\d+)', location_text, re.IGNORECASE)
        range_matches = re.findall(r'lines?\s*(\d+)\s*[-–]\s*(\d+)', location_text, re.IGNORECASE)
        
        if range_matches:
            for start_str, end_str in range_matches:
                region = SuspiciousRegion(
                    start_line=int(start_str),
                    end_line=int(end_str),
                    reason=llm_analysis.root_cause,
                    confidence=llm_analysis.confidence,
                    code_snippet=""
                )
                regions.append(region)
        elif line_matches:
            for line_str in line_matches:
                line_num = int(line_str)
                region = SuspiciousRegion(
                    start_line=line_num,
                    end_line=line_num,
                    reason=llm_analysis.root_cause,
                    confidence=llm_analysis.confidence,
                    code_snippet=""
                )
                regions.append(region)
        
        return regions
    
    def _extract_function_names_from_analysis(self, analysis: AnalysisResult) -> List[str]:
        """Extract function names mentioned in the analysis."""
        functions = []
        
        # Look for function names in the analysis text
        text = f"{analysis.root_cause} {analysis.fix_suggestion} {' '.join(analysis.supporting_evidence)}"
        
        # Simple pattern to find function names (word followed by parentheses)
        function_matches = re.findall(r'(\w+)\s*\(', text)
        functions.extend(function_matches)
        
        return list(set(functions))  # Remove duplicates
    
    def _extract_variable_names_from_analysis(self, analysis: AnalysisResult) -> List[str]:
        """Extract variable names mentioned in the analysis."""
        variables = []
        
        # Look for variable names in the analysis text
        text = f"{analysis.root_cause} {analysis.fix_suggestion} {' '.join(analysis.supporting_evidence)}"
        
        # Simple pattern to find potential variable names
        # This is a heuristic and may need refinement
        var_matches = re.findall(r'\b([a-z_][a-z0-9_]*)\b', text, re.IGNORECASE)
        
        # Filter out common words that are not variables
        common_words = {'the', 'and', 'or', 'not', 'in', 'is', 'to', 'of', 'for', 'with', 'by'}
        variables = [var for var in var_matches if var.lower() not in common_words and len(var) > 1]
        
        return list(set(variables))  # Remove duplicates
    
    def _extract_variables_from_region(self, region_code: str) -> List[str]:
        """Extract variable names from a code region."""
        variables = []
        
        try:
            tree = ast.parse(region_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    variables.append(node.id)
        except SyntaxError:
            # If parsing fails, use regex as fallback
            var_matches = re.findall(r'\b([a-z_][a-z0-9_]*)\b', region_code, re.IGNORECASE)
            variables.extend(var_matches)
        
        return list(set(variables))  # Remove duplicates
    
    def _generate_fix_suggestion(
        self, 
        error_patterns: List[ErrorPattern], 
        invalid_calls: List[FunctionCallValidation], 
        flow_issues: List[str]
    ) -> str:
        """Generate a fix suggestion based on detected issues."""
        suggestions = []
        
        if error_patterns:
            for pattern in error_patterns[:3]:  # Limit to top 3
                if pattern.suggested_fix:
                    suggestions.append(pattern.suggested_fix)
        
        if invalid_calls:
            for call in invalid_calls[:2]:  # Limit to top 2
                suggestions.append(f"Fix function call to {call.function_name}: {', '.join(call.issues)}")
        
        if flow_issues:
            suggestions.extend(flow_issues[:2])  # Limit to top 2
        
        if not suggestions:
            return "Review the code logic and structure"
        
        return "; ".join(suggestions)
    
    def _integrate_analysis_results(
        self,
        llm_analysis: AnalysisResult,
        ast_results: List[AnalysisResult],
        error_patterns: List[ErrorPattern],
        call_validations: List[FunctionCallValidation],
        data_flows: Dict[str, DataFlowTrace]
    ) -> AnalysisResult:
        """Integrate LLM and AST analysis results."""
        # Start with LLM analysis as base
        integrated = AnalysisResult(
            bug_location=llm_analysis.bug_location,
            root_cause=llm_analysis.root_cause,
            fix_suggestion=llm_analysis.fix_suggestion,
            confidence=llm_analysis.confidence,
            reasoning_chain=llm_analysis.reasoning_chain,
            supporting_evidence=llm_analysis.supporting_evidence.copy()
        )
        
        # Add AST analysis evidence
        if ast_results:
            integrated.supporting_evidence.append(f"AST analysis found {len(ast_results)} suspicious regions")
            for result in ast_results:
                integrated.supporting_evidence.extend(result.supporting_evidence)
        
        # Add error pattern evidence
        if error_patterns:
            integrated.supporting_evidence.append(f"Detected {len(error_patterns)} error patterns")
            for pattern in error_patterns:
                integrated.supporting_evidence.append(f"{pattern.pattern_type}: {pattern.description}")
        
        # Add function call validation evidence
        invalid_calls = [v for v in call_validations if not v.is_valid]
        if invalid_calls:
            integrated.supporting_evidence.append(f"Found {len(invalid_calls)} invalid function calls")
            for call in invalid_calls:
                integrated.supporting_evidence.append(f"Invalid call to {call.function_name}: {', '.join(call.issues)}")
        
        # Add data flow evidence
        flow_issues = []
        for var_name, trace in data_flows.items():
            flow_issues.extend(trace.potential_issues)
        
        if flow_issues:
            integrated.supporting_evidence.append(f"Data flow analysis found {len(flow_issues)} potential issues")
            integrated.supporting_evidence.extend(flow_issues)
        
        # Adjust confidence based on AST findings
        if error_patterns or invalid_calls or flow_issues:
            # AST analysis confirms issues, increase confidence slightly
            integrated.confidence = min(1.0, integrated.confidence * 1.1)
        elif ast_results:
            # AST analysis ran but found no major issues, slightly decrease confidence
            integrated.confidence = max(0.1, integrated.confidence * 0.95)
        else:
            # No AST analysis could be performed, reduce confidence more
            integrated.confidence = max(0.1, integrated.confidence * 0.8)
        
        return integrated


class FunctionCallValidator:
    """Helper class for validating function calls."""
    
    def __init__(self, function_signatures: Dict, class_methods: Dict, target_functions: Optional[List[str]] = None):
        self.function_signatures = function_signatures
        self.class_methods = class_methods
        self.target_functions = target_functions
    
    def validate_call(self, call_node: ast.Call) -> Optional[FunctionCallValidation]:
        """Validate a single function call node."""
        function_name = self._get_function_name(call_node)
        
        if not function_name:
            return None
        
        # Skip if we're only validating specific functions
        if self.target_functions and function_name not in self.target_functions:
            return None
        
        validation = FunctionCallValidation(
            function_name=function_name,
            call_location=f"line {call_node.lineno}",
            is_valid=True,
            actual_call=self._format_call(call_node)
        )
        
        # Check if we have signature information
        if function_name in self.function_signatures:
            signature = self.function_signatures[function_name]
            validation.expected_signature = self._format_signature(function_name, signature)
            
            # Validate argument count
            provided_args = len(call_node.args) + len(call_node.keywords)
            expected_args = signature['arg_count']
            
            if not signature['has_varargs'] and provided_args > expected_args:
                validation.is_valid = False
                validation.issues.append(f"Too many arguments: provided {provided_args}, expected {expected_args}")
            elif provided_args < expected_args:
                validation.is_valid = False
                validation.issues.append(f"Too few arguments: provided {provided_args}, expected {expected_args}")
        
        return validation
    
    def _get_function_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def _format_call(self, call_node: ast.Call) -> str:
        """Format the function call for display."""
        function_name = self._get_function_name(call_node)
        if not function_name:
            return "unknown()"
        
        args = [f"arg{i}" for i in range(len(call_node.args))]
        kwargs = [f"{kw.arg}=..." for kw in call_node.keywords]
        all_args = args + kwargs
        
        return f"{function_name}({', '.join(all_args)})"
    
    def _format_signature(self, function_name: str, signature: Dict) -> str:
        """Format function signature for display."""
        args = signature['args']
        return f"{function_name}({', '.join(args)})"


class DataFlowTracer:
    """Helper class for tracing variable data flow."""
    
    def __init__(self, key_variables: List[str], focus_region: Optional[Tuple[int, int]] = None):
        self.key_variables = key_variables
        self.focus_region = focus_region
    
    def trace_variable(self, tree: ast.AST, code_lines: List[str], var_name: str) -> DataFlowTrace:
        """Trace data flow for a specific variable."""
        trace = DataFlowTrace(variable_name=var_name)
        
        class VariableTracer(ast.NodeVisitor):
            def __init__(self, focus_region):
                self.focus_region = focus_region
                
            def visit_Assign(self, node):
                # Check if this assignment defines our variable
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if self._in_focus_region(node.lineno):
                            context = code_lines[node.lineno - 1] if node.lineno <= len(code_lines) else ""
                            trace.definition_points.append((node.lineno, context))
                
                self.generic_visit(node)
            
            def visit_AugAssign(self, node):
                # Check for augmented assignments (+=, -=, etc.)
                if isinstance(node.target, ast.Name) and node.target.id == var_name:
                    if self._in_focus_region(node.lineno):
                        context = code_lines[node.lineno - 1] if node.lineno <= len(code_lines) else ""
                        trace.modifications.append((node.lineno, context))
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if node.id == var_name and isinstance(node.ctx, ast.Load):
                    if self._in_focus_region(node.lineno):
                        context = code_lines[node.lineno - 1] if node.lineno <= len(code_lines) else ""
                        trace.usage_points.append((node.lineno, context))
            
            def _in_focus_region(self, line_num: int) -> bool:
                if not self.focus_region:
                    return True
                return self.focus_region[0] <= line_num <= self.focus_region[1]
        
        tracer = VariableTracer(self.focus_region)
        tracer.visit(tree)
        
        # Analyze for potential issues
        self._analyze_flow_issues(trace)
        
        return trace
    
    def _analyze_flow_issues(self, trace: DataFlowTrace) -> None:
        """Analyze data flow trace for potential issues."""
        # Check for usage before definition
        if trace.usage_points and trace.definition_points:
            first_usage = min(trace.usage_points, key=lambda x: x[0])
            first_definition = min(trace.definition_points, key=lambda x: x[0])
            
            if first_usage[0] < first_definition[0]:
                trace.potential_issues.append(
                    f"Variable {trace.variable_name} used before definition (line {first_usage[0]} before line {first_definition[0]})"
                )
        
        # Check for usage without definition
        if trace.usage_points and not trace.definition_points:
            trace.potential_issues.append(
                f"Variable {trace.variable_name} used but never defined in analyzed scope"
            )
        
        # Check for definition without usage
        if trace.definition_points and not trace.usage_points:
            trace.potential_issues.append(
                f"Variable {trace.variable_name} defined but never used"
            )
        
        # Build flow path
        all_points = []
        all_points.extend([(line, f"defined: {context}") for line, context in trace.definition_points])
        all_points.extend([(line, f"used: {context}") for line, context in trace.usage_points])
        all_points.extend([(line, f"modified: {context}") for line, context in trace.modifications])
        
        # Sort by line number
        all_points.sort(key=lambda x: x[0])
        trace.flow_path = [f"Line {line}: {desc}" for line, desc in all_points]