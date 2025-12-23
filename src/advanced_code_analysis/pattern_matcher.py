"""
Predefined Pattern Matching Engine for the Advanced Code Analysis system.

This module implements intelligent pattern matching for common bug types,
including assignment errors, parameter errors, type errors, and other
frequently occurring programming mistakes. It provides specialized
prompt templates and domain adaptation capabilities.
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from .models import (
    BugPattern, BugType, BugCategory, PromptTemplate, 
    PatternGuidance, AnalysisResult, ContextWindow
)
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface


logger = logging.getLogger(__name__)


@dataclass
class PatternRule:
    """Rule for detecting specific bug patterns in code."""
    rule_id: str
    name: str
    description: str
    pattern_regex: str
    bug_category: BugCategory
    confidence_weight: float = 1.0
    context_requirements: List[str] = field(default_factory=list)
    
    def matches(self, code: str) -> bool:
        """Check if the pattern matches the given code."""
        try:
            return bool(re.search(self.pattern_regex, code, re.MULTILINE | re.DOTALL))
        except re.error as e:
            logger.warning(f"Invalid regex pattern in rule {self.rule_id}: {e}")
            return False


@dataclass
class DomainPattern:
    """Domain-specific pattern for specialized code analysis."""
    domain_name: str
    keywords: Set[str] = field(default_factory=set)
    common_functions: Set[str] = field(default_factory=set)
    typical_imports: Set[str] = field(default_factory=set)
    error_patterns: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)


class PatternDetector(ABC):
    """Abstract base class for pattern detection strategies."""
    
    @abstractmethod
    def detect_patterns(self, code: str, context: ContextWindow) -> List[Tuple[PatternRule, float]]:
        """Detect patterns in code and return matches with confidence scores."""
        pass


class RegexPatternDetector(PatternDetector):
    """Pattern detector using regular expressions."""
    
    def __init__(self, rules: List[PatternRule]):
        self.rules = rules
    
    def detect_patterns(self, code: str, context: ContextWindow) -> List[Tuple[PatternRule, float]]:
        """Detect patterns using regex matching."""
        matches = []
        
        for rule in self.rules:
            if rule.matches(code):
                # Calculate confidence based on context requirements
                confidence = rule.confidence_weight
                
                # Adjust confidence based on context availability
                if rule.context_requirements:
                    available_context = 0
                    for req in rule.context_requirements:
                        if req in context.domain_concepts or any(req in func for func in context.related_functions):
                            available_context += 1
                    
                    context_ratio = available_context / len(rule.context_requirements)
                    confidence *= (0.5 + 0.5 * context_ratio)  # Scale between 0.5 and 1.0
                
                matches.append((rule, confidence))
        
        return matches


class SemanticPatternDetector(PatternDetector):
    """Pattern detector using LLM-based semantic analysis."""
    
    def __init__(self, llm_interface: LLMInterface, patterns: List[BugPattern]):
        self.llm_interface = llm_interface
        self.patterns = patterns
        self.logger = logging.getLogger(__name__)
    
    async def detect_patterns_async(self, code: str, context: ContextWindow) -> List[Tuple[PatternRule, float]]:
        """Detect patterns using semantic analysis (async version)."""
        # Create a prompt for semantic pattern detection
        prompt = self._create_semantic_detection_prompt(code, context)
        
        try:
            response = await self.llm_interface.generate(prompt, max_completion_tokens=1000)
            return self._parse_semantic_response(response.content)
        except Exception as e:
            logger.error(f"Semantic pattern detection failed: {e}")
            return []
    
    def detect_patterns(self, code: str, context: ContextWindow) -> List[Tuple[PatternRule, float]]:
        """Synchronous wrapper for semantic detection."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # Return empty list to avoid blocking
                self.logger.warning("Cannot run async semantic detection in running event loop")
                return []
            else:
                return loop.run_until_complete(self.detect_patterns_async(code, context))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.detect_patterns_async(code, context))
    
    def _create_semantic_detection_prompt(self, code: str, context: ContextWindow) -> str:
        """Create prompt for semantic pattern detection."""
        related_funcs = ', '.join(context.related_functions[:5])
        domain_concepts = ', '.join(context.domain_concepts[:5])
        
        return f"""
Analyze the following code for common bug patterns. Focus on semantic issues that might not be caught by syntax checkers.

Code to analyze:
```
{code}
```

Context information:
- Related functions: {related_funcs}
- Domain concepts: {domain_concepts}

Look for these types of patterns:
1. Assignment errors (assigning constants instead of variables)
2. Parameter mismatches (wrong number or type of arguments)
3. Logic errors (incorrect conditions, off-by-one errors)
4. Resource management issues (unclosed files, memory leaks)
5. Type-related errors (implicit conversions, null pointer issues)

For each pattern found, provide:
- Pattern type
- Confidence (0.0-1.0)
- Brief explanation
- Line numbers if applicable

Format your response as JSON:
{{
  "patterns": [
    {{
      "type": "assignment_error",
      "confidence": 0.8,
      "explanation": "Variable assigned constant value instead of computed result",
      "lines": [5, 6]
    }}
  ]
}}
"""
    
    def _parse_semantic_response(self, response: str) -> List[Tuple[PatternRule, float]]:
        """Parse LLM response for detected patterns."""
        response_text = (response or "").strip()
        if not response_text:
            logger.info("Semantic pattern response was empty")
            return []
        try:
            data = json.loads(response_text)
            patterns = []
            
            for pattern_data in data.get("patterns", []):
                # Create a temporary PatternRule for the detected pattern
                rule = PatternRule(
                    rule_id=f"semantic_{pattern_data['type']}",
                    name=pattern_data['type'].replace('_', ' ').title(),
                    description=pattern_data.get('explanation', ''),
                    pattern_regex="",  # Not used for semantic patterns
                    bug_category=self._map_pattern_to_category(pattern_data['type']),
                    confidence_weight=1.0
                )
                
                confidence = float(pattern_data.get('confidence', 0.5))
                patterns.append((rule, confidence))
            
            return patterns        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Try to salvage JSON block if the model wrapped it with text
            import re
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return [
                        (
                            PatternRule(
                                rule_id=f"semantic_{p.get('type','unknown')}",
                                name=p.get('type', 'unknown').replace('_', ' ').title(),
                                description=p.get('explanation', ''),
                                pattern_regex="",
                                bug_category=self._map_pattern_to_category(p.get('type', 'unknown')),
                                confidence_weight=1.0
                            ),
                            float(p.get('confidence', 0.5))
                        )
                        for p in data.get("patterns", [])
                    ]
                except Exception:
                    pass
            snippet = response_text[:200].replace("\n", " ")
            logger.warning(f"Failed to parse semantic pattern response: {e}; snippet: {snippet}")
            return []
    
    def _map_pattern_to_category(self, pattern_type: str) -> BugCategory:
        """Map pattern type to bug category."""
        mapping = {
            'assignment_error': BugCategory.LOGIC_ERROR,
            'parameter_error': BugCategory.API_ISSUE,
            'type_error': BugCategory.TYPE_ERROR,
            'resource_error': BugCategory.RESOURCE_MANAGEMENT,
            'logic_error': BugCategory.LOGIC_ERROR,
            'boundary_error': BugCategory.BOUNDARY_CONDITION,
        }
        return mapping.get(pattern_type, BugCategory.LOGIC_ERROR)


class PatternMatcher:
    """Main pattern matching engine for detecting predefined bug patterns."""
    
    def __init__(self, config: AdvancedAnalysisConfig, llm_interface: Optional[LLMInterface] = None):
        self.config = config
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)
        
        # Initialize pattern storage
        self.bug_patterns: Dict[str, BugPattern] = {}
        self.pattern_rules: List[PatternRule] = []
        self.domain_patterns: Dict[str, DomainPattern] = {}
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        
        # Initialize detectors
        self.regex_detector = RegexPatternDetector([])
        if llm_interface:
            self.semantic_detector = SemanticPatternDetector(llm_interface, [])
        else:
            self.semantic_detector = None
        
        # Load predefined patterns
        self._initialize_predefined_patterns()
        self._initialize_prompt_templates()
        self._load_stored_patterns()
    
    def _initialize_predefined_patterns(self):
        """Initialize common bug patterns and rules."""
        # Assignment error patterns
        self.pattern_rules.extend([
            PatternRule(
                rule_id="assignment_constant",
                name="Assignment to Constant",
                description="Variable assigned a constant value instead of computed result",
                pattern_regex=r'(\w+)\s*=\s*([0-9]+|"[^"]*"|\'[^\']*\')\s*$',
                bug_category=BugCategory.LOGIC_ERROR,
                confidence_weight=0.7
            ),
            PatternRule(
                rule_id="assignment_same_variable",
                name="Self Assignment",
                description="Variable assigned to itself",
                pattern_regex=r'(\w+)\s*=\s*\1\s*$',
                bug_category=BugCategory.LOGIC_ERROR,
                confidence_weight=0.9
            ),
            PatternRule(
                rule_id="comparison_assignment",
                name="Assignment in Comparison",
                description="Assignment operator used instead of comparison",
                pattern_regex=r'if\s*\(\s*\w+\s*=\s*[^=]',
                bug_category=BugCategory.LOGIC_ERROR,
                confidence_weight=0.8
            )
        ])
        
        # Parameter error patterns
        self.pattern_rules.extend([
            PatternRule(
                rule_id="wrong_parameter_count",
                name="Wrong Parameter Count",
                description="Function called with incorrect number of parameters",
                pattern_regex=r'(\w+)\s*\(\s*\)',
                bug_category=BugCategory.API_ISSUE,
                confidence_weight=0.6,
                context_requirements=["function_signature"]
            ),
            PatternRule(
                rule_id="parameter_type_mismatch",
                name="Parameter Type Mismatch",
                description="Parameter passed with potentially wrong type",
                pattern_regex=r'(\w+)\s*\(\s*"[^"]*"\s*\)',
                bug_category=BugCategory.TYPE_ERROR,
                confidence_weight=0.5,
                context_requirements=["function_signature", "type_info"]
            )
        ])
        
        # Boundary condition patterns
        self.pattern_rules.extend([
            PatternRule(
                rule_id="off_by_one_loop",
                name="Off-by-One in Loop",
                description="Potential off-by-one error in loop condition",
                pattern_regex=r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\([^)]+\)\s*\+\s*1\s*\)',
                bug_category=BugCategory.BOUNDARY_CONDITION,
                confidence_weight=0.7
            ),
            PatternRule(
                rule_id="array_bounds",
                name="Array Bounds Issue",
                description="Potential array bounds violation",
                pattern_regex=r'\w+\[\s*len\s*\([^)]+\)\s*\]',
                bug_category=BugCategory.BOUNDARY_CONDITION,
                confidence_weight=0.8
            )
        ])
        
        # Resource management patterns
        self.pattern_rules.extend([
            PatternRule(
                rule_id="unclosed_file",
                name="Unclosed File Handle",
                description="File opened but not properly closed",
                pattern_regex=r'open\s*\([^)]+\)(?!.*with\s)',
                bug_category=BugCategory.RESOURCE_MANAGEMENT,
                confidence_weight=0.6
            ),
            PatternRule(
                rule_id="memory_leak",
                name="Potential Memory Leak",
                description="Object created but not properly released",
                pattern_regex=r'new\s+\w+\s*\([^)]*\)(?!.*delete)',
                bug_category=BugCategory.RESOURCE_MANAGEMENT,
                confidence_weight=0.5
            )
        ])
        
        # Update regex detector with new rules
        self.regex_detector = RegexPatternDetector(self.pattern_rules)
    
    def _initialize_prompt_templates(self):
        """Initialize specialized prompt templates for different pattern types."""
        self.prompt_templates = {
            "assignment_error": PromptTemplate(
                template_id="assignment_error_analysis",
                content="""
Analyze the following code for assignment-related errors:

Code:
```
{code}
```

Focus on:
1. Variables assigned constant values when they should be computed
2. Self-assignments that don't change the variable
3. Assignment operators used instead of comparison operators
4. Incorrect variable assignments in loops or conditions

Context: {context}

Provide a detailed analysis of any assignment errors found, including:
- Exact location of the error
- What the code is currently doing wrong
- What the correct assignment should be
- Confidence level (0.0-1.0)
""",
                placeholders=["code", "context"]
            ),
            
            "parameter_error": PromptTemplate(
                template_id="parameter_error_analysis",
                content="""
Analyze the following code for parameter-related errors:

Code:
```
{code}
```

Function signatures available:
{function_signatures}

Focus on:
1. Functions called with wrong number of parameters
2. Parameters passed in wrong order
3. Type mismatches between expected and actual parameters
4. Missing required parameters or extra unnecessary parameters

Provide detailed analysis including:
- Function name and expected signature
- Actual call and what's wrong
- Suggested fix
- Confidence level (0.0-1.0)
""",
                placeholders=["code", "function_signatures"]
            ),
            
            "boundary_condition": PromptTemplate(
                template_id="boundary_condition_analysis",
                content="""
Analyze the following code for boundary condition errors:

Code:
```
{code}
```

Focus on:
1. Off-by-one errors in loops and array access
2. Incorrect range boundaries
3. Edge cases not properly handled
4. Index out of bounds issues

Context: {context}

Provide analysis including:
- Specific boundary issue identified
- Why it's problematic
- Correct boundary handling
- Test cases that would expose the issue
- Confidence level (0.0-1.0)
""",
                placeholders=["code", "context"]
            ),
            
            "resource_management": PromptTemplate(
                template_id="resource_management_analysis",
                content="""
Analyze the following code for resource management issues:

Code:
```
{code}
```

Focus on:
1. Files, connections, or handles not properly closed
2. Memory allocations without corresponding deallocations
3. Resources acquired but not released in error paths
4. Missing try-finally or context manager usage

Context: {context}

Provide analysis including:
- Resource management issue identified
- Potential consequences (memory leaks, file handle exhaustion, etc.)
- Proper resource management pattern
- Confidence level (0.0-1.0)
""",
                placeholders=["code", "context"]
            )
        }
    
    def _load_stored_patterns(self):
        """Load previously stored bug patterns from disk."""
        patterns_file = Path(self.config.storage.patterns_db_path)
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                
                for pattern_data in patterns_data.get('patterns', []):
                    pattern = BugPattern(**pattern_data)
                    self.bug_patterns[pattern.pattern_id] = pattern
                
                self.logger.info(f"Loaded {len(self.bug_patterns)} stored patterns")
            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.warning(f"Failed to load stored patterns: {e}")
    
    def detect_patterns(self, code: str, context: ContextWindow) -> List[Tuple[PatternRule, float]]:
        """Detect all matching patterns in the given code."""
        all_matches = []
        
        # Use regex-based detection
        regex_matches = self.regex_detector.detect_patterns(code, context)
        all_matches.extend(regex_matches)
        
        # Use semantic detection if available
        if self.semantic_detector:
            try:
                semantic_matches = self.semantic_detector.detect_patterns(code, context)
                all_matches.extend(semantic_matches)
            except Exception as e:
                self.logger.warning(f"Semantic pattern detection failed: {e}")
        
        # Sort by confidence (highest first)
        all_matches.sort(key=lambda x: x[1], reverse=True)
        
        return all_matches
    
    def match_bug_patterns(self, issue_text: str, code: str) -> List[BugPattern]:
        """Match issue description against stored bug patterns."""
        matched_patterns = []
        
        # Simple text similarity matching
        issue_words = set(issue_text.lower().split())
        
        for pattern in self.bug_patterns.values():
            # Calculate similarity based on problem signature
            pattern_words = set(pattern.problem_signature.lower().split())
            
            # Avoid division by zero
            if len(issue_words | pattern_words) == 0:
                similarity = 0.0
            else:
                similarity = len(issue_words & pattern_words) / len(issue_words | pattern_words)
            
            if similarity >= self.config.analysis.pattern_similarity_threshold:
                matched_patterns.append(pattern)
        
        # Sort by success rate and similarity
        matched_patterns.sort(key=lambda p: p.success_rate, reverse=True)
        
        return matched_patterns
    
    def get_specialized_prompt(self, bug_category: BugCategory, **template_vars) -> Optional[str]:
        """Get specialized prompt template for the given bug category."""
        template_map = {
            BugCategory.LOGIC_ERROR: "assignment_error",
            BugCategory.API_ISSUE: "parameter_error",
            BugCategory.BOUNDARY_CONDITION: "boundary_condition",
            BugCategory.RESOURCE_MANAGEMENT: "resource_management",
        }
        
        template_id = template_map.get(bug_category)
        if not template_id or template_id not in self.prompt_templates:
            return None
        
        template = self.prompt_templates[template_id]
        
        try:
            return template.content.format(**template_vars)
        except KeyError as e:
            self.logger.warning(f"Missing template variable {e} for {template_id}")
            return None
    
    def adapt_to_domain(self, code: str, context: ContextWindow) -> DomainPattern:
        """Quickly adapt to new domain by analyzing code patterns and terminology."""
        # Extract domain characteristics from code and context
        domain_keywords = set()
        common_functions = set()
        typical_imports = set()
        
        # Analyze imports
        import_lines = [line.strip() for line in code.split('\n') 
                       if line.strip().startswith(('import ', 'from '))]
        for line in import_lines:
            # Extract module names
            if line.startswith('import '):
                modules = line[7:].split(',')
                for mod in modules:
                    # Handle "import pandas as pd" -> extract "pandas"
                    mod_clean = mod.strip().split(' as ')[0].split('.')[0]
                    typical_imports.add(mod_clean)
            elif line.startswith('from '):
                parts = line.split()
                if len(parts) >= 2:
                    module = parts[1]
                    typical_imports.add(module.split('.')[0])
        
        # Extract function names and keywords from context
        if context.related_functions:
            common_functions.update(context.related_functions)
        
        if context.domain_concepts:
            domain_keywords.update(context.domain_concepts)
        
        # Analyze code for common patterns
        function_pattern = re.compile(r'def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'class\s+(\w+)\s*[:\(]')
        
        functions = function_pattern.findall(code)
        classes = class_pattern.findall(code)
        
        common_functions.update(functions)
        domain_keywords.update(classes)
        
        # Determine domain name based on imports and keywords
        domain_name = self._infer_domain_name(typical_imports, domain_keywords)
        
        return DomainPattern(
            domain_name=domain_name,
            keywords=domain_keywords,
            common_functions=common_functions,
            typical_imports=typical_imports,
            error_patterns=[],  # Will be populated as patterns are learned
            best_practices=[]   # Will be populated as patterns are learned
        )
    
    def _infer_domain_name(self, imports: Set[str], keywords: Set[str]) -> str:
        """Infer domain name from imports and keywords."""
        # Common domain indicators
        domain_indicators = {
            'web': {'flask', 'django', 'fastapi', 'requests', 'http', 'url', 'api'},
            'data_science': {'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'scipy'},
            'machine_learning': {'tensorflow', 'torch', 'keras', 'model', 'train', 'predict'},
            'database': {'sqlite3', 'psycopg2', 'pymongo', 'sqlalchemy', 'database', 'query'},
            'gui': {'tkinter', 'pyqt', 'kivy', 'window', 'button', 'widget'},
            'testing': {'pytest', 'unittest', 'test', 'mock', 'assert'},
            'networking': {'socket', 'asyncio', 'aiohttp', 'network', 'client', 'server'},
        }
        
        all_terms = imports | keywords
        
        for domain, indicators in domain_indicators.items():
            if indicators & all_terms:
                return domain
        
        return 'general'
    
    def store_pattern(self, pattern: BugPattern):
        """Store a new bug pattern for future use."""
        self.bug_patterns[pattern.pattern_id] = pattern
        
        # Save to disk
        self._save_patterns_to_disk()
        
        self.logger.info(f"Stored new pattern: {pattern.pattern_id}")
    
    def _save_patterns_to_disk(self):
        """Save current patterns to disk."""
        patterns_file = Path(self.config.storage.patterns_db_path)
        patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            patterns_data = {
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'problem_signature': p.problem_signature,
                        'code_pattern': p.code_pattern,
                        'fix_pattern': p.fix_pattern,
                        'success_rate': p.success_rate,
                        'applicable_domains': p.applicable_domains,
                        'usage_count': p.usage_count,
                        'last_updated': p.last_updated,
                    }
                    for p in self.bug_patterns.values()
                ]
            }
            
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            self.logger.error(f"Failed to save patterns to disk: {e}")
    
    async def match_patterns(self, issue_text: str, context: ContextWindow, 
                           bug_type: BugType) -> List[Any]:
        """
        Match patterns against issue text and code context.
        
        Args:
            issue_text: The problem description
            context: Code context window
            bug_type: Classified bug type
            
        Returns:
            List of pattern matches with guidance
        """
        try:
            # Detect patterns in the code (run regex sync, semantic async to avoid event-loop warnings)
            detected_rules: List[Tuple[PatternRule, float]] = []
            regex_matches = self.regex_detector.detect_patterns(context.target_code, context)
            detected_rules.extend(regex_matches)

            if self.semantic_detector:
                try:
                    semantic_matches = await self.semantic_detector.detect_patterns_async(
                        context.target_code, context
                    )
                    detected_rules.extend(semantic_matches)
                except Exception as e:
                    self.logger.warning(f"Semantic pattern detection failed: {e}")

            detected_rules.sort(key=lambda x: x[1], reverse=True)
            
            # Match against stored bug patterns
            matched_patterns = self.match_bug_patterns(issue_text, context.target_code)
            
            # Create pattern guidance
            guidance = self.create_pattern_guidance(matched_patterns, detected_rules)
            
            # Get specialized prompt with required variables
            specialized_prompt = self.get_specialized_prompt(
                bug_type.category,
                code=context.target_code,
                context=f"Issue: {issue_text}",
                function_signatures="\n".join(context.related_functions[:5]) if context.related_functions else "No function signatures available"
            )
            
            # Return combined results
            return {
                'detected_rules': detected_rules,
                'matched_patterns': matched_patterns,
                'guidance': guidance,
                'specialized_prompt': specialized_prompt
            }
            
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {e}")
            return []

    def create_pattern_guidance(self, matched_patterns: List[BugPattern], 
                              detected_rules: List[Tuple[PatternRule, float]]) -> PatternGuidance:
        """Create guidance based on matched patterns and detected rules."""
        if not matched_patterns and not detected_rules:
            return PatternGuidance(confidence=0.0, suggested_approach="No patterns matched")
        
        # Calculate overall confidence
        pattern_confidence = sum(p.success_rate for p in matched_patterns) / len(matched_patterns) if matched_patterns else 0.0
        rule_confidence = sum(conf for _, conf in detected_rules) / len(detected_rules) if detected_rules else 0.0
        
        overall_confidence = (pattern_confidence + rule_confidence) / 2.0
        
        # Create suggested approach
        approaches = []
        if matched_patterns:
            approaches.append(f"Apply learned patterns: {', '.join(p.pattern_id for p in matched_patterns[:3])}")
        
        if detected_rules:
            top_rules = [rule.name for rule, _ in detected_rules[:3]]
            approaches.append(f"Check for: {', '.join(top_rules)}")
        
        suggested_approach = "; ".join(approaches)
        
        # Collect relevant context
        relevant_context = []
        for pattern in matched_patterns:
            relevant_context.extend(pattern.applicable_domains)
        
        for rule, _ in detected_rules:
            relevant_context.extend(rule.context_requirements)
        
        return PatternGuidance(
            matched_patterns=matched_patterns,
            confidence=overall_confidence,
            suggested_approach=suggested_approach,
            relevant_context=list(set(relevant_context))
        )
    
    def cleanup_patterns(self):
        """Clean up old or low-performing patterns."""
        if len(self.bug_patterns) <= self.config.analysis.max_stored_patterns:
            return
        
        # Sort patterns by success rate and usage count
        patterns_list = list(self.bug_patterns.values())
        patterns_list.sort(key=lambda p: (p.success_rate, p.usage_count), reverse=True)
        
        # Keep only the top patterns
        keep_count = int(self.config.analysis.max_stored_patterns * 0.8)  # Keep 80% of max
        patterns_to_keep = patterns_list[:keep_count]
        
        # Update storage
        self.bug_patterns = {p.pattern_id: p for p in patterns_to_keep}
        self._save_patterns_to_disk()
        
        removed_count = len(patterns_list) - keep_count
        self.logger.info(f"Cleaned up {removed_count} low-performing patterns")
