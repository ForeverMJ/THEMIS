"""
Multi-level Concept Mapping Engine for the Advanced Code Analysis system.

This module implements the ConceptMapper class that performs multi-level semantic matching,
hierarchical search strategies, alternative solution generation, and reasoning path construction
to map problem descriptions to specific code locations.
"""

import ast
import re
import difflib
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict
import math

from .models import (
    ContextWindow, 
    AnalysisResult, 
    ReasoningChain, 
    ReasoningStep,
    EvidenceChain,
    BugType
)
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface, LLMResponse
from .context_enhancer import ContextEnhancer, CodeContext


logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Represents a matching result with confidence and evidence."""
    element_name: str
    element_type: str  # function, class, variable, module
    file_path: str
    line_number: int
    match_type: str  # exact, fuzzy, conceptual
    confidence: float
    evidence: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    context_snippet: str = ""


@dataclass
class SearchCandidate:
    """Represents a candidate location for bug fixing."""
    location: str
    description: str
    confidence: float
    reasoning: str
    code_snippet: str
    impact_analysis: str = ""
    similarity_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConceptMappingResult:
    """Result of concept mapping with multiple candidates and reasoning."""
    primary_matches: List[MatchResult] = field(default_factory=list)
    alternative_candidates: List[SearchCandidate] = field(default_factory=list)
    reasoning_chain: ReasoningChain = field(default_factory=ReasoningChain)
    evidence_chain: EvidenceChain = field(default_factory=EvidenceChain)
    overall_confidence: float = 0.0
    search_strategy_used: str = ""


class ConceptMapper:
    """
    Multi-level Concept Mapping Engine that performs semantic matching and hierarchical search.
    
    This class implements:
    1. Multi-level semantic matching (exact, fuzzy, conceptual)
    2. Hierarchical search strategies (module -> class -> function)
    3. Alternative solution generation through code similarity analysis
    4. Reasoning path and evidence chain construction
    """
    
    def __init__(self, config: Optional[AdvancedAnalysisConfig] = None,
                 llm_interface: Optional[LLMInterface] = None,
                 context_enhancer: Optional[ContextEnhancer] = None):
        """Initialize the Concept Mapper."""
        self.config = config or AdvancedAnalysisConfig()
        self.llm_interface = llm_interface or LLMInterface(self.config.llm)
        self.context_enhancer = context_enhancer or ContextEnhancer(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Cache for code analysis results
        self._code_analysis_cache: Dict[str, Any] = {}
        
        # Similarity thresholds for different match types
        self.exact_match_threshold = 1.0
        self.fuzzy_match_threshold = 0.8
        self.conceptual_match_threshold = 0.6
        
        self.logger.info("ConceptMapper initialized")
    
    async def map_concepts_to_code(self, issue_description: str, 
                                 extracted_concepts: List[str],
                                 context: ContextWindow,
                                 bug_type: Optional[BugType] = None) -> ConceptMappingResult:
        """
        Map extracted concepts from issue description to specific code locations.
        
        Args:
            issue_description: Original issue description
            extracted_concepts: List of technical concepts extracted from the issue
            context: Code context window
            bug_type: Optional bug type for strategy selection
            
        Returns:
            ConceptMappingResult containing matches and reasoning
        """
        self.logger.info(f"Starting concept mapping for {len(extracted_concepts)} concepts")
        
        try:
            # Initialize result
            result = ConceptMappingResult()
            result.reasoning_chain = ReasoningChain()
            result.evidence_chain = EvidenceChain()
            
            # Step 1: Perform multi-level semantic matching
            step1 = ReasoningStep(
                step_number=1,
                description="Performing multi-level semantic matching",
                input_data={"concepts": extracted_concepts, "context_size": len(context.target_code)}
            )
            
            primary_matches = await self._perform_semantic_matching(
                extracted_concepts, context, issue_description
            )
            
            step1.output_data = {"primary_matches": len(primary_matches)}
            step1.confidence = self._calculate_matching_confidence(primary_matches)
            result.reasoning_chain.add_step(step1)
            result.primary_matches = primary_matches
            
            # Step 2: Apply hierarchical search strategy
            step2 = ReasoningStep(
                step_number=2,
                description="Applying hierarchical search strategy",
                input_data={"search_scope": "module->class->function"}
            )
            
            hierarchical_matches = await self._apply_hierarchical_search(
                extracted_concepts, context, primary_matches
            )
            
            step2.output_data = {"hierarchical_matches": len(hierarchical_matches)}
            step2.confidence = self._calculate_matching_confidence(hierarchical_matches)
            result.reasoning_chain.add_step(step2)
            
            # Merge hierarchical matches with primary matches
            result.primary_matches.extend(hierarchical_matches)
            result.primary_matches = self._deduplicate_matches(result.primary_matches)
            
            # Step 3: Generate alternative candidates
            step3 = ReasoningStep(
                step_number=3,
                description="Generating alternative candidates through similarity analysis",
                input_data={"base_matches": len(result.primary_matches)}
            )
            
            alternatives = await self._generate_alternative_candidates(
                issue_description, extracted_concepts, context, result.primary_matches
            )
            
            step3.output_data = {"alternatives_generated": len(alternatives)}
            step3.confidence = self._calculate_alternatives_confidence(alternatives)
            result.reasoning_chain.add_step(step3)
            result.alternative_candidates = alternatives
            
            # Step 4: Build evidence chain and reasoning path
            step4 = ReasoningStep(
                step_number=4,
                description="Building evidence chain and reasoning path",
                input_data={"total_candidates": len(result.primary_matches) + len(alternatives)}
            )
            
            await self._build_evidence_chain(result, issue_description, extracted_concepts)
            
            step4.output_data = {"evidence_items": len(result.evidence_chain.evidence_items)}
            step4.confidence = 0.9  # High confidence in evidence construction
            result.reasoning_chain.add_step(step4)
            
            # Calculate overall confidence
            result.overall_confidence = result.reasoning_chain.calculate_overall_confidence()
            result.search_strategy_used = "multi_level_hierarchical"
            
            self.logger.info(f"Concept mapping completed: {len(result.primary_matches)} primary matches, "
                           f"{len(result.alternative_candidates)} alternatives, "
                           f"confidence: {result.overall_confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in concept mapping: {e}")
            return ConceptMappingResult()
    
    async def _perform_semantic_matching(self, concepts: List[str], 
                                       context: ContextWindow,
                                       issue_description: str) -> List[MatchResult]:
        """Perform multi-level semantic matching (exact, fuzzy, conceptual)."""
        self.logger.debug("Performing semantic matching")
        
        matches = []
        
        try:
            # Extract code elements from context
            code_elements = self._extract_code_elements(context)
            
            for concept in concepts:
                concept_lower = concept.lower()
                
                # Level 1: Exact matching
                exact_matches = self._find_exact_matches(concept, code_elements)
                matches.extend(exact_matches)
                
                # Level 2: Fuzzy matching
                fuzzy_matches = self._find_fuzzy_matches(concept, code_elements)
                matches.extend(fuzzy_matches)
                
                # Level 3: Conceptual matching using LLM
                conceptual_matches = await self._find_conceptual_matches(
                    concept, code_elements, issue_description
                )
                matches.extend(conceptual_matches)
            
            # Sort by confidence and remove duplicates
            matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
            matches = self._deduplicate_matches(matches)
            
            # Limit to top matches
            max_matches = 20
            if len(matches) > max_matches:
                matches = matches[:max_matches]
            
            self.logger.debug(f"Semantic matching found {len(matches)} matches")
            return matches
            
        except Exception as e:
            self.logger.error(f"Error in semantic matching: {e}")
            return []
    
    async def _apply_hierarchical_search(self, concepts: List[str], 
                                       context: ContextWindow,
                                       existing_matches: List[MatchResult]) -> List[MatchResult]:
        """Apply hierarchical search strategy from module to function level."""
        self.logger.debug("Applying hierarchical search strategy")
        
        hierarchical_matches = []
        
        try:
            # Get existing match locations to avoid duplication
            existing_locations = {(m.file_path, m.element_name) for m in existing_matches}
            
            # Level 1: Module-level search
            module_matches = await self._search_at_module_level(concepts, context)
            for match in module_matches:
                if (match.file_path, match.element_name) not in existing_locations:
                    hierarchical_matches.append(match)
            
            # Level 2: Class-level search
            class_matches = await self._search_at_class_level(concepts, context)
            for match in class_matches:
                if (match.file_path, match.element_name) not in existing_locations:
                    hierarchical_matches.append(match)
            
            # Level 3: Function-level search
            function_matches = await self._search_at_function_level(concepts, context)
            for match in function_matches:
                if (match.file_path, match.element_name) not in existing_locations:
                    hierarchical_matches.append(match)
            
            # Sort by confidence
            hierarchical_matches = sorted(hierarchical_matches, key=lambda x: x.confidence, reverse=True)
            
            self.logger.debug(f"Hierarchical search found {len(hierarchical_matches)} additional matches")
            return hierarchical_matches
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical search: {e}")
            return []
    
    async def _generate_alternative_candidates(self, issue_description: str,
                                             concepts: List[str],
                                             context: ContextWindow,
                                             primary_matches: List[MatchResult]) -> List[SearchCandidate]:
        """Generate alternative candidates through code similarity analysis."""
        self.logger.debug("Generating alternative candidates")
        
        candidates = []
        
        try:
            # If we have good primary matches, generate fewer alternatives
            if primary_matches and max(m.confidence for m in primary_matches) > 0.8:
                max_alternatives = 5
            else:
                max_alternatives = 10
            
            # Strategy 1: Code similarity analysis
            similarity_candidates = await self._find_similar_code_patterns(
                issue_description, concepts, context
            )
            candidates.extend(similarity_candidates[:max_alternatives//2])
            
            # Strategy 2: Call relationship analysis
            call_candidates = await self._analyze_call_relationships(
                concepts, context, primary_matches
            )
            candidates.extend(call_candidates[:max_alternatives//2])
            
            # Strategy 3: Error pattern matching
            error_candidates = await self._match_error_patterns(
                issue_description, context
            )
            candidates.extend(error_candidates[:3])
            
            # Remove duplicates and sort by confidence
            candidates = self._deduplicate_candidates(candidates)
            candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
            
            # Limit total alternatives
            if len(candidates) > max_alternatives:
                candidates = candidates[:max_alternatives]
            
            self.logger.debug(f"Generated {len(candidates)} alternative candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error generating alternatives: {e}")
            return []
    
    async def _build_evidence_chain(self, result: ConceptMappingResult,
                                  issue_description: str,
                                  concepts: List[str]) -> None:
        """Build evidence chain and reasoning path for the mapping result."""
        self.logger.debug("Building evidence chain")
        
        try:
            # Add evidence from primary matches
            for match in result.primary_matches:
                evidence_text = f"Found {match.match_type} match for '{match.element_name}' " \
                              f"in {match.file_path}:{match.line_number}"
                result.evidence_chain.add_evidence(
                    evidence_text, 
                    f"{match.file_path}:{match.line_number}",
                    match.confidence
                )
                
                # Add match-specific evidence
                for evidence_item in match.evidence:
                    result.evidence_chain.add_evidence(
                        evidence_item,
                        f"{match.file_path}:{match.line_number}",
                        match.confidence * 0.8
                    )
            
            # Add evidence from alternative candidates
            for candidate in result.alternative_candidates:
                evidence_text = f"Alternative candidate: {candidate.description}"
                result.evidence_chain.add_evidence(
                    evidence_text,
                    candidate.location,
                    candidate.confidence * 0.6  # Lower weight for alternatives
                )
            
            # Build reasoning path
            reasoning_path = []
            reasoning_path.append(f"Analyzed {len(concepts)} extracted concepts from issue description")
            reasoning_path.append(f"Performed multi-level semantic matching (exact, fuzzy, conceptual)")
            reasoning_path.append(f"Applied hierarchical search strategy (module->class->function)")
            reasoning_path.append(f"Generated {len(result.alternative_candidates)} alternative candidates")
            reasoning_path.append(f"Built evidence chain with {len(result.evidence_chain.evidence_items)} items")
            
            result.evidence_chain.reasoning_path = reasoning_path
            
        except Exception as e:
            self.logger.error(f"Error building evidence chain: {e}")
    
    # Helper methods for semantic matching
    
    def _extract_code_elements(self, context: ContextWindow) -> Dict[str, List[Dict[str, Any]]]:
        """Extract code elements from context for matching."""
        elements = {
            'functions': [],
            'classes': [],
            'variables': [],
            'modules': []
        }
        
        try:
            # Parse target code to extract elements
            if context.target_code:
                try:
                    tree = ast.parse(context.target_code)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            elements['functions'].append({
                                'name': node.name,
                                'line': getattr(node, 'lineno', 0),
                                'docstring': ast.get_docstring(node) or "",
                                'args': [arg.arg for arg in node.args.args]
                            })
                        elif isinstance(node, ast.ClassDef):
                            elements['classes'].append({
                                'name': node.name,
                                'line': getattr(node, 'lineno', 0),
                                'docstring': ast.get_docstring(node) or "",
                                'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                            })
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    elements['variables'].append({
                                        'name': target.id,
                                        'line': getattr(node, 'lineno', 0),
                                        'context': ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                                    })
                
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error parsing target code: {e}")
            
            # Add elements from related functions
            for func_sig in context.related_functions:
                func_name = func_sig.split('(')[0] if '(' in func_sig else func_sig
                elements['functions'].append({
                    'name': func_name,
                    'line': 0,
                    'docstring': "",
                    'signature': func_sig
                })
            
            # Add elements from class hierarchy
            for class_name, bases in context.class_hierarchy.items():
                elements['classes'].append({
                    'name': class_name,
                    'line': 0,
                    'docstring': "",
                    'bases': bases
                })
            
            # Add module dependencies
            for module in context.module_dependencies:
                elements['modules'].append({
                    'name': module,
                    'line': 0,
                    'type': 'import'
                })
            
        except Exception as e:
            self.logger.error(f"Error extracting code elements: {e}")
        
        return elements
    
    def _find_exact_matches(self, concept: str, code_elements: Dict[str, List[Dict[str, Any]]]) -> List[MatchResult]:
        """Find exact matches for a concept."""
        matches = []
        concept_lower = concept.lower()
        
        for element_type, elements in code_elements.items():
            for element in elements:
                element_name = element['name']
                if element_name.lower() == concept_lower:
                    match = MatchResult(
                        element_name=element_name,
                        element_type=element_type[:-1],  # Remove 's' from plural
                        file_path="target_code",
                        line_number=element.get('line', 0),
                        match_type="exact",
                        confidence=1.0,
                        similarity_score=1.0,
                        evidence=[f"Exact name match: '{concept}' == '{element_name}'"]
                    )
                    matches.append(match)
        
        return matches
    
    def _find_fuzzy_matches(self, concept: str, code_elements: Dict[str, List[Dict[str, Any]]]) -> List[MatchResult]:
        """Find fuzzy matches for a concept using string similarity."""
        matches = []
        concept_lower = concept.lower()
        
        for element_type, elements in code_elements.items():
            for element in elements:
                element_name = element['name']
                element_lower = element_name.lower()
                
                # Calculate similarity using different metrics
                similarity_scores = []
                
                # Sequence matcher similarity
                seq_similarity = difflib.SequenceMatcher(None, concept_lower, element_lower).ratio()
                similarity_scores.append(seq_similarity)
                
                # Substring matching
                if concept_lower in element_lower or element_lower in concept_lower:
                    substring_similarity = min(len(concept_lower), len(element_lower)) / max(len(concept_lower), len(element_lower))
                    similarity_scores.append(substring_similarity)
                
                # Word-based similarity (for camelCase/snake_case)
                concept_words = re.findall(r'[A-Z][a-z]*|[a-z]+', concept)
                element_words = re.findall(r'[A-Z][a-z]*|[a-z]+', element_name)
                
                if concept_words and element_words:
                    word_matches = sum(1 for cw in concept_words for ew in element_words if cw.lower() == ew.lower())
                    word_similarity = word_matches / max(len(concept_words), len(element_words))
                    similarity_scores.append(word_similarity)
                
                # Use maximum similarity score
                max_similarity = max(similarity_scores) if similarity_scores else 0.0
                
                # Lower threshold for fuzzy matching to be more inclusive
                if max_similarity >= 0.3:  # Lower threshold than self.fuzzy_match_threshold
                    # But adjust confidence based on actual similarity
                    confidence = max_similarity if max_similarity >= self.fuzzy_match_threshold else max_similarity * 0.8
                    
                    match = MatchResult(
                        element_name=element_name,
                        element_type=element_type[:-1],
                        file_path="target_code",
                        line_number=element.get('line', 0),
                        match_type="fuzzy",
                        confidence=confidence,
                        similarity_score=max_similarity,
                        evidence=[f"Fuzzy match: '{concept}' ~ '{element_name}' (similarity: {max_similarity:.2f})"]
                    )
                    matches.append(match)
        
        return matches
    
    async def _find_conceptual_matches(self, concept: str, 
                                     code_elements: Dict[str, List[Dict[str, Any]]],
                                     issue_description: str) -> List[MatchResult]:
        """Find conceptual matches using LLM semantic understanding."""
        matches = []
        
        try:
            # Prepare elements for LLM analysis
            element_descriptions = []
            for element_type, elements in code_elements.items():
                for element in elements:
                    description = f"{element_type[:-1]}: {element['name']}"
                    if element.get('docstring'):
                        description += f" - {element['docstring'][:100]}"
                    if element.get('signature'):
                        description += f" ({element['signature']})"
                    element_descriptions.append((element, description))
            
            if not element_descriptions:
                return matches
            
            # Limit elements to avoid token limits
            max_elements = 15  # Reduced from 20 to avoid long responses
            if len(element_descriptions) > max_elements:
                element_descriptions = element_descriptions[:max_elements]
            
            # Create LLM prompt for conceptual matching with stricter JSON requirements
            elements_text = "\n".join([f"{i+1}. {desc}" for i, (_, desc) in enumerate(element_descriptions)])
            
            prompt = f"""
Given the following issue description and concept, identify which code elements are conceptually related.

Issue: {issue_description[:400]}
Concept to match: "{concept}"

Available code elements:
{elements_text}

For each element that is conceptually related to the concept "{concept}", provide:
1. Element number (1-{len(element_descriptions)})
2. Confidence score (0.0-1.0)
3. Brief explanation (one short sentence, no line breaks)

IMPORTANT: Return ONLY valid JSON. Keep explanations short and on one line.

Format your response as JSON:
{{
    "matches": [
        {{
            "element_number": 1,
            "confidence": 0.8,
            "explanation": "Brief explanation here"
        }}
    ]
}}

If no matches found, return: {{"matches": []}}
"""
            
            # Get LLM response with increased token limit
            response = await self.llm_interface.generate(prompt, max_completion_tokens=1500)
            
            # Parse LLM response (robust to empty / fenced / non-JSON outputs)
            try:
                import json
                raw = (response.content or "").strip()
                
                # If content looks like an object repr (e.g., "Response(...)"), drop it so we can try raw_response
                if raw.startswith("Response(") or raw.startswith("Response"):
                    raw = ""
                
                if not raw and response.raw_response:
                    # Try to extract text from raw_response (responses API output)
                    try:
                        output = response.raw_response.get("output") if isinstance(response.raw_response, dict) else None
                        if output:
                            text_parts = []
                            for item in output:
                                for content in item.get("content", []) or []:
                                    text_val = content.get("text") if isinstance(content, dict) else None
                                    if text_val:
                                        text_parts.append(text_val)
                            if text_parts:
                                raw = "\n".join(text_parts).strip()
                    except Exception:
                        pass
                
                # Remove markdown code fences
                if raw.startswith("```"):
                    raw = raw.strip("`").strip()
                    if raw.lower().startswith("json"):
                        raw = raw[len("json"):].lstrip()
                
                if not raw:
                    self.logger.warning("Empty LLM response for conceptual matching; skipping.")
                    return matches
                
                # Try to parse JSON
                try:
                    result_data = json.loads(raw)
                except json.JSONDecodeError as e:
                    # Try to salvage JSON block from mixed output
                    import re
                    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                    if json_match:
                        try:
                            result_data = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            # Try to fix common JSON errors
                            json_str = json_match.group(0)
                            # Fix trailing commas
                            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            # Try again
                            try:
                                result_data = json.loads(json_str)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Could not parse JSON even after cleanup: {e}")
                                return matches
                    else:
                        self.logger.warning(f"No JSON object found in response: {raw[:200]}")
                        return matches
                
                for match_data in result_data.get('matches', []):
                    element_num = match_data.get('element_number', 0) - 1
                    confidence = match_data.get('confidence', 0.0)
                    explanation = match_data.get('explanation', '')
                    
                    if 0 <= element_num < len(element_descriptions) and confidence >= self.conceptual_match_threshold:
                        element, _ = element_descriptions[element_num]
                        
                        match = MatchResult(
                            element_name=element['name'],
                            element_type=list(code_elements.keys())[0][:-1],  # Simplified
                            file_path="target_code",
                            line_number=element.get('line', 0),
                            match_type="conceptual",
                            confidence=confidence,
                            similarity_score=confidence,
                            evidence=[f"Conceptual match: {explanation}"]
                        )
                        matches.append(match)
                
            except (json.JSONDecodeError, KeyError) as e:
                preview = raw[:200] if 'raw' in locals() else ''
                self.logger.warning(f"Could not parse LLM response for conceptual matching: {e} | raw preview: {preview}")
            
        except Exception as e:
            self.logger.error(f"Error in conceptual matching: {e}")
        
        return matches
    
    # Helper methods for hierarchical search
    
    async def _search_at_module_level(self, concepts: List[str], context: ContextWindow) -> List[MatchResult]:
        """Search for matches at module level."""
        matches = []
        
        for concept in concepts:
            for module in context.module_dependencies:
                if concept.lower() in module.lower():
                    match = MatchResult(
                        element_name=module,
                        element_type="module",
                        file_path="imports",
                        line_number=0,
                        match_type="fuzzy",
                        confidence=0.7,
                        similarity_score=0.7,
                        evidence=[f"Module name contains concept: '{concept}' in '{module}'"]
                    )
                    matches.append(match)
        
        return matches
    
    async def _search_at_class_level(self, concepts: List[str], context: ContextWindow) -> List[MatchResult]:
        """Search for matches at class level."""
        matches = []
        
        for concept in concepts:
            for class_name, bases in context.class_hierarchy.items():
                # Check class name
                if concept.lower() in class_name.lower():
                    match = MatchResult(
                        element_name=class_name,
                        element_type="class",
                        file_path="target_code",
                        line_number=0,
                        match_type="fuzzy",
                        confidence=0.8,
                        similarity_score=0.8,
                        evidence=[f"Class name contains concept: '{concept}' in '{class_name}'"]
                    )
                    matches.append(match)
                
                # Check base classes
                for base in bases:
                    if concept.lower() in base.lower():
                        match = MatchResult(
                            element_name=class_name,
                            element_type="class",
                            file_path="target_code",
                            line_number=0,
                            match_type="fuzzy",
                            confidence=0.6,
                            similarity_score=0.6,
                            evidence=[f"Base class contains concept: '{concept}' in '{base}' (class: {class_name})"]
                        )
                        matches.append(match)
        
        return matches
    
    async def _search_at_function_level(self, concepts: List[str], context: ContextWindow) -> List[MatchResult]:
        """Search for matches at function level."""
        matches = []
        
        for concept in concepts:
            # Search in related functions
            for func_sig in context.related_functions:
                func_name = func_sig.split('(')[0] if '(' in func_sig else func_sig
                
                if concept.lower() in func_name.lower():
                    match = MatchResult(
                        element_name=func_name,
                        element_type="function",
                        file_path="target_code",
                        line_number=0,
                        match_type="fuzzy",
                        confidence=0.75,
                        similarity_score=0.75,
                        evidence=[f"Function name contains concept: '{concept}' in '{func_name}'"]
                    )
                    matches.append(match)
            
            # Also search in target code functions (if not already found in semantic matching)
            try:
                if context.target_code:
                    tree = ast.parse(context.target_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            if concept.lower() in func_name.lower():
                                # Check if this wasn't already found in semantic matching
                                existing_names = {m.element_name for m in matches}
                                if func_name not in existing_names:
                                    match = MatchResult(
                                        element_name=func_name,
                                        element_type="function",
                                        file_path="target_code",
                                        line_number=getattr(node, 'lineno', 0),
                                        match_type="fuzzy",
                                        confidence=0.8,
                                        similarity_score=0.8,
                                        evidence=[f"Function name contains concept: '{concept}' in '{func_name}'"]
                                    )
                                    matches.append(match)
            except SyntaxError:
                pass  # Skip if code has syntax errors
        
        return matches
    
    # Helper methods for alternative candidate generation
    
    async def _find_similar_code_patterns(self, issue_description: str,
                                         concepts: List[str],
                                         context: ContextWindow) -> List[SearchCandidate]:
        """Find similar code patterns through analysis."""
        candidates = []
        
        try:
            # Use LLM to identify potential code patterns
            prompt = f"""
Analyze the following issue and suggest potential code locations that might contain the bug:

Issue: {issue_description[:800]}
Key concepts: {', '.join(concepts)}

Based on the issue description, suggest 3-5 potential locations or code patterns where the bug might be located.
For each suggestion, provide:
1. Location description
2. Reasoning for why this location might contain the bug
3. Confidence score (0.0-1.0)

Format as JSON:
{{
    "suggestions": [
        {{
            "location": "user authentication function",
            "reasoning": "The issue mentions login problems which typically occur in auth functions",
            "confidence": 0.8
        }}
    ]
}}
"""
            
            response = await self.llm_interface.generate(prompt, max_completion_tokens=1000)
            
            try:
                import json
                content = (response.content or "").strip()
                if not content:
                    self.logger.info("Pattern matching LLM response was empty")
                    return candidates
                
                # Remove markdown code fences if present
                if content.startswith("```"):
                    content = content.strip("`").strip()
                    if content.lower().startswith("json"):
                        content = content[len("json"):].lstrip()
                
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError as e:
                    # Try to salvage JSON wrapped in text
                    import re
                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        # Fix trailing commas
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        try:
                            result_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse pattern matching JSON: {e}")
                            return candidates
                    else:
                        self.logger.warning(f"No JSON found in pattern matching response")
                        return candidates
                
                for suggestion in result_data.get('suggestions', []):
                    candidate = SearchCandidate(
                        location=suggestion.get('location', ''),
                        description=f"Pattern-based suggestion: {suggestion.get('location', '')}",
                        confidence=suggestion.get('confidence', 0.5),
                        reasoning=suggestion.get('reasoning', ''),
                        code_snippet="",
                        similarity_metrics={'pattern_match': suggestion.get('confidence', 0.5)}
                    )
                    candidates.append(candidate)
                    
            except (json.JSONDecodeError, KeyError) as e:
                snippet = (response.content or "")[:200].replace("\n", " ")
                self.logger.warning(f"Could not parse LLM response for pattern matching: {e}; snippet: {snippet}")
            
        except Exception as e:
            self.logger.error(f"Error finding similar code patterns: {e}")
        
        return candidates
    
    async def _analyze_call_relationships(self, concepts: List[str],
                                        context: ContextWindow,
                                        primary_matches: List[MatchResult]) -> List[SearchCandidate]:
        """Analyze call relationships to find related code."""
        candidates = []
        
        try:
            if not context.dependency_context or not primary_matches:
                return candidates
            
            # Find functions that call or are called by primary matches
            call_graph = context.dependency_context.call_graph
            
            for match in primary_matches:
                if match.element_type == "function":
                    # Find callers of this function
                    callers = [caller for caller, callees in call_graph.items() 
                             if match.element_name in callees]
                    
                    for caller in callers:
                        candidate = SearchCandidate(
                            location=f"function: {caller}",
                            description=f"Calls {match.element_name} which was identified as relevant",
                            confidence=match.confidence * 0.7,
                            reasoning=f"This function calls {match.element_name}, which may indicate related functionality",
                            code_snippet="",
                            similarity_metrics={'call_relationship': match.confidence * 0.7}
                        )
                        candidates.append(candidate)
                    
                    # Find functions called by this function
                    callees = call_graph.get(match.element_name, [])
                    for callee in callees:
                        candidate = SearchCandidate(
                            location=f"function: {callee}",
                            description=f"Called by {match.element_name} which was identified as relevant",
                            confidence=match.confidence * 0.6,
                            reasoning=f"This function is called by {match.element_name}, which may contain related logic",
                            code_snippet="",
                            similarity_metrics={'call_relationship': match.confidence * 0.6}
                        )
                        candidates.append(candidate)
            
        except Exception as e:
            self.logger.error(f"Error analyzing call relationships: {e}")
        
        return candidates
    
    async def _match_error_patterns(self, issue_description: str, 
                                  context: ContextWindow) -> List[SearchCandidate]:
        """Match common error patterns in the issue description."""
        candidates = []
        
        try:
            # Common error patterns and their typical locations
            error_patterns = {
                r'null.*pointer|null.*reference|NoneType': {
                    'location': 'null/None checks',
                    'confidence': 0.8,
                    'reasoning': 'Issue mentions null/None errors which typically occur in variable access'
                },
                r'index.*out.*of.*bounds|list.*index': {
                    'location': 'array/list access',
                    'confidence': 0.9,
                    'reasoning': 'Issue mentions index errors which occur in array/list operations'
                },
                r'key.*error|missing.*key': {
                    'location': 'dictionary access',
                    'confidence': 0.85,
                    'reasoning': 'Issue mentions key errors which occur in dictionary operations'
                },
                r'connection.*error|timeout': {
                    'location': 'network/database operations',
                    'confidence': 0.8,
                    'reasoning': 'Issue mentions connection problems which occur in network code'
                },
                r'permission.*denied|access.*denied': {
                    'location': 'file/permission handling',
                    'confidence': 0.8,
                    'reasoning': 'Issue mentions permission errors which occur in file/security operations'
                }
            }
            
            issue_lower = issue_description.lower()
            
            for pattern, info in error_patterns.items():
                if re.search(pattern, issue_lower):
                    candidate = SearchCandidate(
                        location=info['location'],
                        description=f"Error pattern match: {info['location']}",
                        confidence=info['confidence'],
                        reasoning=info['reasoning'],
                        code_snippet="",
                        similarity_metrics={'error_pattern': info['confidence']}
                    )
                    candidates.append(candidate)
            
        except Exception as e:
            self.logger.error(f"Error matching error patterns: {e}")
        
        return candidates
    
    # Utility methods
    
    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate matches based on element name and file path."""
        seen = set()
        deduplicated = []
        
        for match in matches:
            key = (match.element_name, match.file_path, match.element_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(match)
        
        return deduplicated
    
    def _deduplicate_candidates(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        """Remove duplicate candidates based on location."""
        seen = set()
        deduplicated = []
        
        for candidate in candidates:
            if candidate.location not in seen:
                seen.add(candidate.location)
                deduplicated.append(candidate)
        
        return deduplicated
    
    def _calculate_matching_confidence(self, matches: List[MatchResult]) -> float:
        """Calculate overall confidence for a set of matches."""
        if not matches:
            return 0.0
        
        # Use weighted average based on match confidence
        total_weight = sum(match.confidence for match in matches)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(match.confidence ** 2 for match in matches)
        return weighted_sum / total_weight
    
    def _calculate_alternatives_confidence(self, alternatives: List[SearchCandidate]) -> float:
        """Calculate overall confidence for alternative candidates."""
        if not alternatives:
            return 0.0
        
        # Use average confidence with diminishing returns
        avg_confidence = sum(alt.confidence for alt in alternatives) / len(alternatives)
        return avg_confidence * 0.8  # Reduce confidence for alternatives
