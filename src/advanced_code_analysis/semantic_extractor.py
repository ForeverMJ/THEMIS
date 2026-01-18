"""
Semantic Information Extraction Engine for the Advanced Code Analysis system.

This module implements the SemanticExtractor class that extracts key information
from problem descriptions, including technical concepts, function names, variable names,
and provides confidence assessment and structured summaries.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import (
    BugType, PromptTemplate, ReasoningChain, ReasoningStep, 
    EvidenceChain, AnalysisResult
)
from .llm_interface import LLMInterface, LLMResponse
from .config import AdvancedAnalysisConfig


logger = logging.getLogger(__name__)


class InformationType(Enum):
    """Types of information that can be extracted."""
    TECHNICAL_CONCEPT = "technical_concept"
    FUNCTION_NAME = "function_name"
    VARIABLE_NAME = "variable_name"
    CLASS_NAME = "class_name"
    MODULE_NAME = "module_name"
    ERROR_PATTERN = "error_pattern"
    API_CALL = "api_call"
    DATA_TYPE = "data_type"
    ALGORITHM = "algorithm"
    FRAMEWORK = "framework"


@dataclass
class ExtractedInformation:
    """Represents a piece of extracted information with metadata."""
    content: str
    info_type: InformationType
    confidence: float
    context: str = ""
    source_location: str = ""
    related_terms: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class StructuredSummary:
    """Structured summary of extracted information."""
    problem_type: str
    key_components: List[str]
    technical_concepts: List[str]
    code_elements: List[str]
    error_indicators: List[str]
    confidence_score: float
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class ExtractionResult:
    """Complete result of semantic information extraction."""
    extracted_items: List[ExtractedInformation]
    structured_summary: StructuredSummary
    reasoning_chain: ReasoningChain
    overall_confidence: float
    processing_time: float
    
    def get_items_by_type(self, info_type: InformationType) -> List[ExtractedInformation]:
        """Get all extracted items of a specific type."""
        return [item for item in self.extracted_items if item.info_type == info_type]
    
    def get_high_confidence_items(self, threshold: float = 0.7) -> List[ExtractedInformation]:
        """Get items with confidence above threshold."""
        return [item for item in self.extracted_items if item.confidence >= threshold]


class SemanticExtractor:
    """
    Semantic Information Extraction Engine.
    
    Extracts key information from problem descriptions using LLM-powered analysis
    combined with pattern matching and heuristics.
    """
    
    def __init__(self, config: AdvancedAnalysisConfig, llm_interface: LLMInterface):
        """Initialize the semantic extractor."""
        self.config = config
        self.llm = llm_interface
        self.logger = logging.getLogger(__name__)
        
        # Initialize extraction patterns
        self._init_extraction_patterns()
        
        # Initialize prompt templates
        self._init_prompt_templates()
    
    def _init_extraction_patterns(self) -> None:
        """Initialize regex patterns for different types of information."""
        self.patterns = {
            InformationType.FUNCTION_NAME: [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # function_name(
                r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',   # def function_name
                r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # function function_name
                r'method\s+([a-zA-Z_][a-zA-Z0-9_]*)',    # method method_name
            ],
            InformationType.VARIABLE_NAME: [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=',   # variable =
                r'variable\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # variable var_name
                r'parameter\s+([a-zA-Z_][a-zA-Z0-9_]*)', # parameter param_name
            ],
            InformationType.CLASS_NAME: [
                r'class\s+([A-Z][a-zA-Z0-9_]*)',     # class ClassName
                r'\b([A-Z][a-zA-Z0-9_]*)\s*\(',      # ClassName(
                r'object\s+([A-Z][a-zA-Z0-9_]*)',    # object ClassName
            ],
            InformationType.MODULE_NAME: [
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',   # import module
                r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)',     # from module
                r'module\s+([a-zA-Z_][a-zA-Z0-9_.]*)',   # module module_name
            ],
            InformationType.ERROR_PATTERN: [
                r'(.*Error):\s*(.+)',                # SomeError: message
                r'(Exception):\s*(.+)',              # Exception: message
                r'(fails?|error|bug|issue|problem)', # error indicators
            ],
            InformationType.API_CALL: [
                r'([a-zA-Z_][a-zA-Z0-9_]*\.)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # obj.method(
                r'API\s+([a-zA-Z_][a-zA-Z0-9_.]*)',  # API api_name
            ],
        }
    
    def _init_prompt_templates(self) -> None:
        """Initialize LLM prompt templates for semantic extraction."""
        self.templates = {
            'extract_technical_concepts': PromptTemplate(
                template_id='extract_technical_concepts',
                content="""
Analyze the following problem description and extract key technical concepts, 
function names, variable names, class names, and error patterns.

Problem Description:
{problem_text}

Please provide a structured analysis in JSON format with the following structure:
{{
    "technical_concepts": [
        {{"term": "concept_name", "confidence": 0.9, "context": "where it appears"}}
    ],
    "function_names": [
        {{"name": "function_name", "confidence": 0.8, "context": "usage context"}}
    ],
    "variable_names": [
        {{"name": "variable_name", "confidence": 0.7, "context": "usage context"}}
    ],
    "class_names": [
        {{"name": "class_name", "confidence": 0.9, "context": "usage context"}}
    ],
    "error_patterns": [
        {{"pattern": "error_description", "confidence": 0.8, "type": "error_type"}}
    ],
    "api_calls": [
        {{"call": "api.method", "confidence": 0.8, "context": "usage context"}}
    ],
    "problem_summary": "Brief summary of the core problem",
    "overall_confidence": 0.85
}}

Focus on extracting concrete, actionable information that would help locate 
the relevant code sections. Assign confidence scores based on how clearly 
each element is mentioned in the problem description.
""",
                placeholders=['problem_text']
            ),
            
            'refine_extraction': PromptTemplate(
                template_id='refine_extraction',
                content="""
Review and refine the following extracted information from a problem description.
Identify any missing elements, improve confidence scores, and add related terms.

Original Problem:
{problem_text}

Current Extraction:
{current_extraction}

Please provide an improved extraction with:
1. Additional relevant terms that might have been missed
2. Refined confidence scores based on context clarity
3. Related terms for each extracted element
4. Improved problem categorization

Return the refined extraction in the same JSON format, focusing on completeness and accuracy.
""",
                placeholders=['problem_text', 'current_extraction']
            ),
            
            'generate_summary': PromptTemplate(
                template_id='generate_summary',
                content="""
Based on the extracted information, generate a structured summary of the problem.

Problem Description:
{problem_text}

Extracted Information:
{extracted_info}

Generate a structured summary with:
1. Problem type classification
2. Key components involved
3. Main technical concepts
4. Code elements to investigate
5. Error indicators and patterns
6. Overall confidence in the extraction

Format as JSON:
{{
    "problem_type": "type_of_problem",
    "key_components": ["component1", "component2"],
    "technical_concepts": ["concept1", "concept2"],
    "code_elements": ["element1", "element2"],
    "error_indicators": ["indicator1", "indicator2"],
    "confidence_score": 0.85,
    "reasoning": "Brief explanation of the analysis"
}}
""",
                placeholders=['problem_text', 'extracted_info']
            )
        }
    
    async def extract_information(self, problem_text: str, 
                                bug_type: Optional[BugType] = None) -> ExtractionResult:
        """
        Extract semantic information from problem description.
        
        Args:
            problem_text: The problem description to analyze
            bug_type: Optional bug type to guide extraction
            
        Returns:
            ExtractionResult containing all extracted information
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting semantic information extraction")
        
        # Initialize reasoning chain
        reasoning_chain = ReasoningChain()
        
        try:
            # Step 1: Pattern-based extraction
            pattern_items = await self._extract_with_patterns(problem_text, reasoning_chain)
            
            # Step 2: LLM-based extraction
            llm_items = await self._extract_with_llm(problem_text, reasoning_chain)
            
            # Step 3: Combine and refine results
            combined_items = await self._combine_and_refine(
                pattern_items, llm_items, problem_text, reasoning_chain
            )
            
            # Step 4: Generate structured summary
            summary = await self._generate_structured_summary(
                problem_text, combined_items, reasoning_chain
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(combined_items, summary)
            
            processing_time = time.time() - start_time
            
            result = ExtractionResult(
                extracted_items=combined_items,
                structured_summary=summary,
                reasoning_chain=reasoning_chain,
                overall_confidence=overall_confidence,
                processing_time=processing_time
            )
            
            self.logger.info(f"Extraction completed in {processing_time:.2f}s with "
                           f"confidence {overall_confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during semantic extraction: {e}")
            # Return minimal result on error
            return ExtractionResult(
                extracted_items=[],
                structured_summary=StructuredSummary(
                    problem_type="unknown",
                    key_components=[],
                    technical_concepts=[],
                    code_elements=[],
                    error_indicators=[],
                    confidence_score=0.0
                ),
                reasoning_chain=reasoning_chain,
                overall_confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    async def _extract_with_patterns(self, problem_text: str, 
                                   reasoning_chain: ReasoningChain) -> List[ExtractedInformation]:
        """Extract information using regex patterns."""
        step = ReasoningStep(
            step_number=len(reasoning_chain.steps) + 1,
            description="Pattern-based information extraction",
            input_data={"problem_text_length": len(problem_text)}
        )
        
        extracted_items = []
        
        for info_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, problem_text, re.IGNORECASE)
                for match in matches:
                    # Extract the relevant group (usually the first capturing group)
                    if match.groups():
                        content = match.group(1).strip()
                    else:
                        content = match.group(0).strip()
                    
                    if content and len(content) > 1:  # Filter out single characters
                        # Calculate confidence based on pattern specificity and context
                        confidence = self._calculate_pattern_confidence(
                            content, info_type, match, problem_text
                        )
                        
                        extracted_items.append(ExtractedInformation(
                            content=content,
                            info_type=info_type,
                            confidence=confidence,
                            context=self._extract_context(match, problem_text),
                            source_location=f"pattern:{pattern}"
                        ))
        
        # Remove duplicates and low-confidence items
        extracted_items = self._deduplicate_items(extracted_items)
        
        step.output_data = {
            "items_found": len(extracted_items),
            "types_found": list(set(item.info_type.value for item in extracted_items))
        }
        step.confidence = 0.7  # Pattern matching has moderate confidence
        
        reasoning_chain.add_step(step)
        
        return extracted_items
    
    async def _extract_with_llm(self, problem_text: str, 
                              reasoning_chain: ReasoningChain) -> List[ExtractedInformation]:
        """Extract information using LLM analysis."""
        step = ReasoningStep(
            step_number=len(reasoning_chain.steps) + 1,
            description="LLM-based semantic extraction",
            input_data={"problem_text": problem_text[:200] + "..."}
        )
        
        try:
            # Use LLM to extract technical concepts
            response = await self.llm.generate(
                self.templates['extract_technical_concepts'],
                template_vars={'problem_text': problem_text}
            )
            
            # Parse LLM response
            extracted_data = self._parse_llm_extraction_response(response.content)
            
            # Convert to ExtractedInformation objects
            extracted_items = []
            
            # Process technical concepts
            for concept in extracted_data.get('technical_concepts', []):
                extracted_items.append(ExtractedInformation(
                    content=concept['term'],
                    info_type=InformationType.TECHNICAL_CONCEPT,
                    confidence=concept.get('confidence', 0.5),
                    context=concept.get('context', ''),
                    source_location="llm:technical_concepts"
                ))
            
            # Process function names
            for func in extracted_data.get('function_names', []):
                extracted_items.append(ExtractedInformation(
                    content=func['name'],
                    info_type=InformationType.FUNCTION_NAME,
                    confidence=func.get('confidence', 0.5),
                    context=func.get('context', ''),
                    source_location="llm:function_names"
                ))
            
            # Process variable names
            for var in extracted_data.get('variable_names', []):
                extracted_items.append(ExtractedInformation(
                    content=var['name'],
                    info_type=InformationType.VARIABLE_NAME,
                    confidence=var.get('confidence', 0.5),
                    context=var.get('context', ''),
                    source_location="llm:variable_names"
                ))
            
            # Process class names
            for cls in extracted_data.get('class_names', []):
                extracted_items.append(ExtractedInformation(
                    content=cls['name'],
                    info_type=InformationType.CLASS_NAME,
                    confidence=cls.get('confidence', 0.5),
                    context=cls.get('context', ''),
                    source_location="llm:class_names"
                ))
            
            # Process error patterns
            for error in extracted_data.get('error_patterns', []):
                extracted_items.append(ExtractedInformation(
                    content=error['pattern'],
                    info_type=InformationType.ERROR_PATTERN,
                    confidence=error.get('confidence', 0.5),
                    context=error.get('type', ''),
                    source_location="llm:error_patterns"
                ))
            
            # Process API calls
            for api in extracted_data.get('api_calls', []):
                extracted_items.append(ExtractedInformation(
                    content=api['call'],
                    info_type=InformationType.API_CALL,
                    confidence=api.get('confidence', 0.5),
                    context=api.get('context', ''),
                    source_location="llm:api_calls"
                ))
            
            step.output_data = {
                "items_extracted": len(extracted_items),
                "llm_confidence": extracted_data.get('overall_confidence', 0.5)
            }
            step.confidence = extracted_data.get('overall_confidence', 0.5)
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            extracted_items = []
            step.output_data = {"error": str(e)}
            step.confidence = 0.0
        
        reasoning_chain.add_step(step)
        return extracted_items
    
    async def _combine_and_refine(self, pattern_items: List[ExtractedInformation],
                                llm_items: List[ExtractedInformation],
                                problem_text: str,
                                reasoning_chain: ReasoningChain) -> List[ExtractedInformation]:
        """Combine pattern and LLM results, then refine them."""
        step = ReasoningStep(
            step_number=len(reasoning_chain.steps) + 1,
            description="Combining and refining extraction results",
            input_data={
                "pattern_items": len(pattern_items),
                "llm_items": len(llm_items)
            }
        )
        
        # Combine items
        all_items = pattern_items + llm_items
        
        # Deduplicate and merge similar items
        combined_items = self._merge_similar_items(all_items)
        
        # Refine using LLM if we have items to refine
        if combined_items and self.config.analysis.enable_multi_round_reasoning:
            try:
                current_extraction = self._items_to_dict(combined_items)
                
                response = await self.llm.generate(
                    self.templates['refine_extraction'],
                    template_vars={
                        'problem_text': problem_text,
                        'current_extraction': json.dumps(current_extraction, indent=2)
                    }
                )
                
                refined_data = self._parse_llm_extraction_response(response.content)
                refined_items = self._dict_to_items(refined_data)
                
                # Merge refined results with original
                combined_items = self._merge_similar_items(combined_items + refined_items)
                
                step.confidence = refined_data.get('overall_confidence', 0.7)
                
            except Exception as e:
                self.logger.warning(f"Refinement failed, using combined results: {e}")
                step.confidence = 0.6
        else:
            step.confidence = 0.6
        
        # Filter by confidence threshold
        min_confidence = self.config.analysis.confidence_threshold * 0.5  # Lower threshold for extraction
        combined_items = [item for item in combined_items if item.confidence >= min_confidence]
        
        step.output_data = {
            "final_items": len(combined_items),
            "confidence_threshold": min_confidence
        }
        
        reasoning_chain.add_step(step)
        return combined_items
    
    async def _generate_structured_summary(self, problem_text: str,
                                         extracted_items: List[ExtractedInformation],
                                         reasoning_chain: ReasoningChain) -> StructuredSummary:
        """Generate a structured summary of the extraction results."""
        step = ReasoningStep(
            step_number=len(reasoning_chain.steps) + 1,
            description="Generating structured summary",
            input_data={"extracted_items": len(extracted_items)}
        )
        
        try:
            # Prepare extracted info for LLM
            extracted_info = self._items_to_dict(extracted_items)
            
            response = await self.llm.generate(
                self.templates['generate_summary'],
                template_vars={
                    'problem_text': problem_text,
                    'extracted_info': json.dumps(extracted_info, indent=2)
                }
            )
            
            summary_data = self._parse_summary_response(response.content)
            
            summary = StructuredSummary(
                problem_type=summary_data.get('problem_type', 'unknown'),
                key_components=summary_data.get('key_components', []),
                technical_concepts=summary_data.get('technical_concepts', []),
                code_elements=summary_data.get('code_elements', []),
                error_indicators=summary_data.get('error_indicators', []),
                confidence_score=summary_data.get('confidence_score', 0.5),
                extraction_metadata={
                    'reasoning': summary_data.get('reasoning', ''),
                    'total_items': len(extracted_items)
                }
            )
            
            step.confidence = summary.confidence_score
            step.output_data = {"summary_generated": True}
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            
            # Generate basic summary from extracted items
            summary = self._generate_basic_summary(extracted_items)
            step.confidence = 0.3
            step.output_data = {"error": str(e), "fallback_summary": True}
        
        reasoning_chain.add_step(step)
        return summary
    
    def _calculate_pattern_confidence(self, content: str, info_type: InformationType,
                                    match: re.Match, problem_text: str) -> float:
        """Calculate confidence score for pattern-matched items."""
        base_confidence = 0.6
        
        # Adjust based on content characteristics
        if len(content) < 3:
            base_confidence *= 0.5  # Very short names are less reliable
        elif len(content) > 20:
            base_confidence *= 0.7  # Very long names might be false positives
        
        # Adjust based on context
        context_window = 50
        start = max(0, match.start() - context_window)
        end = min(len(problem_text), match.end() + context_window)
        context = problem_text[start:end].lower()
        
        # Look for confirming keywords
        confirming_keywords = {
            InformationType.FUNCTION_NAME: ['function', 'method', 'call', 'invoke'],
            InformationType.VARIABLE_NAME: ['variable', 'parameter', 'argument', 'value'],
            InformationType.CLASS_NAME: ['class', 'object', 'instance', 'type'],
            InformationType.ERROR_PATTERN: ['error', 'exception', 'fail', 'bug'],
        }
        
        if info_type in confirming_keywords:
            for keyword in confirming_keywords[info_type]:
                if keyword in context:
                    base_confidence *= 1.2
                    break
        
        # Ensure confidence is within valid range
        return min(1.0, max(0.1, base_confidence))
    
    def _extract_context(self, match: re.Match, text: str, window_size: int = 30) -> str:
        """Extract context around a match."""
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        return text[start:end].strip()
    
    def _deduplicate_items(self, items: List[ExtractedInformation]) -> List[ExtractedInformation]:
        """Remove duplicate extracted items."""
        seen = set()
        unique_items = []
        
        for item in items:
            key = (item.content.lower(), item.info_type)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        return unique_items
    
    def _merge_similar_items(self, items: List[ExtractedInformation]) -> List[ExtractedInformation]:
        """Merge similar items and combine their confidence scores."""
        merged = {}
        
        for item in items:
            key = (item.content.lower(), item.info_type)
            
            if key in merged:
                # Merge with existing item
                existing = merged[key]
                # Use weighted average of confidences
                total_weight = existing.confidence + item.confidence
                existing.confidence = (existing.confidence * existing.confidence + 
                                     item.confidence * item.confidence) / total_weight
                
                # Combine contexts
                if item.context and item.context not in existing.context:
                    existing.context += f"; {item.context}"
                
                # Combine related terms
                existing.related_terms.extend(item.related_terms)
                existing.related_terms = list(set(existing.related_terms))
                
            else:
                merged[key] = item
        
        return list(merged.values())
    
    def _parse_llm_extraction_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response for extraction data."""
        try:
            # 1. Try to extract JSON from markdown code blocks
            code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if code_block:
                try:
                    return json.loads(code_block.group(1))
                except json.JSONDecodeError:
                    pass  # Fall through to other methods

            # 2. Try to find the first outer-most JSON object
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Attempt to clean trailing commas which are common in LLM output
                    content_clean = re.sub(r',\s*\}', '}', content)
                    content_clean = re.sub(r',\s*\]', ']', content_clean)
                    try:
                        return json.loads(content_clean)
                    except json.JSONDecodeError:
                        pass # Fall through

            # 3. Fallback: try to parse the entire response as JSON
            return json.loads(response_content)

        except json.JSONDecodeError as e:
            # 4. Try to repair truncated JSON first as it's a common issue with large responses
            try:
                repaired = self._repair_truncated_json(response_content)
                if repaired:
                    self.logger.info(f"JSON parsing failed but was successfully repaired: {e}")
                    return repaired
            except Exception:
                pass

            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            
            # 5. Last resort: try lenient parsing for common issues on the raw content
            try:
                content = response_content
                content = re.sub(r',\s*\}', '}', content)
                content = re.sub(r',\s*\]', ']', content)
                # json.loads with strict=False allows control characters
                return json.loads(content, strict=False)
            except Exception:
                pass
                
            return {}

    def _repair_truncated_json(self, content: str) -> Dict[str, Any]:
        """Attempt to repair truncated JSON by removing incomplete parts and closing brackets."""
        content = content.strip()
        
        # Find the last closing brace or bracket
        last_brace = content.rfind('}')
        last_bracket = content.rfind(']')
        
        cut_point = max(last_brace, last_bracket)
        if cut_point == -1:
            return {}
            
        # Try with the prefix up to the last closing character
        content_prefix = content[:cut_point+1]
        
        # Try appending combinations of closing brackets
        # The structure usually is Object -> Key -> Array -> Object
        # So likely we need ']}' or ']} }' etc.
        suffixes = ['}', ']}', ']]}', '}}', '}]}', ']}}', '}]}}']
        
        for suffix in suffixes:
            try:
                return json.loads(content_prefix + suffix, strict=False)
            except json.JSONDecodeError:
                continue
                
        return {}
    
    def _parse_summary_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response for summary data."""
        return self._parse_llm_extraction_response(response_content)
    
    def _items_to_dict(self, items: List[ExtractedInformation]) -> Dict[str, List[Dict[str, Any]]]:
        """Convert extracted items to dictionary format."""
        result = {}
        
        for item in items:
            type_key = item.info_type.value + 's'  # pluralize
            if type_key not in result:
                result[type_key] = []
            
            result[type_key].append({
                'content': item.content,
                'confidence': item.confidence,
                'context': item.context,
                'related_terms': item.related_terms
            })
        
        return result
    
    def _dict_to_items(self, data: Dict[str, Any]) -> List[ExtractedInformation]:
        """Convert dictionary format back to ExtractedInformation objects."""
        items = []
        
        type_mapping = {
            'technical_concepts': InformationType.TECHNICAL_CONCEPT,
            'function_names': InformationType.FUNCTION_NAME,
            'variable_names': InformationType.VARIABLE_NAME,
            'class_names': InformationType.CLASS_NAME,
            'error_patterns': InformationType.ERROR_PATTERN,
            'api_calls': InformationType.API_CALL,
        }
        
        for key, info_type in type_mapping.items():
            for item_data in data.get(key, []):
                if isinstance(item_data, dict):
                    content = item_data.get('content') or item_data.get('term') or item_data.get('name') or item_data.get('pattern') or item_data.get('call')
                    if content:
                        items.append(ExtractedInformation(
                            content=content,
                            info_type=info_type,
                            confidence=item_data.get('confidence', 0.5),
                            context=item_data.get('context', ''),
                            source_location="llm:refined",
                            related_terms=item_data.get('related_terms', [])
                        ))
        
        return items
    
    def _generate_basic_summary(self, items: List[ExtractedInformation]) -> StructuredSummary:
        """Generate a basic summary when LLM summary generation fails."""
        # Group items by type
        by_type = {}
        for item in items:
            if item.info_type not in by_type:
                by_type[item.info_type] = []
            by_type[item.info_type].append(item)
        
        # Extract key information
        technical_concepts = [item.content for item in by_type.get(InformationType.TECHNICAL_CONCEPT, [])]
        code_elements = []
        code_elements.extend([item.content for item in by_type.get(InformationType.FUNCTION_NAME, [])])
        code_elements.extend([item.content for item in by_type.get(InformationType.CLASS_NAME, [])])
        code_elements.extend([item.content for item in by_type.get(InformationType.VARIABLE_NAME, [])])
        
        error_indicators = [item.content for item in by_type.get(InformationType.ERROR_PATTERN, [])]
        
        # Determine problem type based on extracted information
        problem_type = "general"
        if error_indicators:
            problem_type = "error_analysis"
        elif by_type.get(InformationType.API_CALL):
            problem_type = "api_issue"
        elif by_type.get(InformationType.FUNCTION_NAME):
            problem_type = "function_issue"
        
        # Calculate confidence based on number and quality of items
        confidence = min(0.8, len(items) * 0.1) if items else 0.1
        
        return StructuredSummary(
            problem_type=problem_type,
            key_components=code_elements[:5],  # Top 5
            technical_concepts=technical_concepts[:5],  # Top 5
            code_elements=code_elements,
            error_indicators=error_indicators,
            confidence_score=confidence,
            extraction_metadata={'fallback_generation': True}
        )
    
    def _calculate_overall_confidence(self, items: List[ExtractedInformation], 
                                    summary: StructuredSummary) -> float:
        """Calculate overall confidence for the extraction result."""
        if not items:
            return 0.0
        
        # Average confidence of all items
        item_confidence = sum(item.confidence for item in items) / len(items)
        
        # Summary confidence
        summary_confidence = summary.confidence_score
        
        # Combine with weights
        overall = (item_confidence * 0.6 + summary_confidence * 0.4)
        
        # Adjust based on number of items (more items = higher confidence)
        item_count_factor = min(1.0, len(items) / 10.0)
        overall *= (0.5 + 0.5 * item_count_factor)
        
        return min(1.0, overall)