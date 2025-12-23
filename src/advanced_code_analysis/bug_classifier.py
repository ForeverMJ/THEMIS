"""
Intelligent Bug Classification Engine for the Advanced Code Analysis system.

This module implements the BugClassifier class that uses LLM to intelligently
classify bug types and select appropriate analysis strategies based on the
problem description.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .models import (
    BugType, BugCategory, AnalysisStrategy, PromptTemplate, 
    ClassificationFeedback
)
from .llm_interface import LLMInterface, LLMResponse
from .config import AdvancedAnalysisConfig


logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of bug classification with confidence and reasoning."""
    bug_type: BugType
    reasoning: str
    alternative_types: List[BugType] = field(default_factory=list)
    classification_confidence: float = 0.0
    processing_time: float = 0.0


class PromptTemplateLibrary:
    """Library of prompt templates for different bug classification scenarios."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize the prompt template library."""
        templates = {}
        
        # Main classification prompt
        templates["classification"] = PromptTemplate(
            template_id="bug_classification",
            content="""You are an expert software engineer analyzing bug reports. Your task is to classify the type of bug based on the problem description.

Bug Categories:
1. LOGIC_ERROR: Incorrect program logic, wrong conditions, faulty algorithms
2. API_ISSUE: Problems with API usage, incorrect parameters, missing calls
3. PERFORMANCE: Slow execution, memory leaks, inefficient algorithms
4. BOUNDARY_CONDITION: Edge cases, null checks, array bounds, input validation
5. TYPE_ERROR: Type mismatches, casting issues, incorrect data types
6. CONCURRENCY: Race conditions, deadlocks, thread safety issues
7. RESOURCE_MANAGEMENT: File handles, memory allocation, connection management
8. CONFIGURATION: Settings, environment variables, deployment issues

Problem Description:
{issue_text}

Code Context (if available):
{code_context}

Please analyze this problem and provide:
1. Primary bug category (one of the 8 categories above)
2. Subcategory (specific type within the category)
3. Confidence score (0.0 to 1.0)
4. Key characteristics that led to this classification
5. Brief reasoning for your classification

Respond in JSON format:
{{
    "category": "CATEGORY_NAME",
    "subcategory": "specific_type",
    "confidence": 0.85,
    "characteristics": ["characteristic1", "characteristic2"],
    "reasoning": "Brief explanation of classification logic"
}}""",
            placeholders=["issue_text", "code_context"]
        )
        
        # Refinement prompt for ambiguous cases
        templates["refinement"] = PromptTemplate(
            template_id="classification_refinement",
            content="""The initial classification was ambiguous. Please refine your analysis.

Original Problem:
{issue_text}

Previous Classification:
{previous_classification}

Additional Context:
{additional_context}

Please provide a more specific classification focusing on:
1. The most likely root cause
2. The primary symptom vs underlying issue
3. Which category would be most helpful for selecting analysis strategy

Respond in the same JSON format as before.""",
            placeholders=["issue_text", "previous_classification", "additional_context"]
        )
        
        # Strategy selection prompt
        templates["strategy_selection"] = PromptTemplate(
            template_id="strategy_selection",
            content="""Based on the bug classification, select the most appropriate analysis strategy.

Bug Type: {bug_category} - {subcategory}
Confidence: {confidence}
Characteristics: {characteristics}

Available Strategies:
{available_strategies}

Consider:
1. The complexity of the problem
2. The amount of context needed
3. The type of reasoning required
4. The verification steps needed

Select the strategy that would be most effective for this type of bug and explain why.

Respond in JSON format:
{{
    "selected_strategy": "strategy_name",
    "reasoning": "Why this strategy is most appropriate",
    "confidence": 0.9
}}""",
            placeholders=["bug_category", "subcategory", "confidence", "characteristics", "available_strategies"]
        )
        
        return templates
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a new template to the library."""
        self.templates[template.template_id] = template


class AnalysisStrategyLibrary:
    """Library of analysis strategies for different bug types."""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, AnalysisStrategy]:
        """Initialize the analysis strategy library."""
        strategies = {}
        
        # Logic Error Strategy
        strategies["logic_error_deep_analysis"] = AnalysisStrategy(
            strategy_name="logic_error_deep_analysis",
            prompt_template=PromptTemplate(
                template_id="logic_error_analysis",
                content="""Analyze this logic error by examining the control flow and decision points.

Problem: {issue_text}
Code: {target_code}
Context: {context_info}

Focus on:
1. Conditional statements and their logic
2. Loop conditions and termination
3. Variable state changes
4. Expected vs actual behavior
5. Edge cases in the logic

Provide detailed analysis of where the logic breaks down.""",
                placeholders=["issue_text", "target_code", "context_info"]
            ),
            context_requirements=["control_flow", "variable_states", "function_logic"],
            verification_steps=["trace_execution", "check_conditions", "validate_outputs"],
            max_rounds=4,
            confidence_threshold=0.8
        )
        
        # API Issue Strategy
        strategies["api_issue_analysis"] = AnalysisStrategy(
            strategy_name="api_issue_analysis",
            prompt_template=PromptTemplate(
                template_id="api_issue_analysis",
                content="""Analyze this API-related issue by examining the API usage patterns.

Problem: {issue_text}
Code: {target_code}
API Documentation: {api_docs}

Focus on:
1. Parameter types and values
2. Return value handling
3. Error handling patterns
4. API version compatibility
5. Authentication and permissions

Identify incorrect API usage and suggest corrections.""",
                placeholders=["issue_text", "target_code", "api_docs"]
            ),
            context_requirements=["api_signatures", "parameter_types", "error_handling"],
            verification_steps=["check_parameters", "validate_returns", "test_error_cases"],
            max_rounds=3,
            confidence_threshold=0.85
        )
        
        # Performance Issue Strategy
        strategies["performance_analysis"] = AnalysisStrategy(
            strategy_name="performance_analysis",
            prompt_template=PromptTemplate(
                template_id="performance_analysis",
                content="""Analyze this performance issue by examining algorithmic complexity and resource usage.

Problem: {issue_text}
Code: {target_code}
Performance Metrics: {metrics}

Focus on:
1. Algorithmic complexity (time/space)
2. Inefficient loops or recursion
3. Memory allocation patterns
4. I/O operations and blocking calls
5. Data structure choices

Identify performance bottlenecks and optimization opportunities.""",
                placeholders=["issue_text", "target_code", "metrics"]
            ),
            context_requirements=["complexity_analysis", "resource_usage", "call_patterns"],
            verification_steps=["measure_complexity", "profile_execution", "benchmark_alternatives"],
            max_rounds=3,
            confidence_threshold=0.75
        )
        
        # Boundary Condition Strategy
        strategies["boundary_condition_analysis"] = AnalysisStrategy(
            strategy_name="boundary_condition_analysis",
            prompt_template=PromptTemplate(
                template_id="boundary_condition_analysis",
                content="""Analyze this boundary condition issue by examining edge cases and input validation.

Problem: {issue_text}
Code: {target_code}
Input Constraints: {constraints}

Focus on:
1. Null/undefined value handling
2. Array/list bounds checking
3. Numeric range validation
4. String length and format validation
5. Resource limits and quotas

Identify missing or incorrect boundary checks.""",
                placeholders=["issue_text", "target_code", "constraints"]
            ),
            context_requirements=["input_validation", "bounds_checking", "error_handling"],
            verification_steps=["test_edge_cases", "validate_inputs", "check_bounds"],
            max_rounds=2,
            confidence_threshold=0.9
        )
        
        # Type Error Strategy
        strategies["type_error_analysis"] = AnalysisStrategy(
            strategy_name="type_error_analysis",
            prompt_template=PromptTemplate(
                template_id="type_error_analysis",
                content="""Analyze this type-related issue by examining type usage and conversions.

Problem: {issue_text}
Code: {target_code}
Type Information: {type_info}

Focus on:
1. Variable type declarations and usage
2. Function parameter and return types
3. Type casting and conversions
4. Generic type parameters
5. Interface and inheritance issues

Identify type mismatches and suggest corrections.""",
                placeholders=["issue_text", "target_code", "type_info"]
            ),
            context_requirements=["type_signatures", "inheritance_hierarchy", "generic_constraints"],
            verification_steps=["check_type_compatibility", "validate_casts", "test_type_safety"],
            max_rounds=2,
            confidence_threshold=0.85
        )
        
        # General Multi-Round Strategy
        strategies["multi_round_general"] = AnalysisStrategy(
            strategy_name="multi_round_general",
            prompt_template=PromptTemplate(
                template_id="general_analysis",
                content="""Perform comprehensive analysis of this software issue.

Problem: {issue_text}
Code: {target_code}
Context: {context_info}

Use systematic approach:
1. Understand the problem symptoms
2. Identify potential root causes
3. Analyze code structure and logic
4. Consider environmental factors
5. Propose solutions with confidence levels

Provide thorough analysis with multiple hypotheses if needed.""",
                placeholders=["issue_text", "target_code", "context_info"]
            ),
            context_requirements=["full_context", "dependencies", "environment"],
            verification_steps=["validate_hypotheses", "test_solutions", "check_side_effects"],
            max_rounds=5,
            confidence_threshold=0.7
        )
        
        return strategies
    
    def get_strategy(self, strategy_name: str) -> Optional[AnalysisStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(strategy_name)
    
    def get_strategies_for_category(self, category: BugCategory) -> List[AnalysisStrategy]:
        """Get all strategies suitable for a bug category."""
        category_mapping = {
            BugCategory.LOGIC_ERROR: ["logic_error_deep_analysis", "multi_round_general"],
            BugCategory.API_ISSUE: ["api_issue_analysis", "multi_round_general"],
            BugCategory.PERFORMANCE: ["performance_analysis", "multi_round_general"],
            BugCategory.BOUNDARY_CONDITION: ["boundary_condition_analysis", "multi_round_general"],
            BugCategory.TYPE_ERROR: ["type_error_analysis", "multi_round_general"],
            BugCategory.CONCURRENCY: ["multi_round_general"],
            BugCategory.RESOURCE_MANAGEMENT: ["multi_round_general"],
            BugCategory.CONFIGURATION: ["multi_round_general"],
        }
        
        strategy_names = category_mapping.get(category, ["multi_round_general"])
        return [self.strategies[name] for name in strategy_names if name in self.strategies]
    
    def add_strategy(self, strategy: AnalysisStrategy) -> None:
        """Add a new strategy to the library."""
        self.strategies[strategy.strategy_name] = strategy


class BugClassifier:
    """Intelligent bug classifier that uses LLM to classify problems and select strategies."""
    
    def __init__(self, config: AdvancedAnalysisConfig, llm_interface: LLMInterface):
        self.config = config
        self.llm = llm_interface
        self.prompt_library = PromptTemplateLibrary()
        self.strategy_library = AnalysisStrategyLibrary()
        self.feedback_history: List[ClassificationFeedback] = []
        self.logger = logging.getLogger(__name__)
    
    def _completion_kwargs(self, max_completion_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Build kwargs for LLMInterface.generate, adapting to gpt-5 系列が要求する
        max_completion_tokens と temperature=1 の制約。
        """
        if self.config.llm.model_name.startswith("gpt-5"):
            return {"max_completion_tokens": max_completion_tokens, "temperature": 1}
        return {"max_completion_tokens": max_completion_tokens, "temperature": temperature}
    
    async def classify_bug_type(self, issue_text: str, 
                               code_context: Optional[str] = None) -> ClassificationResult:
        """Classify the bug type based on issue description and optional code context."""
        import time
        start_time = time.time()
        
        try:
            # Prepare context
            context = code_context or "No code context provided"
            
            # Get classification template
            template = self.prompt_library.get_template("classification")
            if not template:
                raise ValueError("Classification template not found")
            
            # Generate classification
            response = await self.llm.generate(
                template,
                template_vars={
                    "issue_text": issue_text,
                    "code_context": context
                },
                **self._completion_kwargs(1000, temperature=0.1)
            )
            
            # Parse response
            classification_data = self._parse_classification_response(response.content)
            
            # Create BugType
            bug_type = BugType(
                category=BugCategory(classification_data["category"].lower()),
                subcategory=classification_data.get("subcategory"),
                confidence=classification_data.get("confidence", 0.0),
                characteristics=classification_data.get("characteristics", [])
            )
            
            # Check if refinement is needed
            if bug_type.confidence < self.config.analysis.classification_confidence_threshold:
                self.logger.info(f"Low confidence ({bug_type.confidence}), attempting refinement")
                refined_result = await self._refine_classification(
                    issue_text, bug_type, context
                )
                if refined_result and refined_result.bug_type.confidence > bug_type.confidence:
                    bug_type = refined_result.bug_type
                    classification_data["reasoning"] = refined_result.reasoning
            
            processing_time = time.time() - start_time
            
            result = ClassificationResult(
                bug_type=bug_type,
                reasoning=classification_data.get("reasoning", ""),
                classification_confidence=bug_type.confidence,
                processing_time=processing_time
            )
            
            self.logger.info(f"Classified bug as {bug_type.category.value} "
                           f"(confidence: {bug_type.confidence:.2f}) in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in bug classification: {e}")
            # Return default classification
            return ClassificationResult(
                bug_type=BugType(
                    category=BugCategory.LOGIC_ERROR,
                    subcategory="unknown",
                    confidence=0.1,
                    characteristics=["classification_failed"]
                ),
                reasoning=f"Classification failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    async def select_analysis_strategy(self, bug_type: BugType) -> AnalysisStrategy:
        """Select the most appropriate analysis strategy for the given bug type."""
        try:
            # Get available strategies for this bug category
            available_strategies = self.strategy_library.get_strategies_for_category(bug_type.category)
            
            if not available_strategies:
                # Fallback to general strategy
                general_strategy = self.strategy_library.get_strategy("multi_round_general")
                if general_strategy:
                    return general_strategy
                else:
                    raise ValueError("No analysis strategies available")
            
            # If only one strategy available, return it
            if len(available_strategies) == 1:
                return available_strategies[0]
            
            # Use LLM to select the best strategy
            template = self.prompt_library.get_template("strategy_selection")
            if not template:
                # Return the first available strategy as fallback
                return available_strategies[0]
            
            # Prepare strategy descriptions
            strategy_descriptions = []
            for strategy in available_strategies:
                desc = f"- {strategy.strategy_name}: {len(strategy.verification_steps)} verification steps, "
                desc += f"max {strategy.max_rounds} rounds, confidence threshold {strategy.confidence_threshold}"
                strategy_descriptions.append(desc)
            
            response = await self.llm.generate(
                template,
                template_vars={
                    "bug_category": bug_type.category.value,
                    "subcategory": bug_type.subcategory or "general",
                    "confidence": str(bug_type.confidence),
                    "characteristics": ", ".join(bug_type.characteristics),
                    "available_strategies": "\n".join(strategy_descriptions)
                },
                **self._completion_kwargs(500, temperature=0.1)
            )
            
            # Parse strategy selection
            selection_data = self._parse_strategy_selection_response(response.content)
            selected_name = selection_data.get("selected_strategy")
            
            # Find and return the selected strategy
            for strategy in available_strategies:
                if strategy.strategy_name == selected_name:
                    self.logger.info(f"Selected strategy: {selected_name}")
                    return strategy
            
            # Fallback to first available strategy
            self.logger.warning(f"Could not find selected strategy '{selected_name}', using fallback")
            return available_strategies[0]
            
        except Exception as e:
            self.logger.error(f"Error in strategy selection: {e}")
            # Return general strategy as fallback
            general_strategy = self.strategy_library.get_strategy("multi_round_general")
            if general_strategy:
                return general_strategy
            else:
                # Create a minimal fallback strategy
                return AnalysisStrategy(
                    strategy_name="fallback",
                    prompt_template=PromptTemplate(
                        template_id="fallback",
                        content="Analyze this issue: {issue_text}",
                        placeholders=["issue_text"]
                    )
                )
    
    def get_prompt_template(self, bug_type: BugType) -> PromptTemplate:
        """Get the appropriate prompt template for the bug type."""
        # Map bug categories to template IDs
        template_mapping = {
            BugCategory.LOGIC_ERROR: "logic_error_analysis",
            BugCategory.API_ISSUE: "api_issue_analysis",
            BugCategory.PERFORMANCE: "performance_analysis",
            BugCategory.BOUNDARY_CONDITION: "boundary_condition_analysis",
            BugCategory.TYPE_ERROR: "type_error_analysis",
        }
        
        template_id = template_mapping.get(bug_type.category, "general_analysis")
        template = self.prompt_library.get_template(template_id)
        
        if not template:
            # Return a general template as fallback
            return PromptTemplate(
                template_id="general_fallback",
                content="""Analyze this software issue:

Problem: {issue_text}
Code: {target_code}

Provide detailed analysis and suggestions for resolution.""",
                placeholders=["issue_text", "target_code"]
            )
        
        return template
    
    def update_classification_model(self, feedback: ClassificationFeedback) -> None:
        """Update the classification model based on feedback."""
        self.feedback_history.append(feedback)
        self.logger.info(f"Added classification feedback: "
                        f"{feedback.original_classification.category.value} -> "
                        f"{feedback.correct_classification.category.value}")
        
        # TODO: Implement learning from feedback
        # This could involve:
        # 1. Updating prompt templates based on common mistakes
        # 2. Adjusting confidence thresholds
        # 3. Creating new classification rules
        # 4. Fine-tuning classification prompts
    
    async def _refine_classification(self, issue_text: str, 
                                   initial_classification: BugType,
                                   context: str) -> Optional[ClassificationResult]:
        """Refine classification for ambiguous cases."""
        try:
            template = self.prompt_library.get_template("refinement")
            if not template:
                return None
            
            # Prepare previous classification info
            prev_classification = {
                "category": initial_classification.category.value,
                "subcategory": initial_classification.subcategory,
                "confidence": initial_classification.confidence,
                "characteristics": initial_classification.characteristics
            }
            
            response = await self.llm.generate(
                template,
                template_vars={
                    "issue_text": issue_text,
                    "previous_classification": json.dumps(prev_classification, indent=2),
                    "additional_context": context
                },
                **self._completion_kwargs(800, temperature=0.05)  # Lower temperature for refinement
            )
            
            # Parse refined classification
            refined_data = self._parse_classification_response(response.content)
            
            refined_bug_type = BugType(
                category=BugCategory(refined_data["category"].lower()),
                subcategory=refined_data.get("subcategory"),
                confidence=refined_data.get("confidence", 0.0),
                characteristics=refined_data.get("characteristics", [])
            )
            
            return ClassificationResult(
                bug_type=refined_bug_type,
                reasoning=refined_data.get("reasoning", ""),
                classification_confidence=refined_bug_type.confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in classification refinement: {e}")
            return None
    
    def _parse_classification_response(self, response_content: str) -> Dict:
        """Parse LLM response for bug classification."""
        try:
            import re

            raw = (response_content or "").strip()
            # Remove code fences if present
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.lower().startswith("json"):
                    raw = raw[len("json"):].lstrip()
            if not raw:
                raise ValueError("Empty classification response")

            # Try to extract JSON block
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Validate required fields
            if "category" not in data:
                raise ValueError("Missing 'category' field in classification response")

            # Normalize category name
            category = data["category"].upper()
            if category not in [cat.name for cat in BugCategory]:
                self.logger.warning(f"Unknown category '{category}', defaulting to LOGIC_ERROR")
                data["category"] = "LOGIC_ERROR"
            else:
                data["category"] = category

            return data

        except (json.JSONDecodeError, ValueError) as e:
            preview = (response_content or "")[:200]
            self.logger.error(f"Error parsing classification response: {e} | raw preview: {preview}")
            # Return default classification
            return {
                "category": "LOGIC_ERROR",
                "subcategory": "unknown",
                "confidence": 0.1,
                "characteristics": ["parse_error"],
                "reasoning": f"Failed to parse response: {str(e)}"
            }
    
    def _parse_strategy_selection_response(self, response_content: str) -> Dict:
        """Parse LLM response for strategy selection."""
        try:
            import re
            
            # Look for JSON block
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try to extract strategy name from text
                strategy_match = re.search(r'strategy["\']?\s*:\s*["\']?(\w+)', response_content, re.IGNORECASE)
                if strategy_match:
                    return {"selected_strategy": strategy_match.group(1)}
                else:
                    return {"selected_strategy": "multi_round_general"}
                    
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing strategy selection response: {e}")
            return {"selected_strategy": "multi_round_general"}
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""
        if not self.feedback_history:
            return {"total_feedback": 0, "accuracy": 0.0}
        
        correct_classifications = sum(
            1 for feedback in self.feedback_history
            if feedback.original_classification.category == feedback.correct_classification.category
        )
        
        accuracy = correct_classifications / len(self.feedback_history)
        
        category_stats = {}
        for category in BugCategory:
            category_feedback = [
                f for f in self.feedback_history
                if f.original_classification.category == category
            ]
            if category_feedback:
                correct = sum(
                    1 for f in category_feedback
                    if f.original_classification.category == f.correct_classification.category
                )
                category_stats[category.value] = {
                    "total": len(category_feedback),
                    "correct": correct,
                    "accuracy": correct / len(category_feedback)
                }
        
        return {
            "total_feedback": len(self.feedback_history),
            "accuracy": accuracy,  # Keep both for compatibility
            "overall_accuracy": accuracy,
            "category_stats": category_stats
        }
