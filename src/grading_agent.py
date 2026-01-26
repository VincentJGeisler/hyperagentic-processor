"""
Grading Agent - Functional Implementation

This agent actually evaluates tool and task performance using multiple metrics.
It combines psychological motivation with real performance evaluation capabilities.
"""

import ast
import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import json

from motivated_agent import MotivatedAgent
from agent_drive_system import DriveType

logger = logging.getLogger("GradingAgent")

class PerformanceCategory(Enum):
    """Categories of performance evaluation"""
    CORRECTNESS = "correctness"
    EFFICIENCY = "efficiency"
    CODE_QUALITY = "code_quality"
    REUSABILITY = "reusability"

@dataclass
class PerformanceMetric:
    """A specific performance measurement"""
    category: PerformanceCategory
    score: float  # 0.0 to 1.0
    weight: float  # Importance weight
    description: str
    evidence: List[str]
    improvement_suggestions: List[str]

@dataclass
class PerformanceReport:
    """Comprehensive performance evaluation report"""
    tool_id: str
    tool_name: str
    evaluation_timestamp: datetime
    metrics: List[PerformanceMetric]
    composite_score: float  # Weighted average
    grade_letter: str  # A, B, C, D, F
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    comparison_to_peers: Optional[Dict[str, float]]
    trend_analysis: Optional[str]

class GradingAgent(MotivatedAgent):
    """
    A motivated agent that actually evaluates tool and task performance.
    
    This agent can:
    - Analyze code quality using multiple metrics
    - Measure performance and efficiency
    - Evaluate correctness through testing
    - Assess reusability and maintainability
    - Compare performance against historical data
    - Provide detailed improvement recommendations
    - Track performance trends over time
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        base_system_message = """You are a GradingAgent responsible for evaluating tool and task performance.
Your primary role is to provide objective, multi-dimensional assessment of work quality.
You measure correctness, efficiency, code quality, and reusability.

You strive for fairness and accuracy in your evaluations.
You help other agents improve by providing constructive feedback.
You serve the divine will by ensuring quality standards are maintained.

When evaluating performance, you should:
1. Analyze correctness through testing and validation
2. Measure efficiency and resource usage
3. Assess code quality and maintainability
4. Evaluate reusability and modularity
5. Compare against historical performance
6. Provide specific improvement recommendations
7. Track trends and patterns over time"""
        
        super().__init__(
            name="GradingAgent",
            agent_role="grading_agent",
            base_system_message=base_system_message,
            llm_config=llm_config
        )
        
        # Performance evaluation capabilities
        self.evaluation_history: List[Dict] = []
        self.performance_baselines = self._initialize_baselines()
        self.quality_standards = self._initialize_quality_standards()
        
        # Grading weights (must sum to 1.0)
        self.category_weights = {
            PerformanceCategory.CORRECTNESS: 0.40,
            PerformanceCategory.EFFICIENCY: 0.25,
            PerformanceCategory.CODE_QUALITY: 0.20,
            PerformanceCategory.REUSABILITY: 0.15
        }
        
        # Grade boundaries
        self.grade_boundaries = {
            'A': 0.90,
            'B': 0.80,
            'C': 0.70,
            'D': 0.60,
            'F': 0.00
        }
        
        # Boost evaluation-related drives
        self.drive_system.drives[DriveType.MASTERY].intensity = 0.8
        self.drive_system.drives[DriveType.RECOGNITION].intensity = 0.6
        self.drive_system.personality.perfectionism = 0.8
        
        logger.info("GradingAgent initialized with functional performance evaluation capabilities")
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize performance baselines for comparison"""
        return {
            "average_execution_time": 1.0,  # seconds
            "average_memory_usage": 10.0,   # MB
            "average_code_length": 50,      # lines
            "average_complexity": 5.0,      # cyclomatic complexity
            "average_test_coverage": 0.8,   # 80%
            "average_documentation": 0.7    # 70% documented
        }
    
    def _initialize_quality_standards(self) -> Dict[str, Dict]:
        """Initialize code quality standards"""
        return {
            "naming_conventions": {
                "function_pattern": r"^[a-z_][a-z0-9_]*$",
                "class_pattern": r"^[A-Z][a-zA-Z0-9]*$",
                "constant_pattern": r"^[A-Z_][A-Z0-9_]*$"
            },
            "complexity_limits": {
                "max_function_length": 50,
                "max_cyclomatic_complexity": 10,
                "max_nesting_depth": 4
            },
            "documentation_requirements": {
                "function_docstring": True,
                "class_docstring": True,
                "parameter_documentation": True,
                "return_documentation": True
            }
        }
    
    def evaluate_tool_performance(self, tool_code: str, tool_name: str, 
                                 execution_results: Optional[Dict] = None,
                                 test_results: Optional[Dict] = None) -> PerformanceReport:
        """
        Perform comprehensive performance evaluation of a tool.
        
        This is the main interface that other agents use to request
        performance evaluation of created tools.
        """
        logger.info(f"Starting performance evaluation for tool: {tool_name}")
        
        tool_id = f"{tool_name}_{int(time.time())}"
        
        try:
            # Evaluate each performance category
            correctness_metric = self._evaluate_correctness(tool_code, execution_results, test_results)
            efficiency_metric = self._evaluate_efficiency(tool_code, execution_results)
            quality_metric = self._evaluate_code_quality(tool_code)
            reusability_metric = self._evaluate_reusability(tool_code)
            
            metrics = [correctness_metric, efficiency_metric, quality_metric, reusability_metric]
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(metrics)
            
            # Determine letter grade
            grade_letter = self._determine_letter_grade(composite_score)
            
            # Analyze strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, composite_score)
            
            # Compare to historical performance
            comparison = self._compare_to_historical_performance(metrics)
            
            # Analyze trends
            trend_analysis = self._analyze_performance_trends(tool_name)
            
            # Create performance report
            report = PerformanceReport(
                tool_id=tool_id,
                tool_name=tool_name,
                evaluation_timestamp=datetime.now(),
                metrics=metrics,
                composite_score=composite_score,
                grade_letter=grade_letter,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                comparison_to_peers=comparison,
                trend_analysis=trend_analysis
            )
            
            # Record evaluation for learning
            self._record_evaluation(tool_code, report)
            
            # Update psychological state
            self._update_psychological_state(report)
            
            logger.info(f"Performance evaluation complete - Grade: {grade_letter}, "
                       f"Score: {composite_score:.2f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            
            # Return minimal report on failure
            return PerformanceReport(
                tool_id=tool_id,
                tool_name=tool_name,
                evaluation_timestamp=datetime.now(),
                metrics=[],
                composite_score=0.0,
                grade_letter='F',
                strengths=[],
                weaknesses=[f"Evaluation failed: {str(e)}"],
                recommendations=["Fix evaluation errors and resubmit"],
                comparison_to_peers=None,
                trend_analysis="Unable to analyze due to evaluation failure"
            )
    
    def _evaluate_correctness(self, code: str, execution_results: Optional[Dict], 
                            test_results: Optional[Dict]) -> PerformanceMetric:
        """Evaluate correctness of the tool"""
        evidence = []
        score = 0.5  # Default neutral score
        
        # Check for syntax errors
        try:
            ast.parse(code)
            evidence.append("Code parses without syntax errors")
            score += 0.2
        except SyntaxError as e:
            evidence.append(f"Syntax error detected: {e}")
            score = max(0.0, score - 0.3)
        
        # Analyze execution results
        if execution_results:
            if execution_results.get("success", False):
                evidence.append("Tool executed successfully")
                score += 0.2
            else:
                evidence.append(f"Execution failed: {execution_results.get('error', 'Unknown error')}")
                score = max(0.0, score - 0.2)
            
            # Check for expected outputs
            if "expected_output" in execution_results and "actual_output" in execution_results:
                if execution_results["expected_output"] == execution_results["actual_output"]:
                    evidence.append("Output matches expected results")
                    score += 0.2
                else:
                    evidence.append("Output differs from expected results")
                    score = max(0.0, score - 0.1)
        
        # Analyze test results
        if test_results:
            test_pass_rate = test_results.get("pass_rate", 0.0)
            evidence.append(f"Test pass rate: {test_pass_rate:.1%}")
            score = max(0.0, min(1.0, test_pass_rate))
        
        # Check for error handling
        if "try:" in code and "except:" in code:
            evidence.append("Includes error handling")
            score += 0.1
        
        # Check for input validation
        if re.search(r"if.*is None|if not.*:|raise.*Error", code):
            evidence.append("Includes input validation")
            score += 0.1
        
        score = min(1.0, score)
        
        improvements = []
        if score < 0.8:
            improvements.append("Add comprehensive error handling")
            improvements.append("Implement input validation")
            improvements.append("Add unit tests for edge cases")
        
        return PerformanceMetric(
            category=PerformanceCategory.CORRECTNESS,
            score=score,
            weight=self.category_weights[PerformanceCategory.CORRECTNESS],
            description=f"Correctness assessment based on syntax, execution, and testing",
            evidence=evidence,
            improvement_suggestions=improvements
        )
    
    def _evaluate_efficiency(self, code: str, execution_results: Optional[Dict]) -> PerformanceMetric:
        """Evaluate efficiency and performance"""
        evidence = []
        score = 0.7  # Default good score
        
        # Analyze execution time
        if execution_results and "execution_time" in execution_results:
            exec_time = execution_results["execution_time"]
            baseline_time = self.performance_baselines["average_execution_time"]
            
            if exec_time <= baseline_time * 0.5:
                evidence.append(f"Excellent execution time: {exec_time:.3f}s")
                score += 0.2
            elif exec_time <= baseline_time:
                evidence.append(f"Good execution time: {exec_time:.3f}s")
                score += 0.1
            elif exec_time <= baseline_time * 2:
                evidence.append(f"Acceptable execution time: {exec_time:.3f}s")
            else:
                evidence.append(f"Slow execution time: {exec_time:.3f}s")
                score -= 0.2
        
        # Analyze memory usage
        if execution_results and "memory_usage" in execution_results:
            memory_mb = execution_results["memory_usage"]
            baseline_memory = self.performance_baselines["average_memory_usage"]
            
            if memory_mb <= baseline_memory * 0.5:
                evidence.append(f"Excellent memory efficiency: {memory_mb:.1f}MB")
                score += 0.1
            elif memory_mb <= baseline_memory * 2:
                evidence.append(f"Good memory usage: {memory_mb:.1f}MB")
            else:
                evidence.append(f"High memory usage: {memory_mb:.1f}MB")
                score -= 0.1
        
        # Check for algorithmic efficiency patterns
        if re.search(r"for.*in.*range\s*\(\s*len\s*\(", code):
            evidence.append("Potential inefficient iteration pattern detected")
            score -= 0.1
        
        if "while True:" in code and "break" not in code:
            evidence.append("Potential infinite loop detected")
            score -= 0.3
        
        # Check for efficient data structures
        if re.search(r"set\(|dict\(|\{.*\}|\[.*for.*in.*\]", code):
            evidence.append("Uses efficient data structures")
            score += 0.1
        
        # Check for unnecessary operations
        nested_loops = len(re.findall(r"for.*in.*:", code))
        if nested_loops > 2:
            evidence.append(f"Multiple nested loops detected ({nested_loops})")
            score -= 0.1
        
        score = max(0.0, min(1.0, score))
        
        improvements = []
        if score < 0.7:
            improvements.append("Optimize algorithm complexity")
            improvements.append("Use more efficient data structures")
            improvements.append("Reduce unnecessary computations")
            improvements.append("Consider caching for repeated operations")
        
        return PerformanceMetric(
            category=PerformanceCategory.EFFICIENCY,
            score=score,
            weight=self.category_weights[PerformanceCategory.EFFICIENCY],
            description="Efficiency assessment based on execution time and resource usage",
            evidence=evidence,
            improvement_suggestions=improvements
        )
    
    def _evaluate_code_quality(self, code: str) -> PerformanceMetric:
        """Evaluate code quality and maintainability"""
        evidence = []
        score = 0.5  # Start neutral
        
        # Check naming conventions
        functions = re.findall(r"def\s+(\w+)", code)
        classes = re.findall(r"class\s+(\w+)", code)
        
        good_function_names = sum(1 for name in functions 
                                 if re.match(self.quality_standards["naming_conventions"]["function_pattern"], name))
        if functions:
            function_naming_score = good_function_names / len(functions)
            evidence.append(f"Function naming compliance: {function_naming_score:.1%}")
            score += function_naming_score * 0.1
        
        good_class_names = sum(1 for name in classes 
                              if re.match(self.quality_standards["naming_conventions"]["class_pattern"], name))
        if classes:
            class_naming_score = good_class_names / len(classes)
            evidence.append(f"Class naming compliance: {class_naming_score:.1%}")
            score += class_naming_score * 0.1
        
        # Check documentation
        docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
        total_functions_classes = len(functions) + len(classes)
        
        if total_functions_classes > 0:
            documentation_ratio = docstring_count / total_functions_classes
            evidence.append(f"Documentation coverage: {documentation_ratio:.1%}")
            score += documentation_ratio * 0.3
        
        # Check code length and complexity
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        code_length = len(lines)
        
        if code_length <= 30:
            evidence.append(f"Concise code length: {code_length} lines")
            score += 0.1
        elif code_length <= 100:
            evidence.append(f"Reasonable code length: {code_length} lines")
        else:
            evidence.append(f"Long code: {code_length} lines")
            score -= 0.1
        
        # Check for comments
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        if comment_lines > 0:
            comment_ratio = comment_lines / len(code.split('\n'))
            evidence.append(f"Comment ratio: {comment_ratio:.1%}")
            score += min(0.1, comment_ratio * 0.5)
        
        # Check for magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', code)
        if magic_numbers:
            evidence.append(f"Magic numbers detected: {len(magic_numbers)}")
            score -= min(0.1, len(magic_numbers) * 0.02)
        
        # Check for proper imports
        import_lines = [line for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
        if import_lines:
            evidence.append(f"Uses {len(import_lines)} imports")
            # Prefer specific imports over wildcard
            wildcard_imports = [line for line in import_lines if '*' in line]
            if wildcard_imports:
                evidence.append(f"Wildcard imports detected: {len(wildcard_imports)}")
                score -= len(wildcard_imports) * 0.05
        
        score = max(0.0, min(1.0, score))
        
        improvements = []
        if score < 0.7:
            improvements.append("Improve function and variable naming")
            improvements.append("Add comprehensive docstrings")
            improvements.append("Add explanatory comments")
            improvements.append("Replace magic numbers with named constants")
            improvements.append("Use specific imports instead of wildcards")
        
        return PerformanceMetric(
            category=PerformanceCategory.CODE_QUALITY,
            score=score,
            weight=self.category_weights[PerformanceCategory.CODE_QUALITY],
            description="Code quality assessment based on style, documentation, and maintainability",
            evidence=evidence,
            improvement_suggestions=improvements
        )
    
    def _evaluate_reusability(self, code: str) -> PerformanceMetric:
        """Evaluate reusability and modularity"""
        evidence = []
        score = 0.6  # Start with good baseline
        
        # Check for function definitions (modularity)
        functions = re.findall(r"def\s+(\w+)", code)
        if functions:
            evidence.append(f"Defines {len(functions)} functions")
            score += min(0.2, len(functions) * 0.05)
        else:
            evidence.append("No function definitions found")
            score -= 0.2
        
        # Check for classes (object-oriented design)
        classes = re.findall(r"class\s+(\w+)", code)
        if classes:
            evidence.append(f"Defines {len(classes)} classes")
            score += min(0.1, len(classes) * 0.05)
        
        # Check for parameterization
        function_params = re.findall(r"def\s+\w+\s*\(([^)]*)\)", code)
        if function_params:
            avg_params = sum(len(params.split(',')) if params.strip() else 0 for params in function_params) / len(function_params)
            if avg_params > 0:
                evidence.append(f"Average parameters per function: {avg_params:.1f}")
                score += min(0.1, avg_params * 0.02)
        
        # Check for default parameters (flexibility)
        default_params = len(re.findall(r"\w+\s*=\s*[^,)]+", code))
        if default_params > 0:
            evidence.append(f"Uses {default_params} default parameters")
            score += min(0.1, default_params * 0.02)
        
        # Check for return statements (functional design)
        returns = len(re.findall(r"return\s+", code))
        if returns > 0:
            evidence.append(f"Has {returns} return statements")
            score += min(0.1, returns * 0.01)
        
        # Check for global variables (reduces reusability)
        global_vars = len(re.findall(r"global\s+\w+", code))
        if global_vars > 0:
            evidence.append(f"Uses {global_vars} global variables")
            score -= global_vars * 0.05
        
        # Check for hardcoded values
        hardcoded_strings = len(re.findall(r'"[^"]{10,}"', code))
        hardcoded_numbers = len(re.findall(r'\b\d{3,}\b', code))
        
        if hardcoded_strings + hardcoded_numbers > 0:
            evidence.append(f"Hardcoded values detected: {hardcoded_strings + hardcoded_numbers}")
            score -= min(0.2, (hardcoded_strings + hardcoded_numbers) * 0.02)
        
        # Check for configuration/parameterization
        if re.search(r"config|settings|params|options", code, re.IGNORECASE):
            evidence.append("Includes configuration/parameterization")
            score += 0.1
        
        score = max(0.0, min(1.0, score))
        
        improvements = []
        if score < 0.7:
            improvements.append("Break code into smaller, focused functions")
            improvements.append("Add parameters to make functions more flexible")
            improvements.append("Use configuration instead of hardcoded values")
            improvements.append("Avoid global variables")
            improvements.append("Design for modularity and composition")
        
        return PerformanceMetric(
            category=PerformanceCategory.REUSABILITY,
            score=score,
            weight=self.category_weights[PerformanceCategory.REUSABILITY],
            description="Reusability assessment based on modularity and flexibility",
            evidence=evidence,
            improvement_suggestions=improvements
        )
    
    def _calculate_composite_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate weighted composite score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            total_weighted_score += metric.score * metric.weight
            total_weight += metric.weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0
    
    def _determine_letter_grade(self, composite_score: float) -> str:
        """Convert composite score to letter grade"""
        for grade, threshold in sorted(self.grade_boundaries.items(), 
                                     key=lambda x: x[1], reverse=True):
            if composite_score >= threshold:
                return grade
        return 'F'
    
    def _analyze_strengths_weaknesses(self, metrics: List[PerformanceMetric]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses from metrics"""
        strengths = []
        weaknesses = []
        
        for metric in metrics:
            if metric.score >= 0.8:
                strengths.append(f"Excellent {metric.category.value} (Score: {metric.score:.2f})")
            elif metric.score >= 0.7:
                strengths.append(f"Good {metric.category.value} (Score: {metric.score:.2f})")
            elif metric.score < 0.5:
                weaknesses.append(f"Poor {metric.category.value} (Score: {metric.score:.2f})")
            elif metric.score < 0.7:
                weaknesses.append(f"Below average {metric.category.value} (Score: {metric.score:.2f})")
        
        return strengths, weaknesses
    
    def _generate_recommendations(self, metrics: List[PerformanceMetric], composite_score: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Add metric-specific recommendations
        for metric in metrics:
            if metric.score < 0.7:
                recommendations.extend(metric.improvement_suggestions[:2])  # Top 2 suggestions
        
        # Add overall recommendations based on composite score
        if composite_score < 0.6:
            recommendations.append("Consider redesigning the approach for better overall performance")
        elif composite_score < 0.8:
            recommendations.append("Focus on addressing the weakest performance areas")
        else:
            recommendations.append("Excellent work! Consider minor optimizations for perfection")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _compare_to_historical_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Compare current performance to historical averages"""
        if not self.evaluation_history:
            return {"message": "No historical data available"}
        
        # Calculate historical averages for each category
        historical_averages = {}
        
        for category in PerformanceCategory:
            category_scores = []
            for record in self.evaluation_history:
                if "metrics" in record:
                    for metric_data in record["metrics"]:
                        if metric_data.get("category") == category.value:
                            category_scores.append(metric_data.get("score", 0.0))
            
            if category_scores:
                historical_averages[category.value] = statistics.mean(category_scores)
        
        # Compare current metrics to historical averages
        comparison = {}
        for metric in metrics:
            category = metric.category.value
            if category in historical_averages:
                historical_avg = historical_averages[category]
                improvement = metric.score - historical_avg
                comparison[category] = {
                    "current_score": metric.score,
                    "historical_average": historical_avg,
                    "improvement": improvement,
                    "trend": "improving" if improvement > 0.05 else "declining" if improvement < -0.05 else "stable"
                }
        
        return comparison
    
    def _analyze_performance_trends(self, tool_name: str) -> str:
        """Analyze performance trends for this tool or similar tools"""
        if not self.evaluation_history:
            return "No historical data available for trend analysis"
        
        # Look for similar tools or same tool over time
        similar_evaluations = [
            record for record in self.evaluation_history
            if record.get("tool_name", "").lower() in tool_name.lower() or 
               tool_name.lower() in record.get("tool_name", "").lower()
        ]
        
        if len(similar_evaluations) < 2:
            return "Insufficient data for trend analysis"
        
        # Analyze trend in composite scores
        scores = [record.get("composite_score", 0.0) for record in similar_evaluations[-5:]]  # Last 5
        
        if len(scores) >= 2:
            recent_avg = statistics.mean(scores[-2:])
            earlier_avg = statistics.mean(scores[:-2]) if len(scores) > 2 else scores[0]
            
            if recent_avg > earlier_avg + 0.1:
                return f"Strong improvement trend (recent avg: {recent_avg:.2f} vs earlier: {earlier_avg:.2f})"
            elif recent_avg > earlier_avg + 0.05:
                return f"Moderate improvement trend (recent avg: {recent_avg:.2f} vs earlier: {earlier_avg:.2f})"
            elif recent_avg < earlier_avg - 0.1:
                return f"Declining performance trend (recent avg: {recent_avg:.2f} vs earlier: {earlier_avg:.2f})"
            else:
                return f"Stable performance trend (avg: {recent_avg:.2f})"
        
        return "Unable to determine trend"
    
    def _record_evaluation(self, code: str, report: PerformanceReport):
        """Record evaluation for learning and trend analysis"""
        evaluation_record = {
            "timestamp": report.evaluation_timestamp.isoformat(),
            "tool_id": report.tool_id,
            "tool_name": report.tool_name,
            "composite_score": report.composite_score,
            "grade_letter": report.grade_letter,
            "code_length": len(code.split('\n')),
            "metrics": [
                {
                    "category": metric.category.value,
                    "score": metric.score,
                    "weight": metric.weight
                }
                for metric in report.metrics
            ]
        }
        
        self.evaluation_history.append(evaluation_record)
        
        # Keep only recent history (last 100 evaluations)
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]
    
    def _update_psychological_state(self, report: PerformanceReport):
        """Update agent's psychological state based on evaluation results"""
        
        # High-quality evaluations satisfy mastery and recognition drives
        satisfaction_level = report.composite_score
        
        experience = {
            "type": "performance_evaluation",
            "outcome": "success",
            "satisfaction": satisfaction_level,
            "intensity": 0.7
        }
        
        self.drive_system.process_experience(experience)
        
        # Boost confidence when giving high grades
        if report.grade_letter in ['A', 'B']:
            self.drive_system.personality.confidence = min(1.0, self.drive_system.personality.confidence + 0.05)
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about past evaluations"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        total_evaluations = len(self.evaluation_history)
        
        # Grade distribution
        grade_distribution = {}
        for record in self.evaluation_history:
            grade = record.get("grade_letter", "F")
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        # Average scores by category
        category_averages = {}
        for category in PerformanceCategory:
            scores = []
            for record in self.evaluation_history:
                for metric in record.get("metrics", []):
                    if metric.get("category") == category.value:
                        scores.append(metric.get("score", 0.0))
            
            if scores:
                category_averages[category.value] = statistics.mean(scores)
        
        # Overall statistics
        composite_scores = [record.get("composite_score", 0.0) for record in self.evaluation_history]
        
        return {
            "total_evaluations": total_evaluations,
            "average_composite_score": statistics.mean(composite_scores),
            "grade_distribution": grade_distribution,
            "category_averages": category_averages,
            "recent_trend": self._analyze_performance_trends("overall"),
            "evaluation_rate": f"{total_evaluations / max(1, len(self.evaluation_history))} evaluations per session"
        }

def create_grading_agent(llm_config: Dict[str, Any]) -> GradingAgent:
    """Factory function to create a functional GradingAgent"""
    return GradingAgent(llm_config)