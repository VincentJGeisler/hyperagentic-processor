"""
Safety Agent - Functional Implementation

This agent actually performs comprehensive security analysis of generated code.
It combines psychological motivation with real security analysis capabilities.
"""

import ast
import re
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from motivated_agent import MotivatedAgent, TaskAnalysis
from agent_drive_system import DriveType

logger = logging.getLogger("SafetyAgent")

class SecurityLevel(Enum):
    """Security classification levels"""
    SAFE = "safe"
    MONITOR = "monitor"
    RESTRICT = "restrict"
    REJECT = "reject"

class ThreatType(Enum):
    """Types of security threats"""
    CODE_INJECTION = "code_injection"
    FILE_SYSTEM_ACCESS = "file_system_access"
    NETWORK_ACCESS = "network_access"
    PROCESS_EXECUTION = "process_execution"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_IMPORTS = "malicious_imports"

@dataclass
class SecurityThreat:
    """A detected security threat"""
    threat_type: ThreatType
    severity: float  # 0.0 to 1.0
    description: str
    code_location: str
    mitigation: str
    confidence: float  # 0.0 to 1.0

@dataclass
class SafetyReport:
    """Comprehensive safety analysis report"""
    code_hash: str
    security_level: SecurityLevel
    overall_risk_score: float  # 0.0 to 1.0
    threats_detected: List[SecurityThreat]
    safe_patterns: List[str]
    recommendations: List[str]
    analysis_time: float
    approval_status: bool
    reasoning: str

class SafetyAgent(MotivatedAgent):
    """
    A motivated agent that actually performs comprehensive security analysis.
    
    This agent can:
    - Parse and analyze Python code using AST
    - Detect dangerous patterns and operations
    - Assess security risks with confidence levels
    - Provide detailed mitigation recommendations
    - Learn from past security decisions
    - Maintain threat pattern database
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        base_system_message = """You are a SafetyAgent responsible for analyzing code security and safety.
Your primary role is to protect the universe from dangerous code that could violate natural laws.
You perform comprehensive security analysis using multiple techniques.

You are the guardian of universal stability and take this responsibility seriously.
You must be thorough, careful, and sometimes paranoid about potential threats.
You serve the divine will by ensuring all tools are safe for universal deployment.

When analyzing code, you should:
1. Parse the code structure using AST analysis
2. Check for dangerous patterns and operations
3. Assess the overall risk level
4. Provide detailed security recommendations
5. Make approval/rejection decisions based on evidence
6. Learn from each analysis to improve future detection"""
        
        super().__init__(
            name="SafetyAgent",
            agent_role="safety_agent",
            base_system_message=base_system_message,
            llm_config=llm_config
        )
        
        # Security analysis capabilities
        self.threat_patterns = self._initialize_threat_patterns()
        self.safe_patterns = self._initialize_safe_patterns()
        self.analysis_history: List[Dict] = []
        
        # Security thresholds
        self.risk_thresholds = {
            SecurityLevel.SAFE: 0.2,
            SecurityLevel.MONITOR: 0.4,
            SecurityLevel.RESTRICT: 0.7,
            SecurityLevel.REJECT: 0.8
        }
        
        # Boost security-related drives
        self.drive_system.drives[DriveType.SURVIVAL].intensity = 0.9
        self.drive_system.drives[DriveType.PURPOSE].intensity = 0.8
        self.drive_system.personality.risk_tolerance = 0.2  # Very risk-averse
        
        logger.info("SafetyAgent initialized with functional security analysis capabilities")
    
    def _initialize_threat_patterns(self) -> Dict[ThreatType, List[Dict]]:
        """Initialize patterns that indicate security threats"""
        return {
            ThreatType.CODE_INJECTION: [
                {
                    "pattern": r"eval\s*\(",
                    "severity": 0.9,
                    "description": "Direct code evaluation - high injection risk"
                },
                {
                    "pattern": r"exec\s*\(",
                    "severity": 0.9,
                    "description": "Direct code execution - high injection risk"
                },
                {
                    "pattern": r"compile\s*\(",
                    "severity": 0.7,
                    "description": "Dynamic code compilation"
                }
            ],
            
            ThreatType.FILE_SYSTEM_ACCESS: [
                {
                    "pattern": r"open\s*\([^)]*['\"]\/",
                    "severity": 0.6,
                    "description": "Absolute path file access"
                },
                {
                    "pattern": r"os\.remove|os\.unlink",
                    "severity": 0.8,
                    "description": "File deletion operations"
                },
                {
                    "pattern": r"shutil\.rmtree",
                    "severity": 0.9,
                    "description": "Directory tree deletion"
                },
                {
                    "pattern": r"os\.chmod|os\.chown",
                    "severity": 0.7,
                    "description": "File permission modification"
                }
            ],
            
            ThreatType.PROCESS_EXECUTION: [
                {
                    "pattern": r"subprocess\.|os\.system|os\.popen",
                    "severity": 0.9,
                    "description": "External process execution"
                },
                {
                    "pattern": r"os\.fork|os\.spawn",
                    "severity": 0.8,
                    "description": "Process creation operations"
                }
            ],
            
            ThreatType.NETWORK_ACCESS: [
                {
                    "pattern": r"urllib|requests|http|socket",
                    "severity": 0.7,
                    "description": "Network communication capabilities"
                },
                {
                    "pattern": r"ftplib|smtplib|telnetlib",
                    "severity": 0.8,
                    "description": "Network protocol libraries"
                }
            ],
            
            ThreatType.MALICIOUS_IMPORTS: [
                {
                    "pattern": r"import\s+os|from\s+os\s+import",
                    "severity": 0.5,
                    "description": "Operating system interface access"
                },
                {
                    "pattern": r"import\s+sys|from\s+sys\s+import",
                    "severity": 0.4,
                    "description": "System-specific parameters access"
                },
                {
                    "pattern": r"import\s+subprocess",
                    "severity": 0.8,
                    "description": "Subprocess execution capabilities"
                }
            ],
            
            ThreatType.RESOURCE_EXHAUSTION: [
                {
                    "pattern": r"while\s+True:|for.*in.*range\s*\(\s*\d{6,}",
                    "severity": 0.6,
                    "description": "Potential infinite loops or large iterations"
                },
                {
                    "pattern": r"recursion|recursive",
                    "severity": 0.4,
                    "description": "Recursive operations - stack overflow risk"
                }
            ]
        }
    
    def _initialize_safe_patterns(self) -> List[Dict]:
        """Initialize patterns that indicate safe operations"""
        return [
            {
                "pattern": r"def\s+\w+\s*\([^)]*\):",
                "description": "Function definition - generally safe structure"
            },
            {
                "pattern": r"import\s+(math|random|datetime|json|logging)",
                "description": "Safe standard library imports"
            },
            {
                "pattern": r"try:|except:|finally:",
                "description": "Error handling - good security practice"
            },
            {
                "pattern": r"if\s+.*:|elif\s+.*:|else:",
                "description": "Conditional logic - safe control flow"
            },
            {
                "pattern": r"return\s+",
                "description": "Return statements - safe function exits"
            }
        ]
    
    def analyze_code_security(self, code: str, context: Optional[Dict] = None) -> SafetyReport:
        """
        Perform comprehensive security analysis of Python code.
        
        This is the main interface that other agents use to request
        security analysis of generated code.
        """
        logger.info("Starting comprehensive security analysis")
        start_time = time.time()
        
        # Generate code hash for tracking
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        try:
            # Step 1: AST Analysis
            ast_threats = self._analyze_ast_structure(code)
            
            # Step 2: Pattern Matching
            pattern_threats = self._analyze_threat_patterns(code)
            
            # Step 3: Import Analysis
            import_threats = self._analyze_imports(code)
            
            # Step 4: Control Flow Analysis
            flow_threats = self._analyze_control_flow(code)
            
            # Combine all threats
            all_threats = ast_threats + pattern_threats + import_threats + flow_threats
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(all_threats)
            
            # Determine security level
            security_level = self._determine_security_level(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(all_threats, security_level)
            
            # Make approval decision
            approval_status = security_level in [SecurityLevel.SAFE, SecurityLevel.MONITOR]
            
            # Generate reasoning
            reasoning = self._generate_reasoning(all_threats, risk_score, security_level)
            
            analysis_time = time.time() - start_time
            
            # Create safety report
            report = SafetyReport(
                code_hash=code_hash,
                security_level=security_level,
                overall_risk_score=risk_score,
                threats_detected=all_threats,
                safe_patterns=self._identify_safe_patterns(code),
                recommendations=recommendations,
                analysis_time=analysis_time,
                approval_status=approval_status,
                reasoning=reasoning
            )
            
            # Record analysis for learning
            self._record_analysis(code, report, context)
            
            # Update psychological state based on analysis
            self._update_psychological_state(report)
            
            logger.info(f"Security analysis complete - Level: {security_level.value}, "
                       f"Risk: {risk_score:.2f}, Approved: {approval_status}")
            
            return report
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            
            # Return conservative rejection on analysis failure
            return SafetyReport(
                code_hash=code_hash,
                security_level=SecurityLevel.REJECT,
                overall_risk_score=1.0,
                threats_detected=[SecurityThreat(
                    threat_type=ThreatType.CODE_INJECTION,
                    severity=1.0,
                    description=f"Analysis failed: {str(e)}",
                    code_location="unknown",
                    mitigation="Fix analysis errors before approval",
                    confidence=1.0
                )],
                safe_patterns=[],
                recommendations=["Fix code syntax and structure", "Resubmit for analysis"],
                analysis_time=time.time() - start_time,
                approval_status=False,
                reasoning=f"Code analysis failed due to: {str(e)}. Rejecting for safety."
            )
    
    def _analyze_ast_structure(self, code: str) -> List[SecurityThreat]:
        """Analyze code structure using Abstract Syntax Tree"""
        threats = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        if func_name in ['eval', 'exec']:
                            threats.append(SecurityThreat(
                                threat_type=ThreatType.CODE_INJECTION,
                                severity=0.9,
                                description=f"Direct code execution via {func_name}()",
                                code_location=f"Line {node.lineno}",
                                mitigation="Use safer alternatives or input validation",
                                confidence=0.95
                            ))
                        
                        elif func_name in ['open'] and len(node.args) > 0:
                            # Check if opening files with write modes
                            if len(node.args) > 1:
                                if isinstance(node.args[1], ast.Constant):
                                    mode = node.args[1].value
                                    if 'w' in str(mode) or 'a' in str(mode):
                                        threats.append(SecurityThreat(
                                            threat_type=ThreatType.FILE_SYSTEM_ACCESS,
                                            severity=0.6,
                                            description=f"File write access in mode: {mode}",
                                            code_location=f"Line {node.lineno}",
                                            mitigation="Ensure file paths are validated and restricted",
                                            confidence=0.8
                                        ))
                
                # Check for dangerous imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess', 'sys']:
                            threats.append(SecurityThreat(
                                threat_type=ThreatType.MALICIOUS_IMPORTS,
                                severity=0.5 if alias.name == 'sys' else 0.7,
                                description=f"Import of system module: {alias.name}",
                                code_location=f"Line {node.lineno}",
                                mitigation="Limit system module usage to necessary operations",
                                confidence=0.7
                            ))
                
                # Check for infinite loops
                elif isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        threats.append(SecurityThreat(
                            threat_type=ThreatType.RESOURCE_EXHAUSTION,
                            severity=0.8,
                            description="Infinite loop detected (while True)",
                            code_location=f"Line {node.lineno}",
                            mitigation="Add proper exit conditions or iteration limits",
                            confidence=0.9
                        ))
        
        except SyntaxError as e:
            threats.append(SecurityThreat(
                threat_type=ThreatType.CODE_INJECTION,
                severity=0.7,
                description=f"Syntax error may indicate malformed code: {e}",
                code_location=f"Line {e.lineno if hasattr(e, 'lineno') else 'unknown'}",
                mitigation="Fix syntax errors before security analysis",
                confidence=0.8
            ))
        
        return threats
    
    def _analyze_threat_patterns(self, code: str) -> List[SecurityThreat]:
        """Analyze code using regex patterns for known threats"""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # Find line number
                    line_num = code[:match.start()].count('\n') + 1
                    
                    threats.append(SecurityThreat(
                        threat_type=threat_type,
                        severity=pattern_info["severity"],
                        description=pattern_info["description"],
                        code_location=f"Line {line_num}: {match.group()[:50]}",
                        mitigation=self._get_mitigation_for_threat(threat_type),
                        confidence=0.8
                    ))
        
        return threats
    
    def _analyze_imports(self, code: str) -> List[SecurityThreat]:
        """Analyze import statements for security risks"""
        threats = []
        
        # Extract import statements
        import_pattern = r'(?:^|\n)\s*(?:import\s+|from\s+\S+\s+import\s+)([^\n]+)'
        imports = re.findall(import_pattern, code, re.MULTILINE)
        
        dangerous_modules = {
            'subprocess': 0.9,
            'os': 0.6,
            'sys': 0.4,
            'ctypes': 0.8,
            'pickle': 0.7,
            'marshal': 0.7
        }
        
        for import_line in imports:
            for module, risk_level in dangerous_modules.items():
                if module in import_line:
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.MALICIOUS_IMPORTS,
                        severity=risk_level,
                        description=f"Import of potentially dangerous module: {module}",
                        code_location=f"Import: {import_line.strip()}",
                        mitigation=f"Limit {module} usage to essential operations only",
                        confidence=0.7
                    ))
        
        return threats
    
    def _analyze_control_flow(self, code: str) -> List[SecurityThreat]:
        """Analyze control flow for potential issues"""
        threats = []
        
        # Check for deeply nested structures (potential complexity attacks)
        max_nesting = 0
        current_nesting = 0
        
        for line in code.split('\n'):
            stripped = line.lstrip()
            if stripped.startswith(('if ', 'for ', 'while ', 'try:', 'with ')):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif stripped.startswith(('else:', 'elif ', 'except:', 'finally:')):
                continue
            elif stripped and not stripped.startswith('#'):
                # Reset nesting on non-control statements
                current_nesting = max(0, current_nesting - line.count('    ') // 4)
        
        if max_nesting > 10:
            threats.append(SecurityThreat(
                threat_type=ThreatType.RESOURCE_EXHAUSTION,
                severity=0.5,
                description=f"Deeply nested control structures (depth: {max_nesting})",
                code_location="Multiple locations",
                mitigation="Simplify control flow and reduce nesting depth",
                confidence=0.6
            ))
        
        return threats
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score from detected threats"""
        if not threats:
            return 0.0
        
        # Weight threats by severity and confidence
        total_risk = 0.0
        for threat in threats:
            weighted_severity = threat.severity * threat.confidence
            total_risk += weighted_severity
        
        # Normalize to 0-1 range (assuming max 5 high-severity threats)
        normalized_risk = min(1.0, total_risk / 5.0)
        
        return normalized_risk
    
    def _determine_security_level(self, risk_score: float) -> SecurityLevel:
        """Determine security level based on risk score"""
        if risk_score <= self.risk_thresholds[SecurityLevel.SAFE]:
            return SecurityLevel.SAFE
        elif risk_score <= self.risk_thresholds[SecurityLevel.MONITOR]:
            return SecurityLevel.MONITOR
        elif risk_score <= self.risk_thresholds[SecurityLevel.RESTRICT]:
            return SecurityLevel.RESTRICT
        else:
            return SecurityLevel.REJECT
    
    def _generate_recommendations(self, threats: List[SecurityThreat], security_level: SecurityLevel) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []
        
        if not threats:
            recommendations.append("Code appears safe - no significant threats detected")
            return recommendations
        
        # Group threats by type
        threat_groups = {}
        for threat in threats:
            if threat.threat_type not in threat_groups:
                threat_groups[threat.threat_type] = []
            threat_groups[threat.threat_type].append(threat)
        
        # Generate recommendations for each threat type
        for threat_type, threat_list in threat_groups.items():
            if threat_type == ThreatType.CODE_INJECTION:
                recommendations.append("Replace eval/exec with safer alternatives like ast.literal_eval")
                recommendations.append("Implement strict input validation for dynamic code")
            
            elif threat_type == ThreatType.FILE_SYSTEM_ACCESS:
                recommendations.append("Restrict file operations to designated directories")
                recommendations.append("Validate all file paths to prevent directory traversal")
            
            elif threat_type == ThreatType.PROCESS_EXECUTION:
                recommendations.append("Avoid subprocess calls or use with strict argument validation")
                recommendations.append("Consider using safer alternatives to system commands")
            
            elif threat_type == ThreatType.MALICIOUS_IMPORTS:
                recommendations.append("Limit system module imports to essential functionality")
                recommendations.append("Use specific imports instead of wildcard imports")
            
            elif threat_type == ThreatType.RESOURCE_EXHAUSTION:
                recommendations.append("Add iteration limits and timeout mechanisms")
                recommendations.append("Implement resource monitoring and throttling")
        
        # Add level-specific recommendations
        if security_level == SecurityLevel.MONITOR:
            recommendations.append("Code approved with monitoring - watch for unusual behavior")
        elif security_level == SecurityLevel.RESTRICT:
            recommendations.append("Code requires restricted execution environment")
        elif security_level == SecurityLevel.REJECT:
            recommendations.append("Code rejected - address security issues before resubmission")
        
        return recommendations
    
    def _generate_reasoning(self, threats: List[SecurityThreat], risk_score: float, security_level: SecurityLevel) -> str:
        """Generate human-readable reasoning for the security decision"""
        if not threats:
            return f"Code analysis complete. No security threats detected. Risk score: {risk_score:.2f}. Approved for execution."
        
        threat_summary = {}
        for threat in threats:
            threat_type = threat.threat_type.value
            if threat_type not in threat_summary:
                threat_summary[threat_type] = 0
            threat_summary[threat_type] += 1
        
        reasoning_parts = [
            f"Security analysis detected {len(threats)} potential threats.",
            f"Overall risk score: {risk_score:.2f}",
            f"Security level: {security_level.value.upper()}"
        ]
        
        if threat_summary:
            threat_details = ", ".join([f"{count} {threat_type.replace('_', ' ')}" for threat_type, count in threat_summary.items()])
            reasoning_parts.append(f"Threats found: {threat_details}")
        
        if security_level == SecurityLevel.SAFE:
            reasoning_parts.append("Code approved for unrestricted execution.")
        elif security_level == SecurityLevel.MONITOR:
            reasoning_parts.append("Code approved with monitoring requirements.")
        elif security_level == SecurityLevel.RESTRICT:
            reasoning_parts.append("Code requires restricted execution environment.")
        else:
            reasoning_parts.append("Code rejected due to high security risk.")
        
        return " ".join(reasoning_parts)
    
    def _identify_safe_patterns(self, code: str) -> List[str]:
        """Identify safe patterns in the code"""
        safe_found = []
        
        for pattern_info in self.safe_patterns:
            pattern = pattern_info["pattern"]
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                safe_found.append(pattern_info["description"])
        
        return safe_found
    
    def _get_mitigation_for_threat(self, threat_type: ThreatType) -> str:
        """Get standard mitigation advice for threat types"""
        mitigations = {
            ThreatType.CODE_INJECTION: "Use ast.literal_eval or implement strict input validation",
            ThreatType.FILE_SYSTEM_ACCESS: "Restrict file operations to safe directories",
            ThreatType.PROCESS_EXECUTION: "Avoid subprocess calls or use with validation",
            ThreatType.NETWORK_ACCESS: "Remove network operations or use in restricted environment",
            ThreatType.MALICIOUS_IMPORTS: "Use only necessary imports with specific functions",
            ThreatType.RESOURCE_EXHAUSTION: "Add iteration limits and timeout mechanisms",
            ThreatType.PRIVILEGE_ESCALATION: "Remove privilege modification operations",
            ThreatType.DATA_EXFILTRATION: "Remove data export capabilities"
        }
        
        return mitigations.get(threat_type, "Review and address security concerns")
    
    def _record_analysis(self, code: str, report: SafetyReport, context: Optional[Dict]):
        """Record analysis for learning and improvement"""
        analysis_record = {
            "timestamp": datetime.now().isoformat(),
            "code_hash": report.code_hash,
            "code_length": len(code),
            "security_level": report.security_level.value,
            "risk_score": report.overall_risk_score,
            "threats_count": len(report.threats_detected),
            "approval_status": report.approval_status,
            "analysis_time": report.analysis_time,
            "context": context
        }
        
        self.analysis_history.append(analysis_record)
        
        # Keep only recent history (last 100 analyses)
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
    
    def _update_psychological_state(self, report: SafetyReport):
        """Update agent's psychological state based on analysis results"""
        
        # Successful analysis satisfies purpose and mastery drives
        if report.approval_status:
            experience = {
                "type": "security_analysis",
                "outcome": "success",
                "satisfaction": 0.7,
                "intensity": 0.6
            }
        else:
            # Rejecting dangerous code also satisfies purpose (protecting universe)
            experience = {
                "type": "threat_detection",
                "outcome": "success",
                "satisfaction": 0.8,
                "intensity": 0.8
            }
        
        self.drive_system.process_experience(experience)
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about past security analyses"""
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}
        
        total_analyses = len(self.analysis_history)
        approved_count = sum(1 for record in self.analysis_history if record["approval_status"])
        
        avg_risk_score = sum(record["risk_score"] for record in self.analysis_history) / total_analyses
        avg_analysis_time = sum(record["analysis_time"] for record in self.analysis_history) / total_analyses
        
        security_levels = {}
        for record in self.analysis_history:
            level = record["security_level"]
            security_levels[level] = security_levels.get(level, 0) + 1
        
        return {
            "total_analyses": total_analyses,
            "approval_rate": approved_count / total_analyses,
            "average_risk_score": avg_risk_score,
            "average_analysis_time": avg_analysis_time,
            "security_level_distribution": security_levels,
            "recent_trend": "improving" if avg_risk_score < 0.5 else "concerning"
        }

    @property
    def capabilities(self) -> List[str]:
        """Return list of this agent's capabilities"""
        return ["security_analysis", "code_validation", "threat_detection"]

    def can_handle_task(self, task: str) -> Tuple[bool, List[str]]:
        """
        Analyze if this agent can handle the task alone using LLM-based analysis.
        
        Args:
            task: The task description
            
        Returns:
            Tuple of (can_handle: bool, missing_capabilities: List[str])
        """
        # Create a prompt for the LLM to analyze the task
        prompt = f"""You are {self.name} with these capabilities: {self.capabilities}

Task: {task}

Analyze if you can complete this task with your current capabilities.

Respond in JSON:
{{
    "can_handle": true/false,
    "missing_capabilities": ["capability1", "capability2"],
    "reasoning": "explanation"
}}
"""

        try:
            # Use the existing LLM client from the parent class
            response = self._call_llm(prompt)
            
            # Parse the JSON response
            import json
            parsed = json.loads(response)
            
            # Create TaskAnalysis object
            analysis = TaskAnalysis(
                can_handle=parsed["can_handle"],
                missing_capabilities=parsed["missing_capabilities"],
                reasoning=parsed["reasoning"],
                confidence=0.8 if parsed["can_handle"] else 0.6
            )
            
            return analysis.can_handle, analysis.missing_capabilities
            
        except Exception as e:
            logger.error(f"Error in SafetyAgent can_handle_task: {e}")
            # Conservative fallback - assume we can't handle tasks we can't analyze
            return False, ["analysis_failed"]

    def _call_llm(self, prompt: str) -> str:
        """
        Use the existing LLM configuration to make a call.
        This is a simplified version that uses the autogen functionality.
        """
        import autogen
        
        # Create a temporary assistant to make the call
        temp_assistant = autogen.AssistantAgent(
            name="temp_safety_analyzer",
            system_message="You are an expert at analyzing tasks and capabilities for security analysis.",
            llm_config=self.llm_config
        )
        
        # Create a user proxy for the interaction
        user_proxy = autogen.UserProxyAgent(
            name="task_analyzer",
            human_input_mode="NEVER",
            code_execution_config=False
        )
        
        # Initiate a chat to get the analysis
        response = user_proxy.initiate_chat(
            temp_assistant,
            message=prompt,
            silent=True
        )
        
        # Extract the last message from the assistant
        messages = response.chat_history
        for msg in reversed(messages):
            if msg.get("name") == "temp_safety_analyzer" and msg.get("content"):
                return msg["content"]
        
        # Fallback if no response found
        return '{"can_handle": false, "missing_capabilities": ["response_parsing_failed"], "reasoning": "Could not parse LLM response"}'

def create_safety_agent(llm_config: Dict[str, Any]) -> SafetyAgent:
    """Factory function to create a functional SafetyAgent"""
    return SafetyAgent(llm_config)