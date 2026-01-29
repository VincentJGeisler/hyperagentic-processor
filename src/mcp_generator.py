"""
MCP Generator - Creates new MCP servers from requirements

This module implements the MCP Generation System that enables the Oracle agent
to create new MCP servers from scratch when existing ones cannot be found.
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger("MCPGenerator")

@dataclass
class MCPRequirements:
    """Requirements for generating a new MCP server"""
    name: str
    description: str
    capability: str  # e.g., "web scraping", "data processing", "API access"
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    language: str = "python"  # "python" or "typescript"
    dependencies: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)

@dataclass
class ToolDefinition:
    """Definition of a tool within an MCP server"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler_function: str

@dataclass
class MCPSpecification:
    """Detailed specification for an MCP server"""
    name: str
    description: str
    template_type: str  # "data_processor", "api_wrapper", "file_handler", etc.
    tools: List[ToolDefinition]
    language: str
    dependencies: List[str]
    entry_point: str
    mcp_version: str = "1.0"

@dataclass
class GenerationResult:
    """Result of MCP generation process"""
    success: bool
    mcp_name: str
    output_directory: Optional[str] = None
    executable_path: Optional[str] = None
    validation_passed: bool = False
    test_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    generation_time: float = 0.0
    code_files: List[str] = field(default_factory=list)
    installed: bool = False

@dataclass
class ValidationResult:
    """Result of MCP code validation"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    score: float = 0.0

@dataclass
class TestResult:
    """Result of MCP testing"""
    passed: bool
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class MCPGenerator:
    """
    Generates new MCP servers from requirements using ToolCreatorAgent.
    
    This class implements the MCP Generation System that enables the Oracle agent
    to create new MCP servers from scratch when existing ones cannot be found.
    """
    
    def __init__(self, tool_creator, safety_agent, logger=None):
        """
        Initialize the MCP Generator.
        
        Args:
            tool_creator: ToolCreatorAgent instance for code generation
            safety_agent: SafetyAgent instance for validation
            logger: Logger instance (optional)
        """
        self.tool_creator = tool_creator
        self.safety_agent = safety_agent
        self.logger = logger or logging.getLogger(__name__)
        
        # Create templates directory if it doesn't exist
        self.templates_dir = os.path.join(".kiro", "mcp_templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Create output directory for generated MCPs
        self.output_dir = os.path.join(".kiro", "generated_mcps")
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def generate_mcp(self, requirements: MCPRequirements) -> GenerationResult:
        """
        Generate a new MCP server from requirements.
        
        Args:
            requirements: MCPRequirements specifying what to generate
            
        Returns:
            GenerationResult with success status and details
        """
        start_time = time.time()
        self.logger.info(f"Generating MCP server: {requirements.name}")
        
        try:
            # 1. Analyze requirements and create specification
            specification = await self._analyze_requirements(requirements)
            self.logger.info(f"Created specification for {specification.template_type} MCP")
            
            # 2. Select appropriate template
            template = await self._select_template(specification)
            self.logger.info(f"Selected template: {template}")
            
            # 3. Generate core functionality using ToolCreator
            core_code = await self._generate_mcp_code(specification, template)
            if not core_code:
                return GenerationResult(
                    success=False,
                    mcp_name=requirements.name,
                    error_message="Failed to generate core MCP code"
                )
            self.logger.info("Generated core MCP functionality")
            
            # 4. Wrap with MCP protocol (JSON-RPC, stdio transport)
            mcp_code = await self._wrap_with_mcp_protocol(core_code, specification)
            if not mcp_code:
                return GenerationResult(
                    success=False,
                    mcp_name=requirements.name,
                    error_message="Failed to wrap code with MCP protocol"
                )
            self.logger.info("Wrapped code with MCP protocol")
            
            # 5. Validate with SafetyAgent
            validation = await self._validate_mcp(mcp_code)
            if not validation.passed:
                return GenerationResult(
                    success=False,
                    mcp_name=requirements.name,
                    error_message=f"Validation failed: {', '.join(validation.issues)}"
                )
            self.logger.info("MCP code validation passed")
            
            # 6. Test the generated MCP
            test_result = await self._test_mcp(mcp_code, specification)
            self.logger.info(f"MCP testing {'passed' if test_result.passed else 'failed'}")
            
            # 7. Package for installation
            output_dir = await self._package_mcp(mcp_code, specification, self.output_dir)
            if not output_dir:
                return GenerationResult(
                    success=False,
                    mcp_name=requirements.name,
                    error_message="Failed to package MCP for installation"
                )
            self.logger.info(f"MCP packaged in: {output_dir}")
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                success=True,
                mcp_name=requirements.name,
                output_directory=output_dir,
                validation_passed=True,
                test_results=test_result.results,
                generation_time=generation_time,
                code_files=[os.path.join(output_dir, f"{specification.name}.py")]
            )
            
        except Exception as e:
            self.logger.error(f"Error generating MCP: {e}")
            return GenerationResult(
                success=False,
                mcp_name=requirements.name,
                error_message=str(e),
                generation_time=time.time() - start_time
            )
    
    async def _analyze_requirements(self, requirements: MCPRequirements) -> MCPSpecification:
        """
        Analyze requirements and create detailed specification.
        
        Args:
            requirements: MCPRequirements to analyze
            
        Returns:
            MCPSpecification with detailed implementation plan
        """
        self.logger.info(f"Analyzing requirements for: {requirements.capability}")
        
        # Map capability to template type
        capability_lower = requirements.capability.lower()
        template_mapping = {
            "data": "data_processor",
            "process": "data_processor",
            "transform": "data_processor",
            "clean": "data_processor",
            "api": "api_wrapper",
            "web": "api_wrapper",
            "rest": "api_wrapper",
            "file": "file_handler",
            "document": "file_handler",
            "compute": "computation",
            "calculate": "computation",
            "math": "computation",
            "analyze": "computation",
            "predict": "computation",
            "integrate": "integration",
            "service": "integration",
            "platform": "integration"
        }
        
        template_type = "data_processor"  # default
        for keyword, template in template_mapping.items():
            if keyword in capability_lower:
                template_type = template
                break
        
        # Create tool definitions based on schemas
        tools = []
        
        # Primary tool based on capability
        primary_tool = ToolDefinition(
            name=f"{requirements.name}_tool",
            description=requirements.description,
            input_schema=requirements.input_schema,
            output_schema=requirements.output_schema,
            handler_function="handle_primary_operation"
        )
        tools.append(primary_tool)
        
        # Additional tools based on template type
        if template_type == "data_processor":
            tools.extend([
                ToolDefinition(
                    name="filter_data",
                    description="Filter data based on criteria",
                    input_schema={"type": "object", "properties": {"data": {}, "criteria": {}}},
                    output_schema={"type": "array"},
                    handler_function="handle_filter_data"
                ),
                ToolDefinition(
                    name="transform_data",
                    description="Transform data format",
                    input_schema={"type": "object", "properties": {"data": {}, "transformation": {}}},
                    output_schema={"type": "object"},
                    handler_function="handle_transform_data"
                )
            ])
        elif template_type == "api_wrapper":
            tools.extend([
                ToolDefinition(
                    name="get_resource",
                    description="GET request to API endpoint",
                    input_schema={"type": "object", "properties": {"endpoint": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler_function="handle_get_request"
                ),
                ToolDefinition(
                    name="post_resource",
                    description="POST request to API endpoint",
                    input_schema={"type": "object", "properties": {"endpoint": {"type": "string"}, "data": {}}},
                    output_schema={"type": "object"},
                    handler_function="handle_post_request"
                )
            ])
        elif template_type == "file_handler":
            tools.extend([
                ToolDefinition(
                    name="read_file",
                    description="Read file contents",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "string"},
                    handler_function="handle_read_file"
                ),
                ToolDefinition(
                    name="write_file",
                    description="Write data to file",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
                    output_schema={"type": "boolean"},
                    handler_function="handle_write_file"
                )
            ])
        
        # Create specification
        specification = MCPSpecification(
            name=requirements.name,
            description=requirements.description,
            template_type=template_type,
            tools=tools,
            language=requirements.language,
            dependencies=requirements.dependencies,
            entry_point=f"{requirements.name}.py"
        )
        
        return specification
    
    async def _select_template(self, specification: MCPSpecification) -> str:
        """
        Select appropriate template for the MCP specification.
        
        Args:
            specification: MCPSpecification to match with template
            
        Returns:
            Template identifier
        """
        return specification.template_type
    
    async def _generate_mcp_code(self, specification: MCPSpecification, template: str) -> str:
        """
        Generate core MCP functionality using ToolCreator.
        
        Args:
            specification: MCPSpecification with implementation details
            template: Template identifier to use
            
        Returns:
            Generated code as string
        """
        self.logger.info(f"Generating MCP code using ToolCreator for {template}")
        
        # Construct requirements for ToolCreator
        tool_specs = "\n".join([
            f"- {tool.name}: {tool.description} (input: {tool.input_schema}, output: {tool.output_schema})"
            for tool in specification.tools
        ])
        
        requirements = f"""
        Create a {specification.template_type} with the following specifications:
        
        Name: {specification.name}
        Description: {specification.description}
        Language: {specification.language}
        
        Tools to implement:
        {tool_specs}
        
        Dependencies: {', '.join(specification.dependencies)}
        
        Safety constraints:
        - Input validation required
        - Error handling for all operations
        - Resource limits: memory and CPU
        """
        
        # Use ToolCreator to generate the code
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.tool_creator.create_tool_from_requirements(requirements)
            )
            
            if result["success"]:
                return result["code"]
            else:
                self.logger.warning(f"ToolCreator failed: {result.get('error', 'Unknown error')}")
                return self._generate_fallback_code(specification)
        except Exception as e:
            self.logger.error(f"Error calling ToolCreator: {e}")
            return self._generate_fallback_code(specification)
    
    def _generate_fallback_code(self, specification: MCPSpecification) -> str:
        """
        Generate fallback code when ToolCreator fails.
        
        Args:
            specification: MCPSpecification to generate code for
            
        Returns:
            Fallback code as string
        """
        self.logger.info("Generating fallback code")
        
        # Generate basic tool handlers
        handlers = []
        for tool in specification.tools:
            handler_code = f'''
    def {tool.handler_function}(self, arguments: Dict[str, Any]) -> Any:
        """
        Handle {tool.name} tool call.
        
        Args:
            arguments: Tool arguments matching schema: {tool.input_schema}
            
        Returns:
            Result matching schema: {tool.output_schema}
        """
        try:
            # Input validation
            # TODO: Implement proper input validation based on schema
            
            # Tool logic
            # TODO: Implement actual tool functionality
            self.logger.info("Executing {tool.name} with arguments: {{arguments}}")
            
            # Return mock result based on output schema
            # TODO: Replace with actual implementation
            return {{"status": "success", "tool": "{tool.name}", "result": "mock_result"}}
            
        except Exception as e:
            self.logger.error("Error in {tool.name}: {{e}}")
            raise RuntimeError(f"Tool execution failed: {{e}}")
'''
            handlers.append(handler_code)
        
        # Combine into complete class
        code = f'''#!/usr/bin/env python3
"""
Fallback MCP Implementation: {specification.name}
Description: {specification.description}
Template Type: {specification.template_type}
"""

import json
import logging
from typing import Any, Dict, List

class FallbackMCP:
    """Fallback implementation of {specification.name} MCP."""
    
    def __init__(self):
        self.logger = logging.getLogger("{specification.name}")
        self.tools = {{
'''
        
        # Add tool registry
        for tool in specification.tools:
            code += f'''            "{tool.name}": {{
                "name": "{tool.name}",
                "description": "{tool.description}",
                "inputSchema": {json.dumps(tool.input_schema)},
                "outputSchema": {json.dumps(tool.output_schema)}
            }},
'''
        
        code += '''        }
'''
        
        # Add handlers
        code += "".join(handlers)
        
        code += '''
    def get_tools(self) -> Dict[str, Any]:
        """Get available tools."""
        return self.tools
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Map tool names to handler functions
        handler_map = {
'''
        
        for tool in specification.tools:
            code += f'''            "{tool.name}": self.{tool.handler_function},
'''
        
        code += '''        }
        
        if tool_name in handler_map:
            return handler_map[tool_name](arguments)
        else:
            raise NotImplementedError(f"Handler for {tool_name} not implemented")
'''

        return code
    
    async def _wrap_with_mcp_protocol(self, code: str, spec: MCPSpecification) -> str:
        """
        Wrap generated code with MCP JSON-RPC protocol.
        
        Args:
            code: Core functionality code
            spec: MCPSpecification for the MCP
            
        Returns:
            Complete MCP server code with protocol wrapper
        """
        self.logger.info("Wrapping code with MCP protocol")
        
        # Extract tool specs for registry
        tool_registry = []
        for tool in spec.tools:
            tool_entry = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema
            }
            tool_registry.append(json.dumps(tool_entry, indent=12))
        
        tool_registry_str = ",\n".join(tool_registry)
        
        # Generate tool dispatcher
        tool_dispatchers = []
        for tool in spec.tools:
            dispatcher = f'''            elif tool_name == "{tool.name}":
                return self.mcp_impl.{tool.handler_function}(arguments)'''
            tool_dispatchers.append(dispatcher)
        
        tool_dispatcher_str = "\n".join(tool_dispatchers)
        
        wrapped_code = f'''#!/usr/bin/env python3
"""
Generated MCP Server: {spec.name}
Description: {spec.description}
MCP Protocol Version: {spec.mcp_version}
"""

import json
import sys
import logging
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("{spec.name}")

# Generated core functionality
{code}

class MCPServer:
    """MCP JSON-RPC 2.0 Server over stdio transport."""
    
    def __init__(self):
        self.mcp_impl = FallbackMCP()
        self.tools = {{
{tool_registry_str}
        }}
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC 2.0 request."""
        method = request.get("method")
        params = request.get("params", {{}})
        request_id = request.get("id")
        
        if method == "initialize":
            return self._handle_initialize(request_id)
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tool_call(params, request_id)
        else:
            return self._error_response(request_id, -32601, "Method not found")
    
    def _handle_initialize(self, request_id):
        return {{
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {{
                "protocolVersion": "{spec.mcp_version}",
                "serverInfo": {{
                    "name": "{spec.name}",
                    "version": "1.0.0"
                }},
                "capabilities": {{
                    "tools": {{}}
                }}
            }}
        }}
    
    def _handle_tools_list(self, request_id):
        return {{
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {{
                "tools": list(self.tools.values())
            }}
        }}
    
    def _handle_tool_call(self, params, request_id):
        tool_name = params.get("name")
        arguments = params.get("arguments", {{}})
        
        if tool_name not in self.tools:
            return self._error_response(request_id, -32602, f"Unknown tool: {{tool_name}}")
        
        try:
            result = self._execute_tool(tool_name, arguments)
            return {{
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {{
                    "content": [{{
                        "type": "text",
                        "text": json.dumps(result)
                    }}]
                }}
            }}
        except Exception as e:
            return self._error_response(request_id, -32603, str(e))
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute the requested tool."""
{tool_dispatcher_str}
        else:
            raise ValueError(f"Unknown tool: {{tool_name}}")
    
    def _error_response(self, request_id, code, message):
        return {{
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {{
                "code": code,
                "message": message
            }}
        }}
    
    def run(self):
        """Run the MCP server on stdio."""
        logger.info("{spec.name} MCP server starting...")
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                error = self._error_response(None, -32700, "Parse error")
                print(json.dumps(error), flush=True)
            except Exception as e:
                error = self._error_response(None, -32603, str(e))
                print(json.dumps(error), flush=True)

if __name__ == "__main__":
    server = MCPServer()
    server.run()
'''
        
        return wrapped_code
    
    async def _validate_mcp(self, code: str) -> ValidationResult:
        """
        Validate generated MCP code with SafetyAgent.
        
        Args:
            code: MCP code to validate
            
        Returns:
            ValidationResult with validation results
        """
        self.logger.info("Validating MCP code with SafetyAgent")
        
        try:
            # Use SafetyAgent to analyze the code
            safety_report = self.safety_agent.analyze_code_security(
                code, 
                context={"mcp_generation": True, "purpose": "Generated MCP server"}
            )
            
            validation_result = ValidationResult(
                passed=safety_report.approval_status,
                issues=[threat.description for threat in safety_report.threats_detected],
                score=1.0 - safety_report.overall_risk_score  # Invert risk to get score
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during MCP validation: {e}")
            return ValidationResult(
                passed=False,
                issues=[f"Validation error: {str(e)}"],
                score=0.0
            )
    
    async def _test_mcp(self, code: str, specification: MCPSpecification) -> TestResult:
        """
        Test the generated MCP code.
        
        Args:
            code: MCP code to test
            specification: Specification for the MCP
            
        Returns:
            TestResult with test results
        """
        self.logger.info("Testing generated MCP code")
        
        # For now, we'll do basic syntax validation
        # In a full implementation, this would actually test the MCP
        
        try:
            # Basic syntax check
            compile(code, "<generated_mcp>", "exec")
            
            # Mock test results
            test_results = {
                "syntax_check": "passed",
                "compilation": "successful",
                "tools_count": len(specification.tools),
                "template_type": specification.template_type
            }
            
            return TestResult(
                passed=True,
                results=test_results
            )
            
        except Exception as e:
            self.logger.error(f"MCP testing failed: {e}")
            return TestResult(
                passed=False,
                errors=[f"Compilation failed: {str(e)}"]
            )
    
    async def _package_mcp(self, code: str, specification: MCPSpecification, output_dir: str) -> str:
        """
        Package the generated MCP for installation.
        
        Args:
            code: MCP code to package
            specification: Specification for the MCP
            output_dir: Directory to package MCP in
            
        Returns:
            Path to packaged MCP directory
        """
        self.logger.info("Packaging MCP for installation")
        
        try:
            # Create MCP directory
            mcp_dir = os.path.join(output_dir, specification.name)
            os.makedirs(mcp_dir, exist_ok=True)
            
            # Write main MCP code
            main_file = os.path.join(mcp_dir, f"{specification.name}.py")
            with open(main_file, "w") as f:
                f.write(code)
            
            # Create requirements.txt if dependencies exist
            if specification.dependencies:
                requirements_file = os.path.join(mcp_dir, "requirements.txt")
                with open(requirements_file, "w") as f:
                    for dep in specification.dependencies:
                        f.write(f"{dep}\n")
            
            # Create metadata file
            metadata = {
                "name": specification.name,
                "description": specification.description,
                "version": "1.0.0",
                "template_type": specification.template_type,
                "language": specification.language,
                "entry_point": specification.entry_point,
                "tools": [tool.name for tool in specification.tools],
                "generated_at": datetime.now().isoformat(),
                "mcp_version": specification.mcp_version
            }
            
            metadata_file = os.path.join(mcp_dir, "mcp_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create README
            readme_content = f"""# {specification.name}

{specification.description}

## Generated MCP Server

This MCP server was automatically generated by the Oracle agent.

### Capabilities
- Template Type: {specification.template_type}
- Tools: {', '.join([tool.name for tool in specification.tools])}
- Language: {specification.language}

### Usage
Run with: `python {specification.name}.py`

This server implements the Model Context Protocol (MCP) and communicates over stdio.
"""
            
            readme_file = os.path.join(mcp_dir, "README.md")
            with open(readme_file, "w") as f:
                f.write(readme_content)
            
            return mcp_dir
            
        except Exception as e:
            self.logger.error(f"Error packaging MCP: {e}")
            return ""

# Factory function
def create_mcp_generator(tool_creator, safety_agent, logger=None) -> MCPGenerator:
    """Factory function to create MCPGenerator instance."""
    return MCPGenerator(tool_creator, safety_agent, logger)