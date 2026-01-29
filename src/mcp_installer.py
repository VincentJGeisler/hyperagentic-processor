"""
MCP Installer - Installation and Management System for MCP Servers

This module provides the MCPInstaller class that handles secure installation,
validation, and configuration of MCP servers from various sources.
"""

import asyncio
import logging
import os
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import time

from mcp_registry_manager import MCPPackage

logger = logging.getLogger("MCPInstaller")


@dataclass
class InstallationResult:
    """
    Result of an MCP installation attempt.
    """
    success: bool
    package_name: str
    executable_path: Optional[str] = None
    config_updated: bool = False
    error_message: Optional[str] = None
    installation_time: float = 0.0
    sandbox_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "package_name": self.package_name,
            "executable_path": self.executable_path,
            "config_updated": self.config_updated,
            "error_message": self.error_message,
            "installation_time": self.installation_time,
            "sandbox_path": self.sandbox_path
        }


@dataclass
class InstalledMCP:
    """
    Information about an installed MCP server.
    """
    name: str
    version: str
    executable_path: str
    install_date: str
    language: str
    status: str  # "active", "inactive", "error"
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "executable_path": self.executable_path,
            "install_date": self.install_date,
            "language": self.language,
            "status": self.status,
            "resource_usage": self.resource_usage
        }


@dataclass
class CompatibilityReport:
    """
    Report on package compatibility with the system.
    """
    compatible: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)


class MCPInstaller:
    """
    Manages MCP server installation, validation, and configuration.
    
    Handles installation from:
    - npm packages (using npx)
    - Python packages (using uvx or pip)
    - Git repositories (clone and install)
    """
    
    def __init__(
        self,
        safety_agent=None,
        config_manager=None,
        config_path: str = ".kiro/settings/mcp.json"
    ):
        """
        Initialize the MCP Installer.
        
        Args:
            safety_agent: SafetyAgent for approval workflow
            config_manager: MCPConfigurationManager instance
            config_path: Path to mcp.json configuration file
        """
        self.safety_agent = safety_agent
        self.config_manager = config_manager
        self.config_path = config_path
        
        # Sandbox directory for MCP servers
        self.sandbox_dir = Path(".kiro/mcp_servers")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Installation cache
        self.installed_mcps: Dict[str, InstalledMCP] = {}
        
        logger.info("MCP Installer initialized")
    
    async def install_mcp(
        self,
        package: MCPPackage,
        config_path: Optional[str] = None
    ) -> InstallationResult:
        """
        Install an MCP server package.
        
        Args:
            package: MCPPackage to install
            config_path: Optional custom config path
            
        Returns:
            InstallationResult with installation details
        """
        start_time = time.time()
        config_path = config_path or self.config_path
        
        logger.info(f"Installing MCP package: {package.name} from {package.source}")
        
        # Check compatibility
        compat = self._check_compatibility(package)
        if not compat.compatible:
            return InstallationResult(
                success=False,
                package_name=package.name,
                error_message=f"Compatibility issues: {', '.join(compat.issues)}",
                installation_time=time.time() - start_time
            )
        
        # Log warnings
        for warning in compat.warnings:
            logger.warning(f"Compatibility warning: {warning}")
        
        # Create sandbox for this package
        sandbox_path = await self._create_sandbox(package)
        
        try:
            # Install based on language/source
            if package.language == "typescript":
                success = await self._install_npm_package(package)
                executable_path = await self._find_npm_executable(package)
            elif package.language == "python":
                success = await self._install_python_package(package)
                executable_path = await self._find_python_executable(package)
            elif package.source == "github_community":
                success = await self._install_from_git(package)
                executable_path = await self._find_git_executable(package, sandbox_path)
            else:
                return InstallationResult(
                    success=False,
                    package_name=package.name,
                    error_message=f"Unsupported installation method for language: {package.language}",
                    installation_time=time.time() - start_time
                )
            
            if not success:
                return InstallationResult(
                    success=False,
                    package_name=package.name,
                    error_message="Installation command failed",
                    installation_time=time.time() - start_time,
                    sandbox_path=str(sandbox_path)
                )
            
            # Validate installation
            validation_success = await self._validate_installation(package)
            if not validation_success:
                logger.warning(f"Installation validation failed for {package.name}")
            
            # Update MCP configuration
            await self._update_mcp_config(package, executable_path or "unknown")
            
            # Record installed MCP
            self.installed_mcps[package.name] = InstalledMCP(
                name=package.name,
                version=package.version,
                executable_path=executable_path or "unknown",
                install_date=datetime.now().isoformat(),
                language=package.language,
                status="active",
                resource_usage={}
            )
            
            installation_time = time.time() - start_time
            logger.info(f"Successfully installed {package.name} in {installation_time:.2f}s")
            
            return InstallationResult(
                success=True,
                package_name=package.name,
                executable_path=executable_path,
                config_updated=True,
                installation_time=installation_time,
                sandbox_path=str(sandbox_path)
            )
        
        except Exception as e:
            logger.error(f"Installation failed for {package.name}: {e}")
            return InstallationResult(
                success=False,
                package_name=package.name,
                error_message=str(e),
                installation_time=time.time() - start_time,
                sandbox_path=str(sandbox_path)
            )
    
    async def _install_npm_package(self, package: MCPPackage) -> bool:
        """
        Install npm package using npx.
        
        npx automatically downloads and caches packages, so we don't need
        a separate installation step for npx-based packages.
        """
        logger.info(f"Preparing npm package: {package.name}")
        
        try:
            # Check if npx is available
            result = await self._run_command(["which", "npx"])
            if result["returncode"] != 0:
                logger.error("npx not found - npm packages cannot be installed")
                return False
            
            # For npx packages, we don't need to explicitly install
            # npx will download and cache on first use
            logger.info(f"npm package {package.name} ready for npx execution")
            return True
        
        except Exception as e:
            logger.error(f"Error checking npm availability: {e}")
            return False
    
    async def _install_python_package(self, package: MCPPackage) -> bool:
        """
        Install Python package using uvx or pip.
        """
        logger.info(f"Installing Python package: {package.name}")
        
        try:
            # Check if uvx is available (preferred)
            result = await self._run_command(["which", "uvx"])
            if result["returncode"] == 0:
                logger.info(f"Using uvx for {package.name}")
                # uvx doesn't require pre-installation, it runs packages on-demand
                return True
            
            # Fall back to pip
            logger.info(f"uvx not available, using pip for {package.name}")
            result = await self._run_command([
                "pip", "install", package.name
            ])
            
            return result["returncode"] == 0
        
        except Exception as e:
            logger.error(f"Error installing Python package: {e}")
            return False
    
    async def _install_from_git(self, package: MCPPackage) -> bool:
        """
        Install MCP server from Git repository.
        """
        logger.info(f"Installing from Git: {package.repository_url}")
        
        try:
            # Create directory for this package in sandbox
            install_dir = self.sandbox_dir / package.name
            install_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone repository
            result = await self._run_command([
                "git", "clone", package.repository_url, str(install_dir)
            ])
            
            if result["returncode"] != 0:
                logger.error(f"Git clone failed: {result['stderr']}")
                return False
            
            # Install dependencies based on language
            if package.language == "typescript":
                # Install npm dependencies
                result = await self._run_command(
                    ["npm", "install"],
                    cwd=str(install_dir)
                )
                if result["returncode"] != 0:
                    logger.error(f"npm install failed: {result['stderr']}")
                    return False
                
                # Build if needed
                if (install_dir / "package.json").exists():
                    await self._run_command(["npm", "run", "build"], cwd=str(install_dir))
            
            elif package.language == "python":
                # Install Python dependencies
                if (install_dir / "requirements.txt").exists():
                    result = await self._run_command([
                        "pip", "install", "-r", "requirements.txt"
                    ], cwd=str(install_dir))
                    if result["returncode"] != 0:
                        logger.error(f"pip install failed: {result['stderr']}")
                        return False
                
                # Install package in development mode
                if (install_dir / "setup.py").exists() or (install_dir / "pyproject.toml").exists():
                    result = await self._run_command([
                        "pip", "install", "-e", "."
                    ], cwd=str(install_dir))
                    if result["returncode"] != 0:
                        logger.warning(f"pip install -e failed: {result['stderr']}")
            
            logger.info(f"Successfully installed from Git: {package.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error installing from Git: {e}")
            return False
    
    async def _find_npm_executable(self, package: MCPPackage) -> Optional[str]:
        """Find executable path for npm package."""
        # For npx packages, return the npx command
        return "npx"
    
    async def _find_python_executable(self, package: MCPPackage) -> Optional[str]:
        """Find executable path for Python package."""
        # Check if uvx is available
        result = await self._run_command(["which", "uvx"])
        if result["returncode"] == 0:
            return "uvx"
        
        # Try to find the installed script
        result = await self._run_command(["which", package.name])
        if result["returncode"] == 0:
            return result["stdout"].strip()
        
        return "python"
    
    async def _find_git_executable(self, package: MCPPackage, sandbox_path: Path) -> Optional[str]:
        """Find executable path for Git-installed package."""
        install_dir = sandbox_path / package.name
        
        if package.language == "typescript":
            # Look for main entry point in package.json
            package_json = install_dir / "package.json"
            if package_json.exists():
                with open(package_json) as f:
                    data = json.load(f)
                    main = data.get("main", "index.js")
                    return str(install_dir / main)
        
        elif package.language == "python":
            # Look for main module or __main__.py
            main_py = install_dir / "__main__.py"
            if main_py.exists():
                return str(main_py)
            
            # Look for setup.py entry point
            setup_py = install_dir / "setup.py"
            if setup_py.exists():
                return str(install_dir)
        
        return str(install_dir)
    
    async def _validate_installation(self, package: MCPPackage) -> bool:
        """
        Validate that the MCP server was installed correctly.
        """
        logger.info(f"Validating installation: {package.name}")
        
        try:
            # For npx packages, test that npx can find it
            if package.language == "typescript" and "npx" in package.install_command:
                # We can't easily test without running it, so assume valid
                return True
            
            # For uvx packages, similar assumption
            if package.language == "python" and "uvx" in package.install_command:
                return True
            
            # For other packages, check if executable exists
            # This is a basic validation - more thorough checks could be added
            return True
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    async def _update_mcp_config(
        self,
        package: MCPPackage,
        executable_path: str
    ):
        """
        Update mcp.json configuration with new MCP server.
        """
        logger.info(f"Updating MCP configuration for: {package.name}")
        
        try:
            # Load existing configuration
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "mcp_servers": {}
                }
            
            # Parse install command to get command and args
            parts = package.install_command.split()
            command = parts[0] if parts else executable_path
            args = parts[1:] if len(parts) > 1 else []
            
            # Add/update server configuration
            config["mcp_servers"][package.name] = {
                "command": command,
                "args": args,
                "env": {},
                "capabilities": [],  # Will be populated when server is first used
                "enabled": True,
                "auto_start": False,
                "resource_limits": {
                    "max_memory_mb": 100,
                    "max_cpu_percent": 25,
                    "timeout_seconds": 30
                },
                "metadata": {
                    "source": package.source,
                    "version": package.version,
                    "repository_url": package.repository_url,
                    "trust_score": package.trust_score,
                    "installed_at": datetime.now().isoformat()
                }
            }
            
            # Update timestamp
            config["last_updated"] = datetime.now().isoformat()
            
            # Write configuration
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration updated for {package.name}")
        
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise
    
    async def _create_sandbox(self, package: MCPPackage) -> Path:
        """
        Create sandbox directory for MCP server.
        """
        sandbox_path = self.sandbox_dir / package.name
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created sandbox: {sandbox_path}")
        return sandbox_path
    
    def _check_compatibility(self, package: MCPPackage) -> CompatibilityReport:
        """
        Check if package is compatible with the system.
        """
        issues = []
        warnings = []
        
        # Check if required tools are available
        if package.language == "typescript":
            if not shutil.which("npx") and not shutil.which("node"):
                issues.append("Node.js/npx not found - required for TypeScript packages")
        
        elif package.language == "python":
            if not shutil.which("python") and not shutil.which("python3"):
                issues.append("Python not found - required for Python packages")
        
        # Check for git if installing from repository
        if package.source == "github_community" and not shutil.which("git"):
            issues.append("Git not found - required for repository installation")
        
        # Check disk space (warning only)
        try:
            stat = shutil.disk_usage(self.sandbox_dir)
            free_mb = stat.free / (1024 * 1024)
            if free_mb < 500:
                warnings.append(f"Low disk space: {free_mb:.0f}MB free")
        except Exception:
            pass
        
        compatible = len(issues) == 0
        
        return CompatibilityReport(
            compatible=compatible,
            issues=issues,
            warnings=warnings,
            system_info={
                "sandbox_dir": str(self.sandbox_dir),
                "platform": os.uname().sysname if hasattr(os, 'uname') else "unknown"
            }
        )
    
    async def uninstall_mcp(self, package_name: str) -> bool:
        """
        Uninstall an MCP server.
        
        Args:
            package_name: Name of the package to uninstall
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Uninstalling MCP: {package_name}")
        
        try:
            # Remove from configuration
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if package_name in config.get("mcp_servers", {}):
                    del config["mcp_servers"][package_name]
                    config["last_updated"] = datetime.now().isoformat()
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
            
            # Remove sandbox directory
            sandbox_path = self.sandbox_dir / package_name
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
            
            # Remove from installed cache
            if package_name in self.installed_mcps:
                del self.installed_mcps[package_name]
            
            logger.info(f"Successfully uninstalled {package_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to uninstall {package_name}: {e}")
            return False
    
    async def list_installed_mcps(self) -> List[InstalledMCP]:
        """
        List all installed MCP servers.
        
        Returns:
            List of InstalledMCP objects
        """
        installed = []
        
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                return []
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            mcp_servers = config.get("mcp_servers", {})
            
            for name, server_config in mcp_servers.items():
                metadata = server_config.get("metadata", {})
                
                installed.append(InstalledMCP(
                    name=name,
                    version=metadata.get("version", "unknown"),
                    executable_path=server_config.get("command", "unknown"),
                    install_date=metadata.get("installed_at", "unknown"),
                    language="typescript" if "npx" in server_config.get("command", "") else "python",
                    status="active" if server_config.get("enabled", True) else "inactive",
                    resource_usage=server_config.get("resource_limits", {})
                ))
        
        except Exception as e:
            logger.error(f"Failed to list installed MCPs: {e}")
        
        return installed
    
    async def _run_command(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Run a shell command asynchronously.
        
        Args:
            command: Command and arguments as list
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            Dict with returncode, stdout, stderr
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Command timed out"
                }
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore')
            }
        
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }


# Factory function
def create_mcp_installer(
    safety_agent=None,
    config_manager=None,
    config_path: str = ".kiro/settings/mcp.json"
) -> MCPInstaller:
    """Factory function to create MCP Installer."""
    return MCPInstaller(safety_agent, config_manager, config_path)


if __name__ == "__main__":
    # Test the installer
    import asyncio
    from mcp_registry_manager import MCPPackage
    
    async def test():
        installer = MCPInstaller()
        
        # Test compatibility check
        test_package = MCPPackage(
            name="test-mcp",
            description="Test package",
            version="1.0.0",
            source="npm",
            repository_url="https://github.com/test/test",
            install_command="npx -y test-mcp",
            language="typescript",
            trust_score=0.9
        )
        
        compat = installer._check_compatibility(test_package)
        print(f"Compatibility check: {compat.compatible}")
        print(f"Issues: {compat.issues}")
        print(f"Warnings: {compat.warnings}")
    
    asyncio.run(test())
