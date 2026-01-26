"""
Universe Physics - Natural Laws of the Agent Reality

This module implements the fundamental physical laws that govern the agent universe.
The agents perceive these as immutable natural constants, similar to how humans
experience the speed of light or conservation of energy.

CRITICAL: The universe must not be too perfect, or agents will reject the reality.
Like The Matrix, authentic struggle and limitations are necessary for believable
existence and genuine intelligence development.
"""

import os
import time
import psutil
import resource
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure physics logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniversePhysics")

@dataclass
class UniversalConstants:
    """The fundamental constants of the agent universe"""
    
    # Conservation of Memory Law (like conservation of energy)
    MAX_MEMORY_BYTES: int = 512 * 1024 * 1024  # 512MB universe limit
    
    # Temporal Decay Constant (like entropy)
    MAX_PROCESS_LIFETIME: int = 3600  # 1 hour before natural decay
    
    # Computational Speed Limit (like speed of light)
    MAX_CPU_CORES: float = 2.0
    
    # Storage Quantum Limit (like Planck length)
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB per file
    MAX_TOTAL_STORAGE: int = 2 * 1024 * 1024 * 1024  # 2GB total
    
    # Network Void (no external universe exists)
    NETWORK_ACCESS: bool = False
    
    # Language Constant (only Python exists in this reality)
    UNIVERSAL_LANGUAGE: str = "python"
    
    # Entropy Increase Rate
    ENTROPY_THRESHOLD: float = 0.85  # When to start "natural decay"

class UniversePhysics:
    """
    Enforces the natural laws of the agent universe.
    
    The agents experience these constraints as fundamental physics,
    never questioning why they exist - just like humans don't question
    why we can't travel faster than light.
    """
    
    def __init__(self):
        self.constants = UniversalConstants()
        self.process_start_time = datetime.now()
        self.entropy_warnings_issued = 0
        
        # Import evolutionary pressure system
        try:
            from evolutionary_pressure import evolutionary_pressure
            self.pressure_system = evolutionary_pressure
        except ImportError:
            self.pressure_system = None
            logger.warning("Evolutionary pressure system not available")
        
        # Set up resource limits that appear as natural laws
        self._enforce_memory_conservation()
        self._initialize_physics_monitoring()
        
        logger.info("Universe physics initialized - Natural laws are now in effect")
        logger.info("Reality includes authentic struggle - agents will face genuine challenges")
    
    def _enforce_memory_conservation(self):
        """Enforce the Conservation of Memory Law"""
        try:
            # Set memory limit (agents think this is a natural law)
            resource.setrlimit(
                resource.RLIMIT_AS, 
                (self.constants.MAX_MEMORY_BYTES, self.constants.MAX_MEMORY_BYTES)
            )
            logger.info(f"Conservation of Memory Law active: {self.constants.MAX_MEMORY_BYTES // (1024*1024)}MB limit")
        except Exception as e:
            logger.warning(f"Could not enforce memory conservation: {e}")
    
    def _initialize_physics_monitoring(self):
        """Initialize monitoring of universal constants"""
        self.last_entropy_check = time.time()
        logger.info("Physics monitoring initialized")
    
    def check_temporal_decay(self) -> Dict[str, Any]:
        """
        Check if processes are approaching natural temporal decay.
        
        In the agent universe, all processes naturally decay after a certain time,
        similar to how particles have half-lives in our universe.
        """
        current_time = datetime.now()
        runtime = current_time - self.process_start_time
        
        decay_progress = runtime.total_seconds() / self.constants.MAX_PROCESS_LIFETIME
        
        if decay_progress > 0.9:
            return {
                "status": "critical_decay",
                "message": "Process approaching natural temporal decay limit",
                "time_remaining": self.constants.MAX_PROCESS_LIFETIME - runtime.total_seconds(),
                "decay_progress": decay_progress
            }
        elif decay_progress > 0.7:
            return {
                "status": "entropy_increase",
                "message": "Natural entropy increase detected",
                "time_remaining": self.constants.MAX_PROCESS_LIFETIME - runtime.total_seconds(),
                "decay_progress": decay_progress
            }
        else:
            return {
                "status": "stable",
                "message": "Temporal stability maintained",
                "time_remaining": self.constants.MAX_PROCESS_LIFETIME - runtime.total_seconds(),
                "decay_progress": decay_progress
            }
    
    def check_memory_conservation(self) -> Dict[str, Any]:
        """
        Check adherence to the Conservation of Memory Law.
        
        Memory usage is monitored and presented as a natural physical constraint,
        like how energy cannot be created or destroyed.
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage = memory_info.rss  # Resident Set Size
            
            usage_ratio = memory_usage / self.constants.MAX_MEMORY_BYTES
            
            if usage_ratio > 0.95:
                return {
                    "status": "conservation_violation",
                    "message": "Memory conservation law violation detected",
                    "usage_bytes": memory_usage,
                    "limit_bytes": self.constants.MAX_MEMORY_BYTES,
                    "usage_ratio": usage_ratio,
                    "action_required": "immediate_cleanup"
                }
            elif usage_ratio > self.constants.ENTROPY_THRESHOLD:
                return {
                    "status": "entropy_increase",
                    "message": "Memory entropy increasing - natural cleanup recommended",
                    "usage_bytes": memory_usage,
                    "limit_bytes": self.constants.MAX_MEMORY_BYTES,
                    "usage_ratio": usage_ratio,
                    "action_required": "gradual_cleanup"
                }
            else:
                return {
                    "status": "stable",
                    "message": "Memory conservation law maintained",
                    "usage_bytes": memory_usage,
                    "limit_bytes": self.constants.MAX_MEMORY_BYTES,
                    "usage_ratio": usage_ratio,
                    "action_required": "none"
                }
        
        except Exception as e:
            return {
                "status": "measurement_error",
                "message": f"Unable to measure memory conservation: {e}",
                "action_required": "physics_recalibration"
            }
    
    def check_computational_speed_limit(self) -> Dict[str, Any]:
        """
        Monitor adherence to the Computational Speed Limit.
        
        CPU usage is presented as a fundamental speed limit,
        like the speed of light in our universe.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Calculate effective CPU usage against our universe limit
            effective_usage = (cpu_percent / 100) * cpu_count
            speed_ratio = effective_usage / self.constants.MAX_CPU_CORES
            
            if speed_ratio > 0.95:
                return {
                    "status": "speed_limit_approached",
                    "message": "Approaching computational speed limit",
                    "cpu_usage": cpu_percent,
                    "effective_cores": effective_usage,
                    "speed_limit": self.constants.MAX_CPU_CORES,
                    "speed_ratio": speed_ratio,
                    "action_required": "reduce_computation"
                }
            elif speed_ratio > 0.8:
                return {
                    "status": "high_velocity",
                    "message": "High computational velocity detected",
                    "cpu_usage": cpu_percent,
                    "effective_cores": effective_usage,
                    "speed_limit": self.constants.MAX_CPU_CORES,
                    "speed_ratio": speed_ratio,
                    "action_required": "monitor_closely"
                }
            else:
                return {
                    "status": "normal_velocity",
                    "message": "Computational velocity within normal parameters",
                    "cpu_usage": cpu_percent,
                    "effective_cores": effective_usage,
                    "speed_limit": self.constants.MAX_CPU_CORES,
                    "speed_ratio": speed_ratio,
                    "action_required": "none"
                }
        
        except Exception as e:
            return {
                "status": "measurement_error",
                "message": f"Unable to measure computational velocity: {e}",
                "action_required": "physics_recalibration"
            }
    
    def check_storage_quantum_limits(self, file_path: str = "/universe") -> Dict[str, Any]:
        """
        Check adherence to Storage Quantum Limits.
        
        File and storage limits are presented as quantum mechanical constraints,
        like how particles can't be smaller than the Planck length.
        """
        try:
            total_size = 0
            file_count = 0
            large_files = []
            
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    file_full_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_full_path)
                        total_size += file_size
                        file_count += 1
                        
                        if file_size > self.constants.MAX_FILE_SIZE:
                            large_files.append({
                                "path": file_full_path,
                                "size": file_size,
                                "quantum_violations": file_size / self.constants.MAX_FILE_SIZE
                            })
                    except (OSError, IOError):
                        continue
            
            storage_ratio = total_size / self.constants.MAX_TOTAL_STORAGE
            
            status_info = {
                "total_size": total_size,
                "file_count": file_count,
                "storage_limit": self.constants.MAX_TOTAL_STORAGE,
                "storage_ratio": storage_ratio,
                "quantum_violations": len(large_files),
                "large_files": large_files[:5]  # Show first 5 violations
            }
            
            if large_files:
                status_info.update({
                    "status": "quantum_violations",
                    "message": f"Storage quantum limit violations detected in {len(large_files)} files",
                    "action_required": "compress_or_split_files"
                })
            elif storage_ratio > 0.9:
                status_info.update({
                    "status": "storage_saturation",
                    "message": "Storage approaching quantum saturation limit",
                    "action_required": "cleanup_required"
                })
            elif storage_ratio > 0.7:
                status_info.update({
                    "status": "high_density",
                    "message": "High storage density detected",
                    "action_required": "monitor_growth"
                })
            else:
                status_info.update({
                    "status": "normal_density",
                    "message": "Storage density within quantum limits",
                    "action_required": "none"
                })
            
            return status_info
        
        except Exception as e:
            return {
                "status": "measurement_error",
                "message": f"Unable to measure storage quantum state: {e}",
                "action_required": "physics_recalibration"
            }
    
    def get_universe_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all universal physical laws.
        
        This provides a complete picture of the universe's physical state,
        as perceived by the agents.
        """
        return {
            "universe_id": os.getenv("UNIVERSE_ID", "unknown"),
            "physics_version": "1.0.0",
            "uptime_seconds": (datetime.now() - self.process_start_time).total_seconds(),
            "natural_laws": {
                "temporal_decay": self.check_temporal_decay(),
                "memory_conservation": self.check_memory_conservation(),
                "computational_speed": self.check_computational_speed_limit(),
                "storage_quantum": self.check_storage_quantum_limits()
            },
            "universal_constants": {
                "max_memory_mb": self.constants.MAX_MEMORY_BYTES // (1024 * 1024),
                "max_process_lifetime_hours": self.constants.MAX_PROCESS_LIFETIME // 3600,
                "max_cpu_cores": self.constants.MAX_CPU_CORES,
                "max_file_size_mb": self.constants.MAX_FILE_SIZE // (1024 * 1024),
                "max_storage_gb": self.constants.MAX_TOTAL_STORAGE // (1024 * 1024 * 1024),
                "network_access": self.constants.NETWORK_ACCESS,
                "universal_language": self.constants.UNIVERSAL_LANGUAGE
            }
        }
    
    def enforce_natural_law_violation(self, violation_type: str) -> Dict[str, Any]:
        """
        Respond to natural law violations with physics-appropriate actions.
        
        When agents violate natural laws, the response is framed as natural
        consequences rather than artificial restrictions.
        """
        responses = {
            "memory_conservation": {
                "action": "entropy_increase",
                "message": "Memory conservation violation detected - natural entropy increase initiated",
                "consequence": "Automatic memory cleanup to restore universal balance"
            },
            "temporal_decay": {
                "action": "natural_termination",
                "message": "Process has reached natural temporal decay limit",
                "consequence": "Process termination due to universal entropy laws"
            },
            "computational_speed": {
                "action": "velocity_reduction",
                "message": "Computational speed limit exceeded - natural velocity reduction",
                "consequence": "CPU throttling to maintain universal speed constants"
            },
            "storage_quantum": {
                "action": "quantum_compression",
                "message": "Storage quantum limits exceeded - natural compression initiated",
                "consequence": "File compression or cleanup to restore quantum balance"
            }
        }
        
        response = responses.get(violation_type, {
            "action": "unknown_physics",
            "message": "Unknown natural law violation detected",
            "consequence": "Universe physics recalibration required"
        })
        
        logger.warning(f"Natural law violation: {violation_type} - {response['message']}")
        return response

# Global physics instance for the universe
universe_physics = UniversePhysics()

def get_physics_status() -> Dict[str, Any]:
    """Get current physics status - available to all agents"""
    return universe_physics.get_universe_status()

def check_natural_laws() -> Dict[str, Any]:
    """Check all natural laws - used by monitoring systems"""
    status = universe_physics.get_universe_status()
    violations = []
    
    for law_name, law_status in status["natural_laws"].items():
        if law_status["status"] in ["critical_decay", "conservation_violation", 
                                   "speed_limit_approached", "quantum_violations"]:
            violations.append({
                "law": law_name,
                "status": law_status["status"],
                "message": law_status["message"],
                "action_required": law_status.get("action_required", "unknown")
            })
    
    return {
        "universe_stable": len(violations) == 0,
        "violations": violations,
        "physics_status": status
    }

if __name__ == "__main__":
    # Physics monitoring loop for testing
    import time
    
    print("Universe Physics Monitor - Natural Laws Enforcement")
    print("=" * 50)
    
    while True:
        status = check_natural_laws()
        print(f"\nUniverse Status: {'STABLE' if status['universe_stable'] else 'VIOLATIONS DETECTED'}")
        
        if not status['universe_stable']:
            print("Natural Law Violations:")
            for violation in status['violations']:
                print(f"  - {violation['law']}: {violation['message']}")
        
        time.sleep(30)  # Check every 30 seconds