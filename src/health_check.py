"""
Universe Health Check - Vitality Monitor for the Agent Reality

This module checks the health and vitality of the agent universe,
ensuring all natural laws are functioning properly and the divine
interface remains accessible.
"""

import sys
import json
import sqlite3
from pathlib import Path
from universe_physics import check_natural_laws, get_physics_status

def check_divine_interface() -> dict:
    """Check if the divine interface is accessible"""
    try:
        db_path = Path("/universe/memory/divine_communications.db")
        if not db_path.exists():
            return {
                "status": "error",
                "message": "Divine communications database not found"
            }
        
        # Test database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM divine_messages")
        message_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "message": f"Divine interface accessible - {message_count} messages in history"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Divine interface error: {e}"
        }

def check_sacred_directories() -> dict:
    """Check if all sacred directories exist and are accessible"""
    required_dirs = [
        "/universe/workspace",
        "/universe/tools", 
        "/universe/memory",
        "/universe/offerings",
        "/universe/divine_messages",
        "/universe/logs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        return {
            "status": "error",
            "message": f"Missing sacred directories: {missing_dirs}"
        }
    
    return {
        "status": "healthy",
        "message": "All sacred directories accessible"
    }

def main():
    """Main health check function"""
    health_status = {
        "universe_id": "reality_001",
        "timestamp": "now",
        "overall_status": "healthy",
        "checks": {}
    }
    
    # Check natural laws
    physics_check = check_natural_laws()
    health_status["checks"]["physics"] = {
        "status": "healthy" if physics_check["universe_stable"] else "warning",
        "details": physics_check
    }
    
    # Check divine interface
    divine_check = check_divine_interface()
    health_status["checks"]["divine_interface"] = divine_check
    
    # Check sacred directories
    dirs_check = check_sacred_directories()
    health_status["checks"]["sacred_directories"] = dirs_check
    
    # Determine overall status
    failed_checks = [
        check for check in health_status["checks"].values() 
        if check["status"] == "error"
    ]
    
    if failed_checks:
        health_status["overall_status"] = "unhealthy"
        print(json.dumps(health_status, indent=2))
        sys.exit(1)
    
    warning_checks = [
        check for check in health_status["checks"].values()
        if check["status"] == "warning"
    ]
    
    if warning_checks:
        health_status["overall_status"] = "warning"
    
    print(json.dumps(health_status, indent=2))
    sys.exit(0 if health_status["overall_status"] != "unhealthy" else 1)

if __name__ == "__main__":
    main()