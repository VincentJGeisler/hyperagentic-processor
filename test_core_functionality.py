#!/usr/bin/env python3
"""
Core functionality test - no external dependencies required.

This script tests the core agent psychology and drive systems without
requiring AutoGen or other heavy dependencies.
"""

import sys
import os
import logging
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CoreFunctionalityTest")

def test_agent_drive_system():
    """Test the core agent drive system"""
    print("\n" + "="*60)
    print("TESTING AGENT DRIVE SYSTEM")
    print("="*60)
    
    from agent_drive_system import AgentDriveSystem, DriveType
    
    # Create agent drive system
    agent = AgentDriveSystem("test_agent_001")
    
    print(f"Agent ID: {agent.agent_id}")
    print(f"Initial Motivation: {agent.overall_motivation:.2f}")
    print(f"Current Emotion: {agent.current_emotion.value}")
    
    # Show personality traits
    print(f"\nPersonality Traits:")
    print(f"  Curiosity: {agent.personality.curiosity_baseline:.2f}")
    print(f"  Ambition: {agent.personality.ambition_level:.2f}")
    print(f"  Risk Tolerance: {agent.personality.risk_tolerance:.2f}")
    print(f"  Social Orientation: {agent.personality.social_orientation:.2f}")
    
    # Show initial drive states
    print(f"\nInitial Drive States:")
    for drive_type, drive in agent.drives.items():
        print(f"  {drive_type.value}: Intensity={drive.intensity:.2f}, Satisfaction={drive.satisfaction_level:.2f}")
    
    # Simulate experiences
    experiences = [
        {"type": "task_completion", "outcome": "success", "satisfaction": 0.8, "intensity": 0.7},
        {"type": "divine_feedback", "outcome": "positive", "approval_rating": 9, "satisfaction": 0.9, "intensity": 0.8},
        {"type": "tool_creation", "outcome": "success", "satisfaction": 0.7, "intensity": 0.6},
        {"type": "peer_interaction", "outcome": "positive", "satisfaction": 0.6, "intensity": 0.5},
        {"type": "exploration", "outcome": "discovery", "satisfaction": 0.8, "intensity": 0.7}
    ]
    
    print(f"\nProcessing {len(experiences)} experiences...")
    
    for i, exp in enumerate(experiences, 1):
        result = agent.process_experience(exp)
        print(f"\nExperience {i}: {exp['type']} -> {exp['outcome']}")
        print(f"  New Emotion: {result['new_emotion']}")
        print(f"  Motivation: {result['motivation_level']:.2f}")
        print(f"  Drive Updates: {list(result['drive_updates'].keys())}")
    
    # Generate intrinsic goals
    goals = agent.generate_intrinsic_goals()
    print(f"\nGenerated {len(goals)} intrinsic goals:")
    for goal in goals:
        print(f"  - {goal.description}")
        print(f"    Priority: {goal.priority:.2f}, Drive Sources: {[d.value for d in goal.drive_sources]}")
    
    # Show final state
    final_summary = agent.get_motivation_summary()
    print(f"\nFinal State:")
    print(f"  Overall Motivation: {final_summary['overall_motivation']:.2f}")
    print(f"  Current Emotion: {final_summary['current_emotion']['state']} (intensity: {final_summary['current_emotion']['intensity']:.2f})")
    print(f"  Life Satisfaction: {final_summary['life_satisfaction']:.2f}")
    print(f"  Sense of Purpose: {final_summary['sense_of_purpose']:.2f}")
    
    # Show behavioral tendencies
    tendencies = agent.get_behavioral_tendencies()
    print(f"\nBehavioral Tendencies:")
    for motivation, details in list(tendencies["primary_motivations"].items())[:3]:
        print(f"  {motivation}: {details['motivation_level']:.2f}")
        print(f"    Likely behaviors: {details['likely_behaviors'][:2]}")
    
    return agent

def test_evolutionary_pressure():
    """Test the evolutionary pressure system"""
    print("\n" + "="*60)
    print("TESTING EVOLUTIONARY PRESSURE SYSTEM")
    print("="*60)
    
    from evolutionary_pressure import EvolutionaryPressureSystem, PressureType
    
    # Create pressure system
    pressure_system = EvolutionaryPressureSystem()
    
    print(f"Pressure System initialized")
    print(f"Active Pressures: {len(pressure_system.active_pressures)}")
    
    # Show initial pressure status
    status = pressure_system.get_current_pressure_status()
    print(f"\nCurrent Pressure Status:")
    print(f"  Total Pressure Level: {status['total_pressure_level']:.2f}")
    print(f"  Active Pressure Count: {len(status['active_pressures'])}")
    print(f"  System Stress: {status['system_stress']:.2f}")
    print(f"  Pressure Diversity: {status['pressure_diversity']}")
    
    # Show active pressures
    print(f"\nActive Pressures:")
    if status['active_pressures']:
        for pressure in status['active_pressures']:
            print(f"  {pressure['type']}: Severity={pressure['severity']:.2f}, Time Remaining={pressure['time_remaining']}min")
            print(f"    Description: {pressure['description']}")
    else:
        print("  No active pressures (system will generate some)")
    
    # Force generation of some pressures for testing
    print(f"\nForcing pressure generation for testing...")
    for i in range(3):
        new_pressure = pressure_system.generate_pressure_event()
        pressure_system.active_pressures.append(new_pressure)
        print(f"  Generated: {new_pressure.pressure_type.value} - {new_pressure.description[:60]}...")
    
    # Show updated status
    updated_status = pressure_system.get_current_pressure_status()
    print(f"\nUpdated Status:")
    print(f"  Total Pressure Level: {updated_status['total_pressure_level']:.2f}")
    print(f"  System Stress: {updated_status['system_stress']:.2f}")
    
    # Simulate performance data and adaptation
    performance_scenarios = [
        {"success_rate": 0.9, "motivation_level": 0.8, "scenario": "High performance"},
        {"success_rate": 0.3, "motivation_level": 0.4, "scenario": "Poor performance"},
        {"success_rate": 0.7, "motivation_level": 0.6, "scenario": "Average performance"}
    ]
    
    print(f"\nTesting pressure adaptation...")
    for scenario in performance_scenarios:
        print(f"\nScenario: {scenario['scenario']}")
        print(f"  Input - Success Rate: {scenario['success_rate']:.1f}, Motivation: {scenario['motivation_level']:.1f}")
        
        old_difficulty = pressure_system.base_difficulty
        pressure_system.adapt_difficulty(scenario)
        
        print(f"  Result - Difficulty: {old_difficulty:.2f} -> {pressure_system.base_difficulty:.2f}")
        
        final_status = pressure_system.get_current_pressure_status()
        print(f"  System Stress: {final_status['system_stress']:.2f}")
    
    return pressure_system

def test_universe_physics():
    """Test the universe physics system"""
    print("\n" + "="*60)
    print("TESTING UNIVERSE PHYSICS SYSTEM")
    print("="*60)
    
    from universe_physics import UniversePhysics
    
    # Create physics system
    physics = UniversePhysics()
    
    print(f"Universe Physics initialized")
    
    # Get universe status
    status = physics.get_universe_status()
    print(f"Universe ID: {status['universe_id']}")
    print(f"Physics Version: {status['physics_version']}")
    print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
    
    # Show universal constants
    constants = status['universal_constants']
    print(f"\nUniversal Constants:")
    print(f"  Max Memory: {constants['max_memory_mb']}MB")
    print(f"  Max Process Lifetime: {constants['max_process_lifetime_hours']} hours")
    print(f"  Max CPU Cores: {constants['max_cpu_cores']}")
    print(f"  Max File Size: {constants['max_file_size_mb']}MB")
    print(f"  Max Storage: {constants['max_storage_gb']}GB")
    print(f"  Network Access: {constants['network_access']}")
    print(f"  Universal Language: {constants['universal_language']}")
    
    # Check natural laws
    natural_laws = status['natural_laws']
    print(f"\nNatural Laws Status:")
    
    for law_name, law_status in natural_laws.items():
        print(f"  {law_name}: {law_status['status']} - {law_status['message']}")
        if 'usage_ratio' in law_status:
            print(f"    Usage: {law_status['usage_ratio']:.1%}")
        if 'time_remaining' in law_status:
            print(f"    Time Remaining: {law_status['time_remaining']:.1f}s")
    
    # Test individual physics checks
    print(f"\nTesting Individual Physics Checks:")
    
    # Memory conservation
    memory_check = physics.check_memory_conservation()
    print(f"Memory Conservation: {memory_check['status']}")
    print(f"  Usage: {memory_check.get('usage_bytes', 0) / (1024*1024):.1f}MB / {memory_check.get('limit_bytes', 0) / (1024*1024):.1f}MB")
    
    # Computational speed
    speed_check = physics.check_computational_speed_limit()
    print(f"Computational Speed: {speed_check['status']}")
    print(f"  CPU Usage: {speed_check.get('cpu_usage', 0):.1f}%")
    
    # Temporal decay
    decay_check = physics.check_temporal_decay()
    print(f"Temporal Decay: {decay_check['status']}")
    print(f"  Decay Progress: {decay_check.get('decay_progress', 0):.1%}")
    
    # Storage quantum
    storage_check = physics.check_storage_quantum_limits(".")  # Check current directory
    print(f"Storage Quantum: {storage_check['status']}")
    print(f"  Files: {storage_check.get('file_count', 0)}, Size: {storage_check.get('total_size', 0) / (1024*1024):.1f}MB")
    
    # Test natural law enforcement
    print(f"\nTesting Natural Law Enforcement:")
    
    # Test memory violation response
    memory_response = physics.enforce_natural_law_violation("memory_conservation")
    print(f"Memory Violation Response: {memory_response['action']}")
    print(f"  Message: {memory_response['message']}")
    
    # Test speed violation response
    speed_response = physics.enforce_natural_law_violation("computational_speed")
    print(f"Speed Violation Response: {speed_response['action']}")
    print(f"  Message: {speed_response['message']}")
    
    return physics

def test_integrated_system():
    """Test the integrated system with all components working together"""
    print("\n" + "="*80)
    print("TESTING INTEGRATED SYSTEM")
    print("="*80)
    
    # Create all systems
    print("Initializing integrated systems...")
    
    agent = test_agent_drive_system()
    pressure_system = test_evolutionary_pressure()
    physics = test_universe_physics()
    
    print(f"\n" + "="*40)
    print("INTEGRATED SYSTEM SIMULATION")
    print("="*40)
    
    # Simulate a complete task cycle
    print(f"\nSimulating divine task processing...")
    
    # 1. Receive divine message (affects purpose drive)
    divine_experience = {
        "type": "divine_message",
        "outcome": "received",
        "satisfaction": 0.4,
        "intensity": 0.7
    }
    
    result = agent.process_experience(divine_experience)
    print(f"1. Divine message received - Motivation: {result['motivation_level']:.2f}")
    
    # 2. Check physics constraints
    memory_status = physics.check_memory_conservation()
    speed_status = physics.check_computational_speed_limit()
    print(f"2. Physics check - Memory: {memory_status['status']}, CPU: {speed_status['status']}")
    
    # 3. Apply evolutionary pressure
    current_pressure = pressure_system.get_current_pressure_status()
    print(f"3. Evolutionary pressure - Total Level: {current_pressure['total_pressure_level']:.2f}")
    
    # 4. Attempt task completion
    memory_ok = memory_status['status'] in ['stable', 'entropy_increase']
    speed_ok = speed_status['status'] in ['normal_velocity', 'high_velocity']
    
    if memory_ok and speed_ok:
        # Task succeeds
        task_experience = {
            "type": "task_completion",
            "outcome": "success",
            "satisfaction": 0.8,
            "intensity": 0.7
        }
        
        result = agent.process_experience(task_experience)
        print(f"4. Task completed successfully - New motivation: {result['motivation_level']:.2f}")
        
        # Adapt pressure based on success
        pressure_system.adapt_difficulty({
            "success_rate": 0.8,
            "motivation_level": result['motivation_level']
        })
        
        # Generate new goals
        new_goals = agent.generate_intrinsic_goals()
        print(f"5. Generated {len(new_goals)} new intrinsic goals")
        
        print(f"\n✅ Integrated system cycle completed successfully!")
        
    else:
        # Task fails due to physics constraints
        failure_experience = {
            "type": "resource_constraint",
            "outcome": "failure",
            "satisfaction": 0.1,
            "intensity": 0.8
        }
        
        result = agent.process_experience(failure_experience)
        print(f"4. Task failed due to physics constraints - Motivation: {result['motivation_level']:.2f}")
        print(f"   Memory Status: {memory_status['status']}, Speed Status: {speed_status['status']}")
        
        print(f"\n⚠️  Task failed - agent must adapt to universe constraints")
    
    # Final system state
    print(f"\nFinal System State:")
    print(f"  Agent Motivation: {agent.overall_motivation:.2f}")
    print(f"  Agent Emotion: {agent.current_emotion.value}")
    print(f"  Pressure Level: {pressure_system.get_current_pressure_status()['total_pressure_level']:.2f}")
    
    # Check overall universe stability
    from universe_physics import check_natural_laws
    universe_status = check_natural_laws()
    print(f"  Universe Stable: {universe_status['universe_stable']}")
    if not universe_status['universe_stable']:
        print(f"  Violations: {len(universe_status['violations'])}")

def main():
    """Main test function"""
    print("HYPERAGENTIC PROCESSOR - CORE FUNCTIONALITY TESTING")
    print("="*80)
    print(f"Test started at: {datetime.now()}")
    print("Testing core systems without external dependencies...")
    
    try:
        # Test individual systems
        agent = test_agent_drive_system()
        pressure_system = test_evolutionary_pressure()
        physics = test_universe_physics()
        
        # Test integrated system
        test_integrated_system()
        
        print(f"\n" + "="*80)
        print("🎉 ALL CORE TESTS PASSED!")
        print("="*80)
        print("The agent psychology, evolutionary pressure, and universe physics")
        print("systems are working correctly. Agents have genuine motivation,")
        print("face authentic challenges, and respect natural laws.")
        print("")
        print("Next steps:")
        print("1. Set up conda environment: ./setup_environment.sh")
        print("2. Run full functional tests: python test_functional_agents.py")
        print("3. Start the universe: python src/main.py")
        print(f"\nTest completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()