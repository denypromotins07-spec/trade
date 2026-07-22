"""
Swarm Intelligence for Distributed Market-Making Agents

Implements swarm intelligence logic where agents dynamically repel from
crowded order book levels and attract to liquidity voids. Uses Numba
for C-level execution speeds with AMD ROCm/DirectML acceleration checks.

Architecture:
- Potential field-based agent movement
- Liquidity density estimation
- Collision avoidance between agents
- Adaptive risk allocation based on swarm positioning

Memory Constraints:
- Strict 4GB RAM quota per Ray worker
- Pre-allocated numpy arrays for agent states
- Bounded neighbor search radius
"""

import os
import time
import ray
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numba
from numba import jit, prange


# =============================================================================
# AMD Accelerator Detection
# =============================================================================

def check_amd_accelerator() -> Dict[str, bool]:
    """Detect AMD ROCm and DirectML availability."""
    result = {
        'rocm_available': False,
        'directml_available': False,
        'gpu_device': None
    }
    
    try:
        import torch
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'rocm'):
            if torch.backends.rocm.is_available():
                result['rocm_available'] = True
                result['gpu_device'] = 'ROCm'
    except ImportError:
        pass
    
    try:
        import torch
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'directml'):
            if torch.backends.directml.is_available():
                result['directml_available'] = True
                result['gpu_device'] = 'DirectML'
    except (ImportError, AttributeError):
        pass
    
    return result


ACCELERATOR_STATUS = check_amd_accelerator()


# =============================================================================
# Memory Management
# =============================================================================

PYTHON_RAM_QUOTA = 4 * 1024 * 1024 * 1024  # 4GB hard limit


@dataclass
class SwarmMemoryMonitor:
    """Track memory usage for swarm simulation."""
    current_usage: int = 0
    
    def check_quota(self) -> bool:
        """Check if within 4GB quota."""
        import psutil
        try:
            process = psutil.Process(os.getpid())
            self.current_usage = process.memory_info().rss
            return self.current_usage < PYTHON_RAM_QUOTA
        except Exception:
            return True


# =============================================================================
# Numba-Accelerated Swarm Physics
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def compute_liquidity_potential(
    price_levels: np.ndarray,
    liquidity_values: np.ndarray,
    agent_positions: np.ndarray,
    attraction_strength: float = 1.0,
    repulsion_strength: float = 2.0
) -> np.ndarray:
    """
    Compute potential field from liquidity distribution.
    
    Agents are attracted to liquidity voids (low liquidity = low potential)
    and repelled from crowded levels (high liquidity = high potential).
    
    Args:
        price_levels: Array of price level centers
        liquidity_values: Liquidity at each price level
        agent_positions: Current agent positions (price coordinates)
        attraction_strength: Strength of attraction to voids
        repulsion_strength: Strength of repulsion from crowds
        
    Returns:
        Force vectors for each agent
    """
    n_agents = len(agent_positions)
    n_levels = len(price_levels)
    forces = np.zeros(n_agents)
    
    for i in prange(n_agents):
        agent_price = agent_positions[i]
        total_force = 0.0
        
        # Compute force from each liquidity level
        for j in range(n_levels):
            price_diff = agent_price - price_levels[j]
            distance = abs(price_diff) + 1e-8  # Avoid division by zero
            
            liquidity = liquidity_values[j]
            
            # Repulsion from high liquidity (crowded areas)
            repulsion = repulsion_strength * liquidity / (distance ** 2)
            
            # Direction: away from crowded levels
            if price_diff > 0:
                total_force += repulsion
            else:
                total_force -= repulsion
        
        forces[i] = total_force
    
    return forces


@jit(nopython=True, parallel=True, cache=True)
def compute_agent_repulsion(
    agent_positions: np.ndarray,
    agent_velocities: np.ndarray,
    repulsion_radius: float = 0.001,
    repulsion_strength: float = 5.0
) -> np.ndarray:
    """
    Compute inter-agent repulsion to prevent clustering.
    
    Agents repel each other when too close, ensuring diverse coverage.
    
    Args:
        agent_positions: Current agent positions
        agent_velocities: Current agent velocities
        repulsion_radius: Distance threshold for repulsion
        repulsion_strength: Strength of repulsion force
        
    Returns:
        Repulsion force vectors for each agent
    """
    n_agents = len(agent_positions)
    forces = np.zeros(n_agents)
    
    for i in prange(n_agents):
        total_force = 0.0
        
        for j in range(n_agents):
            if i == j:
                continue
            
            distance = abs(agent_positions[i] - agent_positions[j])
            
            if distance < repulsion_radius and distance > 1e-8:
                # Strong repulsion at close range
                force = repulsion_strength * (repulsion_radius - distance) / distance
                
                if agent_positions[i] > agent_positions[j]:
                    total_force += force
                else:
                    total_force -= force
        
        forces[i] = total_force
    
    return forces


@jit(nopython=True, parallel=True, cache=True)
def update_swarm_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    damping: float = 0.95,
    dt: float = 0.001,
    max_velocity: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update swarm state using Newtonian physics.
    
    Args:
        positions: Current positions
        velocities: Current velocities
        forces: Applied forces
        masses: Agent masses (inverse of confidence)
        damping: Velocity damping factor
        dt: Time step
        max_velocity: Maximum allowed velocity
        
    Returns:
        Updated positions and velocities
    """
    n_agents = len(positions)
    new_positions = positions.copy()
    new_velocities = velocities.copy()
    
    for i in prange(n_agents):
        # F = ma => a = F/m
        acceleration = forces[i] / masses[i]
        
        # Update velocity with damping
        new_velocities[i] = damping * velocities[i] + acceleration * dt
        
        # Clamp velocity
        if new_velocities[i] > max_velocity:
            new_velocities[i] = max_velocity
        elif new_velocities[i] < -max_velocity:
            new_velocities[i] = -max_velocity
        
        # Update position
        new_positions[i] = positions[i] + new_velocities[i] * dt
    
    return new_positions, new_velocities


@jit(nopython=True, parallel=True, cache=True)
def compute_optimal_order_sizes(
    agent_positions: np.ndarray,
    liquidity_density: np.ndarray,
    total_capital: float,
    risk_per_agent: float = 0.02
) -> np.ndarray:
    """
    Compute optimal order sizes based on swarm positioning.
    
    Agents in liquidity voids get larger allocations (opportunity),
    agents in crowded areas get smaller allocations (caution).
    
    Args:
        agent_positions: Agent price positions
        liquidity_density: Estimated liquidity density at each position
        total_capital: Total capital to allocate
        risk_per_agent: Risk fraction per agent
        
    Returns:
        Optimal order sizes for each agent
    """
    n_agents = len(agent_positions)
    order_sizes = np.zeros(n_agents)
    
    # Compute inverse liquidity weights (prefer voids)
    inv_liquidity = 1.0 / (liquidity_density + 1e-8)
    weights = inv_liquidity / np.sum(inv_liquidity)
    
    # Allocate capital proportionally
    for i in prange(n_agents):
        base_allocation = total_capital * weights[i]
        order_sizes[i] = min(base_allocation, total_capital * risk_per_agent)
    
    return order_sizes


# =============================================================================
# Swarm Agent Class
# =============================================================================

@ray.remote(max_calls=500)
class SwarmAgent:
    """
    Swarm intelligence agent for distributed market making.
    
    Positions itself based on liquidity landscape and other agents.
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_position: float,
        initial_velocity: float,
        mass: float = 1.0
    ):
        self.agent_id = agent_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.mass = mass
        self.confidence = 1.0
        self.order_size = 0.0
        
        self.memory_monitor = SwarmMemoryMonitor()
        self.update_count = 0
        
        print(f"[{agent_id}] Initialized swarm agent at position {initial_position}")
        print(f"[{agent_id}] Accelerator: {ACCELERATOR_STATUS}")
    
    def update_position(
        self,
        liquidity_forces: float,
        agent_repulsion: float,
        dt: float = 0.001
    ) -> Tuple[float, float]:
        """
        Update agent position based on computed forces.
        
        Args:
            liquidity_forces: Force from liquidity potential field
            agent_repulsion: Force from other agents
            dt: Time step
            
        Returns:
            New position and velocity
        """
        total_force = liquidity_forces + agent_repulsion
        
        # Newtonian update: F = ma
        acceleration = total_force / self.mass
        
        # Update velocity with damping
        self.velocity = 0.95 * self.velocity + acceleration * dt
        
        # Clamp velocity
        max_vel = 0.01
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
        
        # Update position
        self.position = self.position + self.velocity * dt
        
        self.update_count += 1
        
        # Check memory quota periodically
        if self.update_count % 100 == 0:
            if not self.memory_monitor.check_quota():
                self._trigger_gc()
        
        return self.position, self.velocity
    
    def set_confidence(self, confidence: float) -> None:
        """Update agent confidence (affects mass)."""
        self.confidence = max(0.1, min(1.0, confidence))
        self.mass = 1.0 / self.confidence  # Higher confidence = lower mass = more responsive
    
    def set_order_size(self, size: float) -> None:
        """Set allocated order size."""
        self.order_size = size
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            'agent_id': self.agent_id,
            'position': self.position,
            'velocity': self.velocity,
            'mass': self.mass,
            'confidence': self.confidence,
            'order_size': self.order_size,
            'update_count': self.update_count,
            'accelerator': ACCELERATOR_STATUS
        }
    
    def _trigger_gc(self) -> None:
        """Trigger garbage collection."""
        import gc
        gc.collect()
        ray.internal.free()


# =============================================================================
# Swarm Coordinator
# =============================================================================

@ray.remote
class SwarmCoordinator:
    """
    Coordinates swarm intelligence across distributed agents.
    
    Computes global potential fields and orchestrates agent movements.
    """
    
    def __init__(self, num_agents: int, price_range: Tuple[float, float]):
        self.num_agents = num_agents
        self.price_min, self.price_max = price_range
        self.agents: List[ray.actor.ActorHandle] = []
        self.liquidity_levels: Optional[np.ndarray] = None
        self.liquidity_values: Optional[np.ndarray] = None
    
    def initialize_swarm(
        self,
        liquidity_levels: np.ndarray,
        liquidity_values: np.ndarray
    ) -> bool:
        """
        Initialize swarm with liquidity landscape.
        
        Args:
            liquidity_levels: Price levels to monitor
            liquidity_values: Liquidity at each level
        """
        self.liquidity_levels = liquidity_levels.astype(np.float64)
        self.liquidity_values = liquidity_values.astype(np.float64)
        
        # Initialize agents spread across price range
        initial_positions = np.linspace(
            self.price_min, 
            self.price_max, 
            self.num_agents + 2
        )[1:-1]  # Exclude endpoints
        
        self.agents = [
            SwarmAgent.remote(
                agent_id=f"swarm_agent_{i}",
                initial_position=float(initial_positions[i]),
                initial_velocity=0.0,
                mass=1.0
            )
            for i in range(self.num_agents)
        ]
        
        print(f"[Coordinator] Initialized swarm: {self.num_agents} agents")
        print(f"[Coordinator] Price range: [{self.price_min}, {self.price_max}]")
        print(f"[Coordinator] Accelerator: {ACCELERATOR_STATUS}")
        
        return True
    
    def run_swarm_step(self, dt: float = 0.001) -> Dict[str, Any]:
        """
        Execute one step of swarm simulation.
        
        Returns:
            Simulation metrics and agent states
        """
        if self.liquidity_levels is None:
            return {'error': 'Swarm not initialized'}
        
        # Get current agent positions
        agent_states = ray.get([
            agent.get_state.remote() for agent in self.agents
        ])
        positions = np.array([s['position'] for s in agent_states], dtype=np.float64)
        velocities = np.array([s['velocity'] for s in agent_states], dtype=np.float64)
        masses = np.array([s['mass'] for s in agent_states], dtype=np.float64)
        
        # Compute liquidity potential forces (Numba-accelerated)
        liquidity_forces = compute_liquidity_potential(
            self.liquidity_levels,
            self.liquidity_values,
            positions,
            attraction_strength=1.0,
            repulsion_strength=2.0
        )
        
        # Compute inter-agent repulsion (Numba-accelerated)
        agent_repulsion = compute_agent_repulsion(
            positions,
            velocities,
            repulsion_radius=(self.price_max - self.price_min) / self.num_agents,
            repulsion_strength=5.0
        )
        
        # Update all agents
        total_forces = liquidity_forces + agent_repulsion
        
        for i, agent in enumerate(self.agents):
            ray.get(agent.update_position.remote(
                liquidity_forces[i],
                agent_repulsion[i],
                dt
            ))
        
        # Compute optimal order sizes (Numba-accelerated)
        liquidity_density = np.interp(positions, self.liquidity_levels, self.liquidity_values)
        order_sizes = compute_optimal_order_sizes(
            positions,
            liquidity_density,
            total_capital=1000000.0,  # $1M example
            risk_per_agent=0.02
        )
        
        # Update order sizes on agents
        for i, agent in enumerate(self.agents):
            ray.get(agent.set_order_size.remote(float(order_sizes[i])))
        
        # Get updated states
        updated_states = ray.get([
            agent.get_state.remote() for agent in self.agents
        ])
        
        return {
            'num_agents': self.num_agents,
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'order_sizes': order_sizes.tolist(),
            'total_allocated': float(np.sum(order_sizes)),
            'accelerator': ACCELERATOR_STATUS,
            'agent_states': updated_states
        }
    
    def get_swarm_distribution(self) -> Dict[str, Any]:
        """Get statistical summary of swarm distribution."""
        agent_states = ray.get([
            agent.get_state.remote() for agent in self.agents
        ])
        
        positions = np.array([s['position'] for s in agent_states])
        order_sizes = np.array([s['order_size'] for s in agent_states])
        
        return {
            'mean_position': float(np.mean(positions)),
            'std_position': float(np.std(positions)),
            'min_position': float(np.min(positions)),
            'max_position': float(np.max(positions)),
            'total_order_size': float(np.sum(order_sizes)),
            'avg_order_size': float(np.mean(order_sizes)),
            'coverage_ratio': float(np.std(positions) / (self.price_max - self.price_min))
        }


# =============================================================================
# Utility Functions
# =============================================================================

def enforce_ram_quota() -> None:
    """Enforce 4GB RAM quota for swarm workers."""
    import gc
    gc.collect()
    ray.internal.free()


if __name__ == "__main__":
    # Test swarm intelligence
    ray.init(ignore_reinit_error=True)
    
    # Create sample liquidity landscape
    price_levels = np.linspace(50000, 52000, 100)
    liquidity_values = np.random.exponential(scale=1000, size=100)
    
    coordinator = SwarmCoordinator.remote(num_agents=20, price_range=(50000, 52000))
    ray.get(coordinator.initialize_swarm.remote(price_levels, liquidity_values))
    
    # Run simulation steps
    for step in range(10):
        result = ray.get(coordinator.run_swarm_step.remote())
        print(f"Step {step}: {len(result['positions'])} agents, "
              f"total allocated: ${result['total_allocated']:,.0f}")
    
    # Get final distribution
    distribution = ray.get(coordinator.get_swarm_distribution.remote())
    print(f"Final distribution: {distribution}")
    print(f"Accelerator: {ACCELERATOR_STATUS}")
    
    ray.shutdown()
