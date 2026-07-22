"""
MuZero-Lite MCTS Planner for Market Execution

Implements a lightweight Monte Carlo Tree Search planner that simulates future
market trajectories in learned latent space to evaluate execution actions without
environment interaction. Designed for the 4GB Python RAM quota.

Key Features:
- Latent-space MCTS (no environment interaction during planning)
- Memory-bounded search tree
- UCB1-based action selection with PUCT exploration
- Value and policy heads from world model
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
from collections import defaultdict


@dataclass
class MCTSConfig:
    """Configuration for MCTS Planner"""
    
    # Search parameters
    num_simulations: int = 800  # MCTS iterations per decision
    max_depth: int = 50  # Maximum rollout depth
    c_puct: float = 1.5  # PUCT exploration constant
    
    # Discount and scaling
    gamma: float = 0.99  # Reward discount factor
    value_scale: float = 0.1  # Scale factor for value predictions
    
    # Memory constraints (4GB quota)
    max_tree_nodes: int = 100_000  # Maximum nodes in search tree
    prune_threshold: int = 5  # Visit count threshold for pruning
    
    # Action space
    num_actions: int = 10  # Number of discrete actions
    action_names: List[str] = field(default_factory=lambda: [
        'buy_aggressive', 'buy_passive', 'hold', 'sell_passive', 'sell_aggressive',
        'cancel_bids', 'cancel_asks', 'rebalance_low', 'rebalance_high', 'hedge'
    ])


class MCTSNode:
    """
    Single node in the MCTS search tree.
    Stores statistics for UCB1 action selection.
    """
    
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children', 
                 'state_hidden', 'state_latent', 'action', 'depth']
    
    def __init__(
        self,
        prior: float = 0.0,
        state_hidden: Optional[torch.Tensor] = None,
        state_latent: Optional[torch.Tensor] = None,
        action: Optional[int] = None,
        depth: int = 0,
    ):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.state_hidden = state_hidden
        self.state_latent = state_latent
        self.action = action
        self.depth = depth
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0
    
    def ucb_value(self, c_puct: float, parent_visit_count: int) -> float:
        """Calculate UCB1 value for action selection."""
        if self.visit_count == 0:
            return c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + parent_visit_count)
        
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / self.visit_count
        
        return exploitation + exploration


class MCTSPlanner:
    """
    MuZero-lite MCTS planner operating in latent space.
    
    Uses world model to simulate trajectories without environment interaction.
    Memory-bounded to respect 4GB Python RAM quota.
    """
    
    def __init__(self, world_model, config: Optional[MCTSConfig] = None):
        """
        Initialize MCTS planner.
        
        Args:
            world_model: WorldModel instance for latent dynamics
            config: MCTS configuration
        """
        self.world_model = world_model
        self.config = config or MCTSConfig()
        
        # Search tree root
        self.root: Optional[MCTSNode] = None
        
        # Node counter for memory management
        self.node_count = 0
        
        # Action space
        self.actions = list(range(self.config.num_actions))
    
    def plan(
        self,
        initial_observation: np.ndarray,
        available_actions: Optional[List[int]] = None,
    ) -> Tuple[int, Dict]:
        """
        Run MCTS planning from initial observation.
        
        Args:
            initial_observation: Current LOB observation
            available_actions: Subset of actions to consider (optional)
            
        Returns:
            Best action and search statistics
        """
        if available_actions is None:
            available_actions = self.actions
        
        # Encode initial observation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(initial_observation).unsqueeze(0)
            latent_probs, hidden_state = self.world_model.encode_observation(obs_tensor)
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
        
        # Create root node
        self.root = MCTSNode(
            prior=1.0,
            state_hidden=hidden_state.cpu(),
            state_latent=latent_probs.cpu(),
            depth=0,
        )
        self.node_count = 1
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(self.root, available_actions)
        
        # Select best action based on visit count
        best_action = max(
            available_actions,
            key=lambda a: self.root.children.get(a, MCTSNode()).visit_count
            if a in self.root.children else 0
        )
        
        # Gather statistics
        stats = self._gather_statistics()
        
        # Prune tree to manage memory
        self._prune_tree()
        
        return best_action, stats
    
    def _simulate(self, node: MCTSNode, available_actions: List[int]) -> float:
        """
        Run single MCTS simulation.
        
        Returns the value obtained from the simulation.
        """
        # Track path for backpropagation
        path = []
        current_node = node
        current_hidden = node.state_hidden
        current_latent = node.state_latent
        discount = 1.0
        total_reward = 0.0
        
        # Selection & Expansion
        while not current_node.is_leaf and current_node.depth < self.config.max_depth:
            path.append(current_node)
            
            # Select action using UCB1
            action = self._select_action(current_node, available_actions)
            
            # Transition to child
            if action in current_node.children:
                child = current_node.children[action]
            else:
                # Expand new node
                child = self._expand_node(current_node, action, current_hidden, current_latent)
                current_node.children[action] = child
            
            current_node = child
            
            # Simulate dynamics in latent space
            if current_node.state_hidden is not None and current_node.state_latent is not None:
                with torch.no_grad():
                    action_tensor = torch.zeros(1, self.world_model.config.action_dim)
                    if action < action_tensor.shape[1]:
                        action_tensor[0, action] = 1.0
                    
                    next_latent, next_hidden = self.world_model.predict_next_state(
                        current_node.state_latent,
                        current_node.state_hidden,
                        action_tensor,
                    )
                    
                    # Decode and get reward prediction
                    pred_obs = self.world_model.decode_latent(next_latent, next_hidden)
                    reward_pred = self.world_model.reward_head(
                        torch.cat([next_latent, next_hidden], dim=-1)
                    )
                    
                    total_reward += discount * reward_pred.item() * self.config.value_scale
                    discount *= self.config.gamma
                    
                    current_hidden = next_hidden.cpu()
                    current_latent = next_latent.cpu()
        
        # Backpropagation
        value = total_reward
        if current_node.is_leaf and current_node.depth >= self.config.max_depth:
            # Use value head at max depth
            with torch.no_grad():
                value_pred = self.world_model.reward_head(
                    torch.cat([current_latent, current_hidden], dim=-1)
                )
                value = total_reward + discount * value_pred.item() * self.config.value_scale
        
        # Update all nodes in path
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value *= self.config.gamma  # Discount for parent
        
        return value
    
    def _select_action(self, node: MCTSNode, available_actions: List[int]) -> int:
        """Select action using PUCT algorithm."""
        best_action = None
        best_value = -float('inf')
        
        for action in available_actions:
            if action in node.children:
                child = node.children[action]
                ucb_val = child.ucb_value(self.config.c_puct, node.visit_count)
            else:
                # Unexplored action - use prior
                prior = 1.0 / len(available_actions)  # Uniform prior
                ucb_val = self.config.c_puct * prior * math.sqrt(node.visit_count) / (1 + node.visit_count)
            
            if ucb_val > best_value:
                best_value = ucb_val
                best_action = action
        
        return best_action if best_action is not None else available_actions[0]
    
    def _expand_node(
        self,
        parent: MCTSNode,
        action: int,
        parent_hidden: torch.Tensor,
        parent_latent: torch.Tensor,
    ) -> MCTSNode:
        """Expand a new node in the search tree."""
        # Check memory limit
        if self.node_count >= self.config.max_tree_nodes:
            # Return a minimal node without full expansion
            return MCTSNode(prior=0.0, action=action, depth=parent.depth + 1)
        
        # Simulate one step in latent space
        with torch.no_grad():
            action_tensor = torch.zeros(1, self.world_model.config.action_dim)
            if action < action_tensor.shape[1]:
                action_tensor[0, action] = 1.0
            
            next_latent, next_hidden = self.world_model.predict_next_state(
                parent_latent,
                parent_hidden,
                action_tensor,
            )
            
            # Get policy prior for this action
            policy_logits = self.world_model.dynamics.prior_head(next_hidden)
            policy_probs = torch.softmax(policy_logits, dim=-1)
            
            prior = policy_probs[0, action].item() if action < policy_probs.shape[1] else 0.1
        
        self.node_count += 1
        
        return MCTSNode(
            prior=prior,
            state_hidden=next_hidden.cpu(),
            state_latent=next_latent.cpu(),
            action=action,
            depth=parent.depth + 1,
        )
    
    def _gather_statistics(self) -> Dict:
        """Gather MCTS search statistics."""
        if self.root is None:
            return {}
        
        # Visit counts per action
        visit_counts = {
            self.config.action_names[a] if a < len(self.config.action_names) else f'action_{a}": 
            self.root.children.get(a, MCTSNode()).visit_count
            for a in self.actions
        }
        
        # Values per action
        values = {
            self.config.action_names[a] if a < len(self.config.action_names) else f'action_{a}": 
            self.root.children.get(a, MCTSNode()).value
            for a in self.actions
        }
        
        # Find best action
        best_action = max(visit_counts, key=visit_counts.get)
        
        return {
            'visit_counts': visit_counts,
            'values': values,
            'best_action': best_action,
            'total_nodes': self.node_count,
            'total_visits': self.root.visit_count,
        }
    
    def _prune_tree(self):
        """Prune search tree to manage memory."""
        if self.root is None:
            return
        
        def prune_node(node: MCTSNode) -> bool:
            """Recursively prune low-visit children. Returns True if node should be kept."""
            # Keep nodes with sufficient visits
            if node.visit_count >= self.config.prune_threshold:
                # Recursively prune children
                to_remove = []
                for action, child in node.children.items():
                    if not prune_node(child):
                        to_remove.append(action)
                
                for action in to_remove:
                    del node.children[action]
                    self.node_count -= 1
                
                return True
            else:
                # Remove this node's children
                self.node_count -= len(node.children)
                node.children = {}
                return False
        
        # Prune children of root (keep root)
        to_remove = []
        for action, child in self.root.children.items():
            if not prune_node(child):
                to_remove.append(action)
        
        for action in to_remove:
            del self.root.children[action]
            self.node_count -= 1


class LatentRolloutBuffer:
    """
    Memory-bounded buffer for latent rollouts.
    Enforces 4GB RAM quota by limiting stored trajectories.
    """
    
    def __init__(self, max_trajectories: int = 10_000, max_length: int = 100):
        self.max_trajectories = max_trajectories
        self.max_length = max_length
        
        self.trajectories = []
        self.latent_dim = 256  # Will be set dynamically
        self.hidden_dim = 512
    
    def add_trajectory(
        self,
        latents: np.ndarray,
        hiddens: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ):
        """Add trajectory with memory bounds enforcement."""
        # Truncate to max length
        length = min(len(latents), self.max_length)
        
        trajectory = {
            'latents': latents[:length],
            'hiddens': hiddens[:length],
            'actions': actions[:length],
            'rewards': rewards[:length],
        }
        
        self.trajectories.append(trajectory)
        
        # Enforce memory bound
        while len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)
    
    def sample_batch(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Sample batch of trajectories for training."""
        if len(self.trajectories) == 0:
            return {}
        
        indices = np.random.choice(
            len(self.trajectories),
            min(batch_size, len(self.trajectories)),
            replace=False
        )
        
        batch = {
            'latents': [],
            'hiddens': [],
            'actions': [],
            'rewards': [],
        }
        
        for idx in indices:
            traj = self.trajectories[idx]
            batch['latents'].append(traj['latents'])
            batch['hiddens'].append(traj['hiddens'])
            batch['actions'].append(traj['actions'])
            batch['rewards'].append(traj['rewards'])
        
        return batch
    
    def clear(self):
        """Clear buffer to free memory."""
        self.trajectories.clear()
    
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        if len(self.trajectories) == 0:
            return 0.0
        
        # Estimate per-trajectory size
        avg_len = sum(len(t['latents']) for t in self.trajectories) / len(self.trajectories)
        bytes_per_traj = avg_len * (self.latent_dim * 4 + self.hidden_dim * 4 + 8 + 8)  # float32
        
        return (len(self.trajectories) * bytes_per_traj) / (1024 * 1024)


# Example usage
if __name__ == "__main__":
    print("MCTS Planner module loaded successfully")
    
    # Test configuration
    config = MCTSConfig(
        num_simulations=100,  # Reduced for testing
        max_depth=20,
        num_actions=5,
    )
    
    print(f"MCTS Config: {config.num_simulations} simulations, max depth {config.max_depth}")
    print(f"Action space: {config.action_names[:config.num_actions]}")
