"""
Evolutionary Strategy Module for Hyperparameter Mutation
Automatically mutates underperforming model hyperparameters.
Breeds new algorithmic variations based on past failures.
Safely quarantines degraded models before live capital exposure.
AMD ROCm/DirectML acceleration support for tensor operations.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
import random


def check_amd_acceleration() -> str:
    """Detect available AMD acceleration backend."""
    if os.environ.get("ROCM_PATH") or os.path.exists("/opt/rocm"):
        try:
            import torch
            if torch.cuda.is_available():
                return "rocm"
        except ImportError:
            pass
    
    if os.environ.get("DIRECTML_ENABLED") == "1":
        try:
            import torch_directml
            return "directml"
        except ImportError:
            pass
    
    return "cpu"


class ModelStatus(Enum):
    """Model lifecycle status."""
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    RETIRED = "retired"
    CANDIDATE = "candidate"


@dataclass
class ModelGenome:
    """Hyperparameter genome for a trading model."""
    model_id: str
    # Strategy parameters
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    position_size_pct: float = 0.01
    max_positions: int = 5
    # Risk parameters
    max_drawdown_pct: float = 0.10
    kelly_fraction: float = 0.25
    # RL parameters
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    # Performance tracking
    fitness_score: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    # Status
    status: ModelStatus = ModelStatus.CANDIDATE
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary."""
        return {
            "model_id": self.model_id,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "max_drawdown_pct": self.max_drawdown_pct,
            "kelly_fraction": self.kelly_fraction,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "fitness_score": self.fitness_score,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "status": self.status.value,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


class EvolutionaryStrategy:
    """
    Evolutionary algorithm for hyperparameter optimization.
    
    Uses genetic algorithm principles:
    - Selection: Tournament selection based on fitness
    - Crossover: Blend parent hyperparameters
    - Mutation: Gaussian perturbation with adaptive variance
    - Quarantine: Isolate underperforming models
    """
    
    def __init__(
        self,
        population_size: int = 50,
        elite_count: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        quarantine_threshold: float = -0.5,
    ):
        """
        Initialize evolutionary strategy.
        
        Parameters
        ----------
        population_size : int
            Number of models in population
        elite_count : int
            Number of top models preserved each generation
        mutation_rate : float
            Probability of mutating each gene
        crossover_rate : float
            Probability of crossover vs cloning
        quarantine_threshold : float
            Fitness threshold for quarantine
        """
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.quarantine_threshold = quarantine_threshold
        self.accelerator = check_amd_acceleration()
        
        # Population storage
        self.population: Dict[str, ModelGenome] = {}
        self.quarantine: Dict[str, ModelGenome] = {}
        self.history: List[Dict] = []
        
        # Generation counter
        self.current_generation = 0
        
        # Adaptive mutation parameters
        self.mutation_std = 0.1
        self.min_mutation_std = 0.01
        self.max_mutation_std = 0.5
    
    def initialize_population(self, base_params: Optional[Dict] = None) -> List[str]:
        """
        Initialize population with random variations.
        
        Returns
        -------
        List[str]
            IDs of created models
        """
        model_ids = []
        base = base_params or {}
        
        for i in range(self.population_size):
            model_id = f"GEN{self.current_generation:03d}_M{i:03d}"
            
            genome = ModelGenome(
                model_id=model_id,
                entry_threshold=base.get("entry_threshold", 2.0) + np.random.uniform(-0.5, 0.5),
                exit_threshold=base.get("exit_threshold", 0.5) + np.random.uniform(-0.2, 0.2),
                stop_loss_pct=base.get("stop_loss_pct", 0.02) + np.random.uniform(-0.005, 0.005),
                take_profit_pct=base.get("take_profit_pct", 0.05) + np.random.uniform(-0.01, 0.01),
                position_size_pct=base.get("position_size_pct", 0.01) * (1 + np.random.uniform(-0.3, 0.3)),
                max_positions=base.get("max_positions", 5) + random.randint(-1, 1),
                kelly_fraction=base.get("kelly_fraction", 0.25) * (1 + np.random.uniform(-0.2, 0.2)),
                learning_rate=base.get("learning_rate", 0.001) * (10 ** np.random.uniform(-1, 1)),
                exploration_rate=base.get("exploration_rate", 0.1) * (1 + np.random.uniform(-0.5, 0.5)),
                generation=self.current_generation,
            )
            
            # Ensure valid ranges
            genome.entry_threshold = max(0.1, genome.entry_threshold)
            genome.exit_threshold = max(0.01, genome.exit_threshold)
            genome.stop_loss_pct = np.clip(genome.stop_loss_pct, 0.001, 0.1)
            genome.take_profit_pct = np.clip(genome.take_profit_pct, 0.01, 0.2)
            genome.position_size_pct = np.clip(genome.position_size_pct, 0.001, 0.1)
            genome.max_positions = max(1, genome.max_positions)
            genome.kelly_fraction = np.clip(genome.kelly_fraction, 0.01, 1.0)
            genome.learning_rate = np.clip(genome.learning_rate, 1e-6, 0.1)
            genome.exploration_rate = np.clip(genome.exploration_rate, 0.01, 0.5)
            
            self.population[model_id] = genome
            model_ids.append(model_id)
        
        return model_ids
    
    def calculate_fitness(
        self, 
        genome: ModelGenome, 
        returns: np.ndarray,
        drawdowns: np.ndarray
    ) -> float:
        """
        Calculate fitness score for a genome.
        
        Combines multiple metrics:
        - Sharpe ratio (risk-adjusted returns)
        - Win rate
        - Maximum drawdown penalty
        - Trade count regularization
        """
        if genome.total_trades < 10:
            return -1.0  # Not enough data
        
        # Sharpe component (annualized)
        if len(returns) > 1 and np.std(returns) > 1e-6:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Win rate component
        win_rate_component = genome.win_rate - 0.5  # Center around 0.5
        
        # Drawdown penalty
        max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        dd_penalty = max(0, abs(max_dd) - genome.max_drawdown_pct) * 10
        
        # Trade count regularization (prefer more trades up to a point)
        trade_bonus = min(np.log10(genome.total_trades + 1) / 4, 1.0)
        
        # Combined fitness
        fitness = (
            0.4 * sharpe +
            0.2 * win_rate_component +
            0.3 * (-dd_penalty) +
            0.1 * trade_bonus
        )
        
        return fitness
    
    def tournament_selection(
        self, 
        tournament_size: int = 5
    ) -> ModelGenome:
        """Select a parent using tournament selection."""
        candidates = random.sample(
            list(self.population.values()),
            min(tournament_size, len(self.population))
        )
        return max(candidates, key=lambda g: g.fitness_score)
    
    def crossover(
        self, 
        parent1: ModelGenome, 
        parent2: ModelGenome
    ) -> ModelGenome:
        """
        Create offspring by blending parent hyperparameters.
        Uses blend crossover (BLX-α) for continuous parameters.
        """
        alpha = 0.3  # BLX-α parameter
        
        child = ModelGenome(
            model_id=f"GEN{self.current_generation:03d}_C{random.randint(0, 999):03d}",
            generation=self.current_generation,
            parent_ids=[parent1.model_id, parent2.model_id],
        )
        
        # Blend continuous parameters
        for param in ["entry_threshold", "exit_threshold", "stop_loss_pct", 
                      "take_profit_pct", "position_size_pct", "kelly_fraction",
                      "learning_rate", "exploration_rate"]:
            p1_val = getattr(parent1, param)
            p2_val = getattr(parent2, param)
            
            min_val = min(p1_val, p2_val)
            max_val = max(p1_val, p2_val)
            diff = max_val - min_val
            
            # BLX-α range
            lower = max(0, min_val - alpha * diff)
            upper = max_val + alpha * diff
            
            setattr(child, param, np.random.uniform(lower, upper))
        
        # Integer parameters (simple average with rounding)
        child.max_positions = round((parent1.max_positions + parent2.max_positions) / 2)
        child.max_positions = max(1, child.max_positions)
        
        # Discount factor (special handling)
        child.discount_factor = np.random.uniform(
            min(parent1.discount_factor, parent2.discount_factor),
            max(parent1.discount_factor, parent2.discount_factor)
        )
        
        return child
    
    def mutate(self, genome: ModelGenome) -> ModelGenome:
        """
        Apply Gaussian mutation to hyperparameters.
        Uses adaptive mutation rate based on population diversity.
        """
        mutated = copy.deepcopy(genome)
        mutated.model_id = f"GEN{self.current_generation:03d}_M{random.randint(0, 999):03d}"
        
        # Continuous parameters
        float_params = [
            "entry_threshold", "exit_threshold", "stop_loss_pct",
            "take_profit_pct", "position_size_pct", "kelly_fraction",
            "learning_rate", "exploration_rate"
        ]
        
        for param in float_params:
            if random.random() < self.mutation_rate:
                current_val = getattr(mutated, param)
                
                # Adaptive mutation step
                step_size = current_val * self.mutation_std
                
                # Add mutation
                mutation = np.random.normal(0, step_size)
                new_val = current_val + mutation
                
                # Apply bounds
                bounds = self._get_param_bounds(param)
                new_val = np.clip(new_val, bounds[0], bounds[1])
                
                setattr(mutated, param, new_val)
        
        # Integer parameters
        if random.random() < self.mutation_rate:
            mutated.max_positions += random.choice([-1, 0, 1])
            mutated.max_positions = max(1, mutated.max_positions)
        
        return mutated
    
    def _get_param_bounds(self, param: str) -> Tuple[float, float]:
        """Get valid bounds for a parameter."""
        bounds = {
            "entry_threshold": (0.1, 10.0),
            "exit_threshold": (0.01, 5.0),
            "stop_loss_pct": (0.001, 0.1),
            "take_profit_pct": (0.01, 0.2),
            "position_size_pct": (0.001, 0.1),
            "kelly_fraction": (0.01, 1.0),
            "learning_rate": (1e-6, 0.1),
            "exploration_rate": (0.01, 0.5),
            "discount_factor": (0.9, 0.999),
        }
        return bounds.get(param, (0.0, 1.0))
    
    def evolve(self, performance_data: Dict[str, Dict]) -> List[str]:
        """
        Run one generation of evolution.
        
        Parameters
        ----------
        performance_data : Dict[str, Dict]
            Performance metrics for each model
        
        Returns
        -------
        List[str]
            IDs of new candidate models
        """
        # Update fitness scores
        for model_id, perf in performance_data.items():
            if model_id in self.population:
                genome = self.population[model_id]
                genome.fitness_score = self.calculate_fitness(
                    genome,
                    np.array(perf.get("returns", [])),
                    np.array(perf.get("drawdowns", []))
                )
                genome.win_rate = perf.get("win_rate", 0.5)
                genome.total_trades = perf.get("total_trades", 0)
                
                # Check for quarantine
                if genome.fitness_score < self.quarantine_threshold:
                    self.quarantine_model(model_id)
        
        # Sort by fitness
        sorted_pop = sorted(
            self.population.values(),
            key=lambda g: g.fitness_score,
            reverse=True
        )
        
        # Elitism: preserve top performers
        new_population = {}
        for genome in sorted_pop[:self.elite_count]:
            new_population[genome.model_id] = genome
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover or clone
            if random.random() < self.crossover_rate and parent1.model_id != parent2.model_id:
                child = self.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
                child.model_id = f"GEN{self.current_generation:03d}_N{random.randint(0, 999):03d}"
            
            # Mutation
            child = self.mutate(child)
            
            new_population[child.model_id] = child
        
        # Update population
        self.population = new_population
        self.current_generation += 1
        
        # Record history
        self.history.append({
            "generation": self.current_generation,
            "best_fitness": max(g.fitness_score for g in self.population.values()),
            "avg_fitness": np.mean([g.fitness_score for g in self.population.values()]),
            "population_size": len(self.population),
            "quarantine_count": len(self.quarantine),
        })
        
        # Adaptive mutation rate
        self._adapt_mutation_rate()
        
        return list(self.population.keys())
    
    def quarantine_model(self, model_id: str) -> bool:
        """Move underperforming model to quarantine."""
        if model_id in self.population:
            genome = self.population.pop(model_id)
            genome.status = ModelStatus.QUARANTINED
            self.quarantine[model_id] = genome
            return True
        return False
    
    def promote_from_quarantine(self, model_id: str) -> bool:
        """Promote a quarantined model back to population."""
        if model_id in self.quarantine:
            genome = self.quarantine.pop(model_id)
            genome.status = ModelStatus.CANDIDATE
            self.population[model_id] = genome
            return True
        return False
    
    def _adapt_mutation_rate(self) -> None:
        """Adapt mutation rate based on population diversity."""
        fitnesses = [g.fitness_score for g in self.population.values()]
        
        if len(fitnesses) > 1:
            diversity = np.std(fitnesses)
            
            # Low diversity -> increase mutation
            # High diversity -> decrease mutation
            if diversity < 0.1:
                self.mutation_std = min(self.max_mutation_std, self.mutation_std * 1.1)
            elif diversity > 0.5:
                self.mutation_std = max(self.min_mutation_std, self.mutation_std * 0.9)
    
    def get_best_models(self, n: int = 5) -> List[ModelGenome]:
        """Get top N models by fitness."""
        sorted_pop = sorted(
            self.population.values(),
            key=lambda g: g.fitness_score,
            reverse=True
        )
        return sorted_pop[:n]
    
    def get_active_models(self) -> List[ModelGenome]:
        """Get all active (non-quarantined) models."""
        return [
            g for g in self.population.values()
            if g.status == ModelStatus.ACTIVE or g.status == ModelStatus.CANDIDATE
        ]


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    es = EvolutionaryStrategy(population_size=20, elite_count=3)
    
    # Initialize population
    model_ids = es.initialize_population({
        "entry_threshold": 2.0,
        "kelly_fraction": 0.25,
    })
    
    print(f"Initialized {len(model_ids)} models")
    
    # Simulate performance data
    performance_data = {}
    for model_id in model_ids:
        performance_data[model_id] = {
            "returns": np.random.randn(100) * 0.02 + 0.001,
            "drawdowns": np.cummin(np.random.randn(100) * 0.01),
            "win_rate": 0.4 + np.random.uniform(0, 0.3),
            "total_trades": random.randint(50, 200),
        }
    
    # Evolve
    new_ids = es.evolve(performance_data)
    
    print(f"\nGeneration {es.current_generation}:")
    best = es.get_best_models(3)
    for model in best:
        print(f"  {model.model_id}: fitness={model.fitness_score:.3f}, win_rate={model.win_rate:.2f}")
    
    print(f"\nQuarantined models: {len(es.quarantine)}")
    print(f"History entries: {len(es.history)}")
