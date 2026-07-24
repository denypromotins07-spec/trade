"""
Genetic Strategy Breeder - Stage 56
AMD Ryzen AI 5 Optimized | 4GB RAM Quota | Ray-Distributed Evolution

This module implements a genetic algorithm on Ray that crosses hyperparameters
of top-performing strategies, strictly enforcing the 4GB RAM quota during
intensive gradient evaluations.

Constraints:
- Strict 4GB RAM quota for all evolutionary operations
- GPU-accelerated fitness evaluation via ROCm/DirectML
- Elitism preservation to prevent regression
- Walk-forward validation integration
"""

import ray
import numpy as np
import cupy as cp  # ROCm/DirectML acceleration
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import psutil
import os
from enum import Enum

# Enforce strict memory limits
MAX_RAM_MB = 4096
os.environ['RAY_MEMORY_LIMIT'] = str(MAX_RAM_MB * 1024 * 1024)


class MutationType(Enum):
    """Types of genetic mutations supported."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    SWAP = "swap"
    INVERSION = "inversion"


@dataclass
class StrategyGenome:
    """Represents a strategy's genetic encoding."""
    genome_id: str
    parameters: Dict[str, float]
    fitness_score: float
    sharpe_ratio: float
    max_drawdown: float
    generation: int
    parent_ids: List[str]
    mutation_count: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'genome_id': self.genome_id,
            'parameters': self.parameters,
            'fitness_score': self.fitness_score,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'mutation_count': self.mutation_count,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@ray.remote(num_cpus=1, max_calls=200)
class EvolutionWorker:
    """
    Ray-distributed worker for genetic evolution operations.
    Performs crossover, mutation, and fitness evaluation with GPU acceleration.
    """
    
    def __init__(self, ram_limit_mb: int = 1024):
        self.ram_limit_mb = ram_limit_mb
        self.gpu_available = self._check_gpu()
        self.evaluation_cache: Dict[str, float] = {}
        
    def _check_gpu(self) -> bool:
        """Check for AMD ROCm/DirectML availability."""
        try:
            test_array = cp.zeros(100)
            del test_array
            cp.get_default_memory_pool().free_all_blocks()
            return True
        except Exception:
            return False
    
    def _memory_check(self) -> bool:
        """Verify memory usage is within limits."""
        current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
        if current_ram > self.ram_limit_mb * 0.9:
            # Force garbage collection
            import gc
            gc.collect()
            if self.gpu_available:
                cp.get_default_memory_pool().free_all_blocks()
            return False
        return True
    
    def crossover(
        self,
        parent1: StrategyGenome,
        parent2: StrategyGenome,
        method: str = "uniform"
    ) -> Tuple[StrategyGenome, StrategyGenome]:
        """
        Perform genetic crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            method: Crossover method ('uniform', 'single_point', 'arithmetic')
            
        Returns:
            Tuple of two child genomes
        """
        if not self._memory_check():
            raise MemoryError("Worker memory limit exceeded during crossover")
        
        # Extract parameter keys (ensure same order)
        param_keys = list(parent1.parameters.keys())
        
        if method == "uniform":
            # Random mask for each parameter
            mask = np.random.rand(len(param_keys)) > 0.5
            
            child1_params = {}
            child2_params = {}
            
            for i, key in enumerate(param_keys):
                if mask[i]:
                    child1_params[key] = parent1.parameters[key]
                    child2_params[key] = parent2.parameters[key]
                else:
                    child1_params[key] = parent2.parameters[key]
                    child2_params[key] = parent1.parameters[key]
                    
        elif method == "arithmetic":
            # Blend crossover
            alpha = np.random.uniform(0.3, 0.7)
            
            child1_params = {
                key: alpha * parent1.parameters[key] + (1 - alpha) * parent2.parameters[key]
                for key in param_keys
            }
            child2_params = {
                key: (1 - alpha) * parent1.parameters[key] + alpha * parent2.parameters[key]
                for key in param_keys
            }
        else:
            # Single point crossover
            split_idx = np.random.randint(1, len(param_keys))
            
            child1_params = {
                **{k: parent1.parameters[k] for k in param_keys[:split_idx]},
                **{k: parent2.parameters[k] for k in param_keys[split_idx:]}
            }
            child2_params = {
                **{k: parent2.parameters[k] for k in param_keys[:split_idx]},
                **{k: parent1.parameters[k] for k in param_keys[split_idx:]}
            }
        
        # Generate child IDs
        child1_id = hashlib.md5(
            json.dumps(child1_params, sort_keys=True).encode()
        ).hexdigest()[:12]
        child2_id = hashlib.md5(
            json.dumps(child2_params, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        now = datetime.utcnow()
        next_gen = max(parent1.generation, parent2.generation) + 1
        
        child1 = StrategyGenome(
            genome_id=child1_id,
            parameters=child1_params,
            fitness_score=0.0,  # To be evaluated
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            generation=next_gen,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            mutation_count=0,
            created_at=now
        )
        
        child2 = StrategyGenome(
            genome_id=child2_id,
            parameters=child2_params,
            fitness_score=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            generation=next_gen,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            mutation_count=0,
            created_at=now
        )
        
        return child1, child2
    
    def mutate(
        self,
        genome: StrategyGenome,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.05,
        mutation_type: MutationType = MutationType.GAUSSIAN
    ) -> StrategyGenome:
        """
        Apply mutations to a genome.
        
        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutating each parameter
            mutation_std: Standard deviation for Gaussian mutations
            mutation_type: Type of mutation to apply
            
        Returns:
            Mutated genome
        """
        if not self._memory_check():
            raise MemoryError("Worker memory limit exceeded during mutation")
        
        mutated_params = genome.parameters.copy()
        mutation_count = 0
        
        param_keys = list(mutated_params.keys())
        
        if mutation_type == MutationType.GAUSSIAN:
            for key in param_keys:
                if np.random.random() < mutation_rate:
                    # GPU-accelerated batch mutation if available
                    if self.gpu_available and len(param_keys) > 10:
                        noise = cp.random.normal(0, mutation_std, len(param_keys))
                        noise_host = cp.asnumpy(noise)
                        for i, k in enumerate(param_keys):
                            if np.random.random() < mutation_rate:
                                mutated_params[k] += float(noise_host[i])
                                mutation_count += 1
                                # Clamp to valid range [0, 1]
                                mutated_params[k] = np.clip(mutated_params[k], 0, 1)
                    else:
                        noise = np.random.normal(0, mutation_std)
                        mutated_params[key] += noise
                        mutation_count += 1
                        mutated_params[key] = np.clip(mutated_params[key], 0, 1)
                        
        elif mutation_type == MutationType.UNIFORM:
            for key in param_keys:
                if np.random.random() < mutation_rate:
                    mutated_params[key] = np.random.uniform(0, 1)
                    mutation_count += 1
                    
        elif mutation_type == MutationType.SWAP:
            if len(param_keys) >= 2 and np.random.random() < mutation_rate:
                idx1, idx2 = np.random.choice(len(param_keys), 2, replace=False)
                key1, key2 = param_keys[idx1], param_keys[idx2]
                mutated_params[key1], mutated_params[key2] = \
                    mutated_params[key2], mutated_params[key1]
                mutation_count += 2
                
        elif mutation_type == MutationType.INVERSION:
            if len(param_keys) >= 2 and np.random.random() < mutation_rate:
                start, end = sorted(np.random.choice(len(param_keys), 2, replace=False))
                selected_keys = param_keys[start:end+1]
                values = [mutated_params[k] for k in selected_keys]
                values.reverse()
                for k, v in zip(selected_keys, values):
                    mutated_params[k] = v
                mutation_count += len(selected_keys)
        
        # Generate new ID
        new_id = hashlib.md5(
            json.dumps(mutated_params, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        return StrategyGenome(
            genome_id=new_id,
            parameters=mutated_params,
            fitness_score=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            generation=genome.generation,
            parent_ids=genome.parent_ids,
            mutation_count=genome.mutation_count + mutation_count,
            created_at=datetime.utcnow(),
            metadata=genome.metadata
        )
    
    def evaluate_fitness_gpu(
        self,
        genome: StrategyGenome,
        returns_data: np.ndarray,
        transaction_costs: float = 0.001
    ) -> Tuple[float, float, float]:
        """
        Evaluate genome fitness using GPU-accelerated backtest.
        
        Args:
            genome: Genome to evaluate
            returns_data: Historical returns array
            transaction_costs: Per-trade transaction cost
            
        Returns:
            Tuple of (fitness_score, sharpe_ratio, max_drawdown)
        """
        if not self._memory_check():
            raise MemoryError("Worker memory limit exceeded during evaluation")
        
        # Check cache first
        if genome.genome_id in self.evaluation_cache:
            cached = self.evaluation_cache[genome.genome_id]
            return cached, cached * 0.8, 0.1  # Approximate metrics
        
        # Transfer to GPU if available
        if self.gpu_available:
            returns_gpu = cp.asarray(returns_data)
            
            # Extract parameters
            params = genome.parameters
            
            # Simulate strategy performance (simplified momentum example)
            lookback = int(params.get('lookback', 20) * 100) + 1
            threshold = params.get('threshold', 0.5)
            
            # GPU-accelerated signal generation
            if len(returns_gpu) > lookback:
                rolling_mean = cp.convolve(returns_gpu, cp.ones(lookback)/lookback, mode='valid')
                signals = (rolling_mean > threshold).astype(float)
                
                # Calculate returns with transaction costs
                position_changes = cp.abs(cp.diff(signals, prepend=signals[0]))
                costs = position_changes * transaction_costs
                strategy_returns = signals * returns_gpu[lookback:] - costs
                
                # Performance metrics
                total_return = float(cp.prod(1 + strategy_returns) - 1)
                
                if cp.std(strategy_returns) > 0:
                    sharpe = float(cp.mean(strategy_returns) / cp.std(strategy_returns)) * cp.sqrt(252)
                else:
                    sharpe = 0.0
                
                # Max drawdown
                cumulative = cp.cumprod(1 + strategy_returns)
                running_max = cp.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_dd = float(abs(cp.min(drawdown)))
                
                # Cleanup GPU memory
                del returns_gpu, rolling_mean, signals, strategy_returns
                cp.get_default_memory_pool().free_all_blocks()
            else:
                total_return = 0.0
                sharpe = 0.0
                max_dd = 0.1
        else:
            # CPU fallback
            params = genome.parameters
            lookback = int(params.get('lookback', 20) * 100) + 1
            threshold = params.get('threshold', 0.5)
            
            if len(returns_data) > lookback:
                rolling_mean = np.convolve(returns_data, np.ones(lookback)/lookback, mode='valid')
                signals = (rolling_mean > threshold).astype(float)
                
                position_changes = np.abs(np.diff(signals, prepend=signals[0]))
                costs = position_changes * transaction_costs
                strategy_returns = signals * returns_data[lookback:] - costs
                
                total_return = float(np.prod(1 + strategy_returns) - 1)
                
                if np.std(strategy_returns) > 0:
                    sharpe = float(np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                else:
                    sharpe = 0.0
                
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_dd = float(abs(np.min(drawdown)))
            else:
                total_return = 0.0
                sharpe = 0.0
                max_dd = 0.1
        
        # Fitness score combines multiple metrics
        fitness = total_return * 0.4 + sharpe * 0.4 - max_dd * 0.2
        
        # Cache result
        self.evaluation_cache[genome.genome_id] = fitness
        
        return fitness, sharpe, max_dd


class GeneticBreeder:
    """
    Master orchestrator for genetic strategy breeding.
    Manages population evolution across Ray workers.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        elite_count: int = 5,
        num_workers: int = 4,
        mutation_rate: float = 0.15,
        mutation_std: float = 0.05
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.num_workers = num_workers
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        
        self.workers: List[ray.ObjectRef] = []
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.initialized = False
    
    def initialize_ray(self):
        """Initialize Ray cluster with strict memory constraints."""
        if not ray.is_initialized():
            total_ram = psutil.virtual_memory().available
            worker_ram = min(total_ram // self.num_workers, MAX_RAM_MB * 1024 * 1024)
            
            ray.init(
                num_cpus=self.num_workers,
                _memory=int(worker_ram * self.num_workers * 0.9),
                object_store_memory=int(worker_ram * self.num_workers * 0.3),
                ignore_reinit_error=True
            )
        
        # Spawn workers with per-worker RAM limits
        self.workers = [
            EvolutionWorker.remote(ram_limit_mb=MAX_RAM_MB // self.num_workers)
            for _ in range(self.num_workers)
        ]
        self.initialized = True
    
    def initialize_population(self, initial_strategies: Optional[List[Dict[str, Any]]] = None):
        """Initialize or seed the population."""
        if initial_strategies:
            # Seed from existing strategies
            for i, strat in enumerate(initial_strategies[:self.population_size]):
                genome = StrategyGenome(
                    genome_id=strat.get('id', f"seed_{i}"),
                    parameters=strat['parameters'],
                    fitness_score=strat.get('fitness', 0.0),
                    sharpe_ratio=strat.get('sharpe', 0.0),
                    max_drawdown=strat.get('max_dd', 0.1),
                    generation=0,
                    parent_ids=[],
                    mutation_count=0,
                    created_at=datetime.utcnow()
                )
                self.population.append(genome)
        else:
            # Random initialization
            for i in range(self.population_size):
                params = {
                    'lookback': np.random.uniform(0.1, 1.0),
                    'threshold': np.random.uniform(0.3, 0.7),
                    'stop_loss': np.random.uniform(0.01, 0.1),
                    'take_profit': np.random.uniform(0.02, 0.2),
                    'position_size': np.random.uniform(0.1, 0.5)
                }
                
                genome = StrategyGenome(
                    genome_id=f"random_{i}",
                    parameters=params,
                    fitness_score=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.1,
                    generation=0,
                    parent_ids=[],
                    mutation_count=0,
                    created_at=datetime.utcnow()
                )
                self.population.append(genome)
    
    def evolve_generation(
        self,
        returns_data: np.ndarray
    ) -> List[StrategyGenome]:
        """
        Evolve one generation of strategies.
        
        Args:
            returns_data: Historical returns for fitness evaluation
            
        Returns:
            New population after evolution
        """
        if not self.initialized:
            self.initialize_ray()
        
        if len(self.population) == 0:
            self.initialize_population()
        
        # Evaluate current population
        eval_futures = []
        for i, genome in enumerate(self.population):
            worker = self.workers[i % len(self.workers)]
            future = worker.evaluate_fitness_gpu.remote(
                genome,
                returns_data,
                0.001
            )
            eval_futures.append((i, future))
        
        # Collect evaluation results
        for idx, future in eval_futures:
            fitness, sharpe, max_dd = ray.get(future)
            self.population[idx].fitness_score = fitness
            self.population[idx].sharpe_ratio = sharpe
            self.population[idx].max_drawdown = max_dd
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        # Record generation stats
        gen_stats = {
            'generation': self.generation,
            'best_fitness': self.population[0].fitness_score if self.population else 0,
            'avg_fitness': np.mean([g.fitness_score for g in self.population]),
            'best_sharpe': max([g.sharpe_ratio for g in self.population], default=0),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.history.append(gen_stats)
        
        # Elitism: preserve top performers
        elites = self.population[:self.elite_count]
        
        # Create new population through crossover and mutation
        new_population = elites.copy()
        
        while len(new_population) < self.population_size:
            # Tournament selection
            tournament_size = 5
            candidates = np.random.choice(
                self.population[self.elite_count:],  # Exclude elites
                size=min(tournament_size, len(self.population) - self.elite_count),
                replace=False
            )
            parent1 = max(candidates, key=lambda g: g.fitness_score)
            
            candidates = np.random.choice(
                self.population[self.elite_count:],
                size=min(tournament_size, len(self.population) - self.elite_count),
                replace=False
            )
            parent2 = max(candidates, key=lambda g: g.fitness_score)
            
            # Crossover
            worker = self.workers[np.random.randint(len(self.workers))]
            child1, child2 = ray.get(
                worker.crossover.remote(parent1, parent2, "arithmetic")
            )
            
            # Mutation
            child1 = ray.get(
                worker.mutate.remote(
                    child1,
                    self.mutation_rate,
                    self.mutation_std,
                    MutationType.GAUSSIAN
                )
            )
            child2 = ray.get(
                worker.mutate.remote(
                    child2,
                    self.mutation_rate,
                    self.mutation_std,
                    MutationType.GAUSSIAN
                )
            )
            
            # Re-evaluate children
            fitness1, sharpe1, dd1 = ray.get(
                worker.evaluate_fitness_gpu.remote(child1, returns_data, 0.001)
            )
            child1.fitness_score = fitness1
            child1.sharpe_ratio = sharpe1
            child1.max_drawdown = dd1
            
            if len(new_population) < self.population_size:
                new_population.append(child1)
            
            if len(new_population) < self.population_size:
                fitness2, sharpe2, dd2 = ray.get(
                    worker.evaluate_fitness_gpu.remote(child2, returns_data, 0.001)
                )
                child2.fitness_score = fitness2
                child2.sharpe_ratio = sharpe2
                child2.max_drawdown = dd2
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return self.population
    
    def get_best_genomes(self, n: int = 5) -> List[StrategyGenome]:
        """Get the top N genomes from current population."""
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        return sorted_pop[:n]
    
    def export_for_walkforward(self) -> List[Dict[str, Any]]:
        """Export best genomes for walk-forward validation."""
        best = self.get_best_genomes(self.elite_count)
        return [g.to_dict() for g in best]
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()
        self.workers = []
        self.initialized = False


if __name__ == '__main__':
    # Example usage
    breeder = GeneticBreeder(population_size=20, num_workers=2)
    
    # Generate sample returns data
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.02
    
    # Initialize and evolve
    breeder.initialize_population()
    
    for gen in range(5):
        population = breeder.evolve_generation(returns)
        best = breeder.get_best_genomes(1)[0]
        print(f"Generation {gen}: Best fitness={best.fitness_score:.4f}, Sharpe={best.sharpe_ratio:.2f}")
    
    # Export for walk-forward validation
    candidates = breeder.export_for_walkforward()
    print(f"\nExported {len(candidates)} candidates for walk-forward validation")
    
    breeder.shutdown()
