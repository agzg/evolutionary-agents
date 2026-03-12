"""
Entry point for the Multi-Agent POMDP Door Navigation Experiment.

Usage:
    python run_experiment.py                    # Run with defaults
    python run_experiment.py --trials 20        # 20 trials
    python run_experiment.py --analyze results/ # Analyze existing results
    python run_experiment.py --dry-run          # Test without LLM calls

Environment:
    Set ANTHROPIC_API_KEY in .env file or environment.
"""
import argparse
import os
import sys
import time

from dotenv import load_dotenv

from src.config import ExperimentConfig
from src.simulator import Simulator, save_results
from src.analysis import run_full_analysis


def create_llm(config: ExperimentConfig, dry_run: bool = False):
    """Initialize the Anthropic LLM client."""
    if dry_run:
        return _create_mock_llm()

    try:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        # Quick test
        test = client.messages.create(
            model=config.model_name,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        print(f"LLM initialized: {config.model_name}")
        print(f"Test response: {test.content[0].text[:50]}")
        return client
    except ImportError:
        print("anthropic not installed. Install with:")
        print("  pip install anthropic python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        print("Check your ANTHROPIC_API_KEY in .env")
        sys.exit(1)


def _create_mock_llm():
    """Mock LLM for dry-run testing without API calls."""
    import random

    class MockMessages:
        def create(self, **kwargs):
            # Detect graph mode from system prompt
            system = kwargs.get('system', '')
            is_graph = 'node' in system and 'neighboring' in system
            if is_graph:
                action = random.choice(['node 0', 'node 1', 'node 2', 'node 3', 'stay'])
            else:
                action = random.choice(['north', 'south', 'east', 'west'])
            text = f"REASONING: Exploring systematically.\nACTION: {action}"
            class MockTextBlock:
                def __init__(self, t):
                    self.text = t
            class MockResponse:
                def __init__(self):
                    self.content = [MockTextBlock(text)]
            return MockResponse()

    class MockLLM:
        def __init__(self):
            self.messages = MockMessages()

    print("Using MOCK LLM (dry-run mode)")
    return MockLLM()


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent POMDP Door Experiment")
    parser.add_argument("--trials", type=int, default=None, help="Number of trials")
    parser.add_argument("--grid-size", type=int, default=None, help="Grid dimension (NxN)")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--analyze", type=str, default=None, help="Path to results dir to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Test with mock LLM")
    parser.add_argument("--mutation-rate", type=float, default=None, help="Prior mutation rate (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Analysis-only mode
    if args.analyze:
        results_path = os.path.join(args.analyze, "results.json")
        if not os.path.exists(results_path):
            print(f"No results.json found in {args.analyze}")
            sys.exit(1)
        run_full_analysis(results_path, args.analyze)
        return

    # Load env
    load_dotenv()

    # Build config
    config = ExperimentConfig()
    if args.trials:
        config.num_trials = args.trials
    if args.grid_size:
        config.grid_size = (args.grid_size, args.grid_size)
    if args.model:
        config.model_name = args.model
    if args.mutation_rate is not None:
        config.prior_mutation_rate = args.mutation_rate
    if args.seed is not None:
        config.random_seed = args.seed

    # Output dir
    exp_id = config.experiment_id()
    output_dir = args.output or os.path.join(config.results_dir, exp_id)
    os.makedirs(output_dir, exist_ok=True)

    print("Multi-Agent POMDP Door Navigation Experiment")
    if config.use_graph_world:
        print(f"Graph: {config.num_nodes} nodes, radius={config.connection_radius}, Doors: {config.num_doors}")
    else:
        print(f"Grid: {config.grid_size}, Doors: {config.num_doors}")
    print(f"Trials: {config.num_trials}, Model: {config.model_name}")
    print(f"Interactions/lifetime: {config.interactions_per_lifetime}")
    print(f"Mutation rate: {config.prior_mutation_rate}")
    print(f"Output: {output_dir}")

    # Initialize LLM
    llm = create_llm(config, dry_run=args.dry_run)

    # Run experiment
    start = time.time()
    sim = Simulator(config, llm)
    results = sim.run_all()
    elapsed = time.time() - start

    print(f"\nExperiment completed in {elapsed:.1f}s")

    # Save results
    save_results(results, config, output_dir)

    # Run analysis
    results_path = os.path.join(output_dir, "results.json")
    run_full_analysis(results_path, output_dir)

    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()