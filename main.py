"""
Main Command-Line Interface (CLI) Entry Point.

Unified runner for interactive simulation, single-flight evaluation,
parallel Monte Carlo benchmarks, publication figure rendering, and test execution.
"""

import sys
import os
import argparse
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


def run_interactive_simulation(args):
    """Launches the real-time Pygame simulator with live telemetry HUD."""
    source = args.flight or "flight_dji0003_13_5"
    cmd = [sys.executable, os.path.join(ROOT_DIR, "simulation", "visualizer.py"), "--source", source]
    subprocess.run(cmd)


def run_evaluation(args):
    """Executes headless comparative evaluation of Model 1 vs Model 2 on a single flight."""
    source = args.flight or "flight_dji0003_13_5"
    frames = str(args.frames or 500)
    profile = os.path.join(ROOT_DIR, "calibration", "profiles", "universal_profile.json")
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "experiments", "evaluate_flight.py"),
        "--source", source,
        "--frames", frames,
        "--profile", profile
    ]
    subprocess.run(cmd)


def run_benchmark(args):
    """Runs the parallel multi-core Monte Carlo benchmark campaign across all flights."""
    seeds = str(args.seeds or 10)
    workers = str(args.workers or 16)
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "experiments", "run_monte_carlo.py"),
        "--seeds", seeds,
        "--workers", workers
    ]
    subprocess.run(cmd)


def run_figures(args):
    """Generates all publication figures in results/figures/."""
    from experiments.generate_figures import generate_all_figures
    generate_all_figures()
    print(f"Publication figures successfully written to {os.path.join(ROOT_DIR, 'results', 'figures')}")


def run_tests(args):
    """Runs the automated pytest test suite."""
    cmd = [sys.executable, "-m", "pytest", os.path.join(ROOT_DIR, "tests")]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Hetero-Traffic-Control: Empirical Traffic Modelling & Adaptive Signal Control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --sim --flight flight_dji0003_13_5
  python main.py --eval --flight flight_dji0003_13_5 --frames 500
  python main.py --benchmark --seeds 10 --workers 16
  python main.py --figures
  python main.py --test
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sim", action="store_true", help="Launch interactive Pygame visual simulator")
    group.add_argument("--eval", action="store_true", help="Run head-to-head dual model evaluation")
    group.add_argument("--benchmark", action="store_true", help="Run parallel Monte Carlo benchmark suite")
    group.add_argument("--figures", action="store_true", help="Generate all publication figures")
    group.add_argument("--test", action="store_true", help="Execute pytest test suite")

    parser.add_argument("--flight", type=str, default="flight_dji0003_13_5",
                        help="Drone flight dataset (e.g. flight_dji0003_13_5, flight_dji0007, flight_dji0005_21_8, flight_dji003_21_8)")
    parser.add_argument("--frames", type=int, default=500, help="Simulation duration in frames (default: 500)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds for benchmark (default: 10)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes for benchmark (default: 16)")

    args = parser.parse_args()

    if args.sim:
        run_interactive_simulation(args)
    elif args.eval:
        run_evaluation(args)
    elif args.benchmark:
        run_benchmark(args)
    elif args.figures:
        run_figures(args)
    elif args.test:
        run_tests(args)


if __name__ == "__main__":
    main()
