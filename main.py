import os
import argparse
import pickle
import time
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from interactive_demo import InteractiveCarEnvironment
from car_environment import CarEnvironment
from interactive_demo import create_demonstrations
from monte_carlo import (
    monte_carlo_with_demonstrations,
    evaluate_policy,
    plot_learning_curves,
    replay_policy,
    # visualize_trajectory,
    visualize_detailed_trajectory,
    visualize_demonstration,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="U-Turn with demonstrations and RL")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=[
            "demo",
            "train",
            "train_with_demos",
            "evaluate",
            "visualize",
            "visualize_detailed",
            "compare",
            "improved_train",
            "analyze_demos",
        ],
        help="Mode to run",
    )
    parser.add_argument(
        "--width", type=float, default=6.5, help="Street width in meters"
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--demo_bias",
        type=float,
        default=0.7,
        help="Initial bias toward demonstrated actions",
    )
    parser.add_argument(
        "--demo_decay",
        type=float,
        default=0.99,
        help="Decay rate for demonstration bias",
    )
    parser.add_argument(
        "--stage_size", type=int, default=5000, help="Number of episodes per stage"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize training in real-time"
    )
    return parser.parse_args()


def train_with_improved_approach(street_width=6.5, episodes=15000):
    """
    Complete training approach with all improvements
    """
    print(f"Starting improved U-turn training for street width {street_width}m")

    # First, analyze demonstrations
    demo_files = glob.glob(f"demos/demo_width_{street_width}m_*.txt")
    print(f"Found {len(demo_files)} demonstrations")

    for demo_file in demo_files:
        visualize_demonstration(demo_file)

    # Create environment
    env = CarEnvironment(street_width=street_width)

    # Train with improved parameters
    Q, policy, metrics = monte_carlo_with_demonstrations(
        env,
        demo_files=demo_files,
        num_episodes=episodes,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_stages=[0.5, 0.3, 0.2, 0.1, 0.05],
        stage_size=2500,
        demo_bias_start=0.98,
        demo_bias_decay=0.99995,
        learning_rate=0.2,
        visualize_training=True,
    )

    # Save the policy
    with open(f"results/improved_policy_{street_width}m.pkl", "wb") as f:
        pickle.dump(dict(policy), f)

    # Evaluate the policy
    print("\nEvaluating the learned policy:")
    for i, start_pos in enumerate(env.get_starting_positions()):
        print(f"\nStarting position {i + 1}:")
        visualize_detailed_trajectory(
            env, policy, start_pos, save_path=f"results/final_trajectory_pos{i + 1}.png"
        )

    return Q, policy, metrics


def analyze_demonstration_rewards(demo_files, env):
    """Analyze rewards from demonstration files"""
    print("\n===== Demonstration Reward Analysis =====")

    total_demos = 0
    successful_demos = 0
    all_rewards = []
    successful_rewards = []

    for demo_file in demo_files:
        success = False
        states = []
        actions = []

        # Read demonstration file
        with open(demo_file, "r") as f:
            for line in f:
                if "Success: Yes" in line:
                    success = True
                elif not line.startswith("#") and "," in line:
                    parts = line.strip().split(",")
                    if len(parts) >= 5:
                        x = float(parts[1].strip())
                        y = float(parts[2].strip())
                        gamma = float(parts[3].strip())
                        action = int(parts[4].strip())

                        states.append((x, y, gamma))
                        actions.append(action)

        # Calculate rewards for each step
        if len(states) > 0 and len(actions) > 0:
            total_demos += 1
            if success:
                successful_demos += 1

            # Initialize environment to first state
            env.reset(states[0])

            # Simulate the demonstration and track rewards
            demo_rewards = []
            total_reward = 0

            for i in range(len(actions)):
                # Set the environment to the demonstration state
                if i > 0:
                    env.state = states[i - 1]

                # Take the action and get reward
                _, reward, _ = env.step(actions[i])
                demo_rewards.append(reward)
                total_reward += reward

            all_rewards.append(total_reward)
            if success:
                successful_rewards.append(total_reward)

            # Print summary for this demonstration
            print(f"\nDemo: {demo_file}")
            print(f"Success: {success}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Average reward per step: {total_reward / len(actions):.2f}")
            print(
                f"Min/Max step rewards: {min(demo_rewards):.2f}/{max(demo_rewards):.2f}"
            )

            # Visualize the demonstration
            env.reset(states[0])
            for action in actions:
                env.step(action)
            env.visualize(
                show=True, save_path=f"demo_analysis_{os.path.basename(demo_file)}.png"
            )

    if total_demos > 0:
        print("\n===== Overall Demonstration Statistics =====")
        print(f"Total demonstrations: {total_demos}")
        print(f"Successful demonstrations: {successful_demos}")
        print(f"Average total reward (all): {np.mean(all_rewards):.2f}")
        if successful_demos > 0:
            print(
                f"Average total reward (successful only): {np.mean(successful_rewards):.2f}"
            )

    return all_rewards


def main():
    """Main function to run the program"""
    args = parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    if args.mode == "demo":
        # Interactive mode to create demonstrations
        print(f"Creating demonstrations for street width {args.width}m")
        create_demonstrations(street_width=args.width)

    elif args.mode == "train":
        # Standard Monte Carlo training without demonstrations
        if args.visualize:
            env = InteractiveCarEnvironment(street_width=args.width)
        else:
            env = CarEnvironment(street_width=args.width)

        print(f"\nTraining without demonstrations for street width {args.width}m")

        start_time = time.time()
        _, policy, metrics = monte_carlo_with_demonstrations(
            env,
            demo_files=None,  # No demonstrations
            num_episodes=args.episodes,
            gamma=args.gamma,
            visualize_training=True,
        )
        training_time = time.time() - start_time

        # Save results
        with open(f"results/policy_standard_{args.width}m.pkl", "wb") as f:
            pickle.dump(dict(policy), f)

        # Plot learning curves
        plot_learning_curves(
            metrics,
            title=f"Standard Monte Carlo (Width {args.width}m)",
            save_path=f"results/learning_standard_{args.width}m.png",
        )

        # Evaluate policy
        eval_metrics = evaluate_policy(env, policy, n_episodes=10)
        print("\nEvaluation results:")
        print(f"Success rate: {eval_metrics['success_rate']:.2%}")
        print(f"Average reward: {eval_metrics['avg_reward']:.2f}")
        print(f"Average steps: {eval_metrics['avg_steps']:.1f}")
        print(f"Training time: {training_time:.1f}s")

    elif args.mode == "train_with_demos":
        # Monte Carlo training with demonstrations
        if args.visualize:
            env = InteractiveCarEnvironment(street_width=args.width)
        else:
            env = CarEnvironment(street_width=args.width)

        # Find demo files for current street width
        demo_pattern = f"demos/demo_width_{args.width}m_*.txt"
        demo_files = glob.glob(demo_pattern)

        if not demo_files:
            print(f"No demonstrations found for width {args.width}m")
            print("Please create demonstrations first with --mode demo")
            return

        print(
            f"\nTraining with {len(demo_files)} demonstrations for street width {args.width}m"
        )

        start_time = time.time()
        _, policy, metrics = monte_carlo_with_demonstrations(
            env,
            demo_files=demo_files,
            num_episodes=args.episodes,
            gamma=args.gamma,
            epsilon_stages=[0.7, 0.5, 0.3, 0.2, 0.1, 0.05],
            stage_size=2000,
            demo_bias_start=args.demo_bias,
            demo_bias_decay=args.demo_decay,
            learning_rate=0.5,
            visualize_training=True,
        )
        training_time = time.time() - start_time

        # Save results
        with open(f"results/policy_with_demos_{args.width}m.pkl", "wb") as f:
            pickle.dump(dict(policy), f)

        # Plot learning curves
        plot_learning_curves(
            metrics,
            title=f"Monte Carlo with Demonstrations (Width {args.width}m)",
            save_path=f"results/learning_with_demos_{args.width}m.png",
        )

        # Evaluate policy
        eval_metrics = evaluate_policy(env, policy, n_episodes=10)
        print("\nEvaluation results:")
        print(f"Success rate: {eval_metrics['success_rate']:.2%}")
        print(f"Average reward: {eval_metrics['avg_reward']:.2f}")
        print(f"Average steps: {eval_metrics['avg_steps']:.1f}")
        print(f"Training time: {training_time:.1f}s")

    elif args.mode == "evaluate":
        # Evaluate saved policy
        env = CarEnvironment(street_width=args.width)

        # Try to load both policies
        policies = {}
        for policy_type in ["standard", "with_demos"]:
            policy_path = f"results/policy_{policy_type}_{args.width}m.pkl"
            if os.path.exists(policy_path):
                with open(policy_path, "rb") as f:
                    policy_dict = pickle.load(f)
                    policy = defaultdict(lambda: 0)
                    policy.update(policy_dict)
                    policies[policy_type] = policy
                    print(f"Loaded {policy_type} policy from {policy_path}")

        if not policies:
            print("No saved policies found. Train a policy first.")
            return

        # Evaluate each policy
        for policy_type, policy in policies.items():
            print(f"\nEvaluating {policy_type} policy:")
            eval_metrics = evaluate_policy(env, policy, n_episodes=10)
            print(f"Success rate: {eval_metrics['success_rate']:.2%}")
            print(f"Average reward: {eval_metrics['avg_reward']:.2f}")
            print(f"Average steps: {eval_metrics['avg_steps']:.1f}")

    elif args.mode == "visualize":
        # Visualize saved policy
        env = InteractiveCarEnvironment(street_width=args.width)

        # Try to load both policies
        policies = {}
        for policy_type in ["standard", "with_demos"]:
            policy_path = f"results/policy_{policy_type}_{args.width}m.pkl"
            if os.path.exists(policy_path):
                with open(policy_path, "rb") as f:
                    policy_dict = pickle.load(f)
                    policy = defaultdict(lambda: 0)
                    policy.update(policy_dict)
                    policies[policy_type] = policy
                    print(f"Loaded {policy_type} policy from {policy_path}")

        if not policies:
            print("No saved policies found. Train a policy first.")
            return

        # Visualize trajectories for each policy
        for policy_type, policy in policies.items():
            print(f"\nVisualizing {policy_type} policy:")
            for i, start_pos in enumerate(env.get_starting_positions()):
                print(f"\nStarting position {i + 1}:")
                # visualize_trajectory(env, policy, start_pos)
                replay_policy(env, policy, start_pos, delay=200)

    elif args.mode == "visualize_detailed":
        # Visualize a policy with detailed trajectory information
        env = CarEnvironment(street_width=args.width)

        # Load policy
        policy_path = f"results/policy_with_demos_{args.width}m.pkl"
        if os.path.exists(policy_path):
            with open(policy_path, "rb") as f:
                policy_dict = pickle.load(f)
                policy = defaultdict(lambda: 0)
                policy.update(policy_dict)
                print(f"Loaded policy from {policy_path}")
        else:
            print("No policy found. Train a policy first.")
            return

        # Visualize each starting position
        for i, start_pos in enumerate(env.get_starting_positions()):
            print(f"\nStarting position {i + 1}:")
            visualize_detailed_trajectory(
                env,
                policy,
                start_pos,
                save_path=f"results/detailed_trajectory_pos{i + 1}.png",
            )

    elif args.mode == "compare":
        # Compare standard and demonstration-guided learning
        env = CarEnvironment(street_width=args.width)

        # Find demo files
        demo_pattern = f"demos/demo_width_{args.width}m_*.txt"
        demo_files = glob.glob(demo_pattern)

        if not demo_files:
            print(f"No demonstrations found for width {args.width}m")
            print("Please create demonstrations first with --mode demo")
            return

        print(f"Found {len(demo_files)} demonstrations for width {args.width}m")

        # Run standard MC
        print("\nTraining with standard Monte Carlo...")
        _, _, standard_metrics = monte_carlo_with_demonstrations(
            env, demo_files=None, num_episodes=args.episodes, gamma=args.gamma
        )

        # Reset environment
        env = CarEnvironment(street_width=args.width)

        # Run demonstration-guided MC
        print("\nTraining with demonstration-guided Monte Carlo...")
        _, _, demo_metrics = monte_carlo_with_demonstrations(
            env,
            demo_files=demo_files,
            num_episodes=args.episodes,
            gamma=args.gamma,
            demo_bias_start=args.demo_bias,
            demo_bias_decay=args.demo_decay,
        )

        # Compare learning curves
        plt.figure(figsize=(15, 10))

        # Success rate comparison
        plt.subplot(1, 1, 1)
        plt.plot(standard_metrics["success_rate"], "b-", label="Standard MC")
        plt.plot(demo_metrics["success_rate"], "r-", label="Demo-guided MC")
        plt.title(f"Success Rate Comparison (Width {args.width}m)")
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"results/comparison_{args.width}m.png")
        plt.show()

    elif args.mode == "improved_train":
        # Run the improved training approach
        train_with_improved_approach(street_width=args.width, episodes=args.episodes)

    elif args.mode == "analyze_demos":
        # Analyze demonstration rewards
        env = CarEnvironment(street_width=args.width)
        demo_pattern = f"demos/demo_width_{args.width}m_*.txt"
        demo_files = glob.glob(demo_pattern)
        analyze_demonstration_rewards(demo_files, env)


if __name__ == "__main__":
    main()
