import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pygame
from interactive_demo import InteractiveCarEnvironment
from car_environment import CarEnvironment
import os


def discretize_state(state, env):
    """
    Improved state discretization focused on relevant features for U-turns
    """
    x, y, gamma = state

    x_bins = np.linspace(0, env.street_width, 30)
    y_bins = np.linspace(0, env.street_width * 3, 40)

    gamma_bins = np.linspace(0, 2 * np.pi, 72)

    # Discretize
    x_bin = np.digitize(x, x_bins)
    y_bin = np.digitize(y, y_bins)
    gamma_bin = np.digitize(gamma, gamma_bins)

    return (x_bin, y_bin, gamma_bin)


def load_demonstrations(demo_files):
    """
    Load demonstration files and return state-action pairs
    """
    demonstrations = []

    for demo_file in demo_files:
        try:
            demo_data = {"states": [], "actions": [], "street_width": 0}

            with open(demo_file, "r") as f:
                lines = f.readlines()

                # Parse header
                for line in lines[:10]:
                    if line.startswith("# Street Width:"):
                        demo_data["street_width"] = float(
                            line.split(":")[1].strip().rstrip("m")
                        )

                # Parse data
                for line in lines:
                    if not line.startswith("#") and "," in line:
                        parts = line.strip().split(",")
                        if len(parts) >= 5:
                            x = float(parts[1].strip())
                            y = float(parts[2].strip())
                            gamma = float(parts[3].strip())
                            action = int(parts[4].strip())

                            demo_data["states"].append((x, y, gamma))
                            demo_data["actions"].append(action)

            # Check if demonstration has data
            if len(demo_data["states"]) > 0 and len(demo_data["actions"]) > 0:
                demonstrations.append(demo_data)
                print(f"Loaded demonstration from {demo_file}")

        except Exception as e:
            print(f"Error loading {demo_file}: {e}")

    return demonstrations


def initialize_q_from_demonstrations(env, demonstrations, gamma=0.95, visualize=False):
    """
    Initialize Q-values from demonstrations with visualization support
    """
    # Number of actions
    n_actions = len(env.wheel_angles) * len(env.speeds)

    # Initialize Q-values and returns
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns = defaultdict(lambda: defaultdict(list))

    # Keep track of progress for visualization
    total_states = sum(len(demo["states"]) for demo in demonstrations)
    processed_states = 0
    update_interval = max(1, total_states // 100)  # Update display every 1% progress

    print(f"Processing {total_states} demonstration states...")

    # Process each demonstration
    for _, demo in enumerate(demonstrations):
        if abs(demo["street_width"] - env.street_width) > 0.1:
            print(
                f"Warning: Demonstration street width ({demo['street_width']}m) "
                + f"differs from environment ({env.street_width}m)"
            )

        # For each state-action pair in the demonstration
        for i in range(len(demo["actions"])):
            # Get state and action
            state = demo["states"][i]
            action = demo["actions"][i]

            # For visualization, set the environment state to match the demonstration
            if visualize and isinstance(env, InteractiveCarEnvironment):
                env.reset(state)

                # Update display every update_interval steps
                if processed_states % update_interval == 0:
                    # Process Pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return Q, returns

                    # Display the current state being processed
                    env.render()
                    pygame.display.flip()

                    # Show progress
                    progress = (processed_states / total_states) * 100
                    print(f"Initialization progress: {progress:.1f}%", end="\r")

                    # Small delay to allow display update
                    pygame.time.delay(1)  # Very short delay

            # Discretize state
            disc_state = discretize_state(state, env)

            # Calculate return from this point in the demonstration
            G = 0
            for t in range(i, len(demo["actions"])):
                # We need to calculate rewards for this demonstration
                # Reset the environment to the demonstration state
                temp_env = type(env)(street_width=env.street_width)
                temp_env.reset(demo["states"][i if t == i else t - 1])

                # Take the action and get reward
                _, reward, _ = temp_env.step(demo["actions"][t])

                # Accumulate discounted reward
                G += (gamma ** (t - i)) * reward

            # Update returns and Q-values
            returns[disc_state][action].append(G)
            Q[disc_state][action] = np.mean(returns[disc_state][action])

            processed_states += 1

    # Print a newline after progress updates
    if visualize:
        print()

    # Count how many state-action pairs were initialized
    count = sum(1 for s in Q for a in range(n_actions) if Q[s][a] != 0)
    print(f"Initialized {count} state-action pairs from demonstrations")

    return Q, returns


def monte_carlo_with_demonstrations(
    env,
    demo_files=None,
    num_episodes=10000,
    gamma=0.99,
    epsilon_start=0.5,
    epsilon_stages=[0.3, 0.2, 0.1, 0.05],  # Values for each 2000-episode stage
    stage_size=2500,  # Episodes per stage
    demo_bias_start=0.95,
    demo_bias_decay=0.9995,
    learning_rate=0.2,
    visualize_training=False,
):
    """
    Monte Carlo control with demonstration initialization and staged epsilon decay
    """
    print(f"Environment type: {type(env)}")
    print(f"Visualization flag: {visualize_training}")

    # Number of actions
    n_actions = len(env.wheel_angles) * len(env.speeds)

    # Load demonstrations if provided
    demonstrations = []
    if demo_files:
        demonstrations = load_demonstrations(demo_files)

    # Initialize Q-values and returns from demonstrations
    if demonstrations:
        print("Starting Q-value initialization...")
        Q, returns = initialize_q_from_demonstrations(
            env, demonstrations, gamma, visualize_training
        )
        print("Q-value initialization complete.")
    else:
        # Default initialization
        Q = defaultdict(lambda: np.zeros(n_actions))
        returns = defaultdict(lambda: defaultdict(list))

    # Keep track of demonstrated state-actions for biased exploration
    demo_actions = {}
    for demo in demonstrations:
        for i in range(len(demo["actions"])):
            state = discretize_state(demo["states"][i], env)
            action = demo["actions"][i]
            if state not in demo_actions:
                demo_actions[state] = []
            if action not in demo_actions[state]:
                demo_actions[state].append(action)

    # Initialize policy
    policy = defaultdict(lambda: 0)
    for state in Q:
        policy[state] = np.argmax(Q[state])

    # Initialize epsilon with the start value
    epsilon = epsilon_start
    demo_bias = demo_bias_start  # Probability of selecting demonstrated actions

    # Track metrics
    all_rewards = []
    all_steps = []
    success_rate = []
    success_window = []

    start_time = time.time()

    for episode in range(num_episodes):
        # Determine the current epsilon based on the episode stage
        current_stage = min(episode // stage_size, len(epsilon_stages) - 1)
        epsilon = epsilon_stages[current_stage]

        # Log when we change stages
        if episode % stage_size == 0 and episode > 0:
            print(
                f"\n===== Changing to stage {current_stage + 1}, epsilon={epsilon} =====\n"
            )

        # Choose a random starting position
        start_position = random.choice(env.get_starting_positions())
        state = env.reset(start_position)

        # Generate an episode
        episode_states = []
        episode_actions = []
        episode_rewards = []

        done = False

        while not done:
            # Discretize state
            disc_state = discretize_state(state, env)

            # Action selection with demonstration bias
            if disc_state in demo_actions and random.random() < demo_bias:
                # Select from demonstrated actions
                action = random.choice(demo_actions[disc_state])
            else:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, n_actions - 1)
                else:
                    action = np.argmax(Q[disc_state])

            # Take step
            next_state, reward, done = env.step(action)

            # Store state, action, reward
            episode_states.append(disc_state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # Update state
            state = next_state

        # Track metrics
        total_reward = sum(episode_rewards)
        all_rewards.append(total_reward)
        all_steps.append(len(episode_rewards))

        # Check if U-turn was successful
        success = env._is_terminal(state) and env._is_valid_state(state)
        success_window.append(int(success))

        if len(success_window) > 100:
            success_window.pop(0)

        success_rate.append(sum(success_window) / len(success_window))

        # Calculate returns and update Q-values using first-visit MC
        G = 0
        for t in range(len(episode_rewards) - 1, -1, -1):
            G = gamma * G + episode_rewards[t]
            state = episode_states[t]
            action = episode_actions[t]

            # Check if first occurrence
            if not any(
                (state == s and action == a)
                for s, a in zip(episode_states[:t], episode_actions[:t])
            ):
                returns[state][action].append(G)

                old_value = Q[state][action]
                new_value = np.mean(returns[state][action])
                Q[state][action] = old_value + learning_rate * (new_value - old_value)

                # Update policy
                policy[state] = np.argmax(Q[state])

        # Decay demonstration bias
        demo_bias *= demo_bias_decay

        # Print progress
        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(all_rewards[-100:])
            mean_steps = np.mean(all_steps[-100:])
            current_success_rate = sum(success_window) / len(success_window)
            elapsed_time = time.time() - start_time

            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward: {mean_reward:.2f}, "
                f"Avg Steps: {mean_steps:.1f}, "
                f"Success Rate: {current_success_rate:.2%}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Demo Bias: {demo_bias:.3f}, "
                f"Time: {elapsed_time:.1f}s"
            )

    metrics = {
        "rewards": all_rewards,
        "steps": all_steps,
        "success_rate": success_rate,
        "training_time": time.time() - start_time,
    }

    return Q, policy, metrics


def evaluate_policy(env, policy, n_episodes=10):
    """
    Evaluate a policy over multiple episodes and calculate success metrics
    """
    total_rewards = []
    total_steps = []
    successes = 0

    for _ in range(n_episodes):
        # Use all starting positions
        starting_positions = env.get_starting_positions()
        for start_pos in starting_positions:
            state = env.reset(start_pos)

            episode_reward = 0
            done = False

            while not done:
                # Discretize state
                disc_state = discretize_state(state, env)

                # Get action from policy
                action = policy[disc_state]

                # Take step
                state, reward, done = env.step(action)
                episode_reward += reward

                if env.current_steps > 400:  # Safety
                    break

            total_rewards.append(episode_reward)
            total_steps.append(env.current_steps)

            # Check if U-turn was successful
            if env._is_terminal(state) and env._is_valid_state(state):
                successes += 1

    # Calculate metrics
    success_rate = successes / (n_episodes * len(starting_positions))
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }


def plot_learning_curves(metrics, title="Learning Curves", save_path=None):
    """
    Plot learning curves from training metrics
    """
    plt.figure(figsize=(15, 10))

    # Reward curve
    plt.subplot(3, 1, 1)
    plt.plot(metrics["rewards"], "b-", alpha=0.3)
    window_size = 100
    moving_avg = [
        np.mean(metrics["rewards"][max(0, i - window_size) : i + 1])
        for i in range(len(metrics["rewards"]))
    ]
    plt.plot(moving_avg, "r-")
    plt.title(f"{title} - Average Reward")
    plt.ylabel("Reward")
    plt.grid(True)

    # Steps curve
    plt.subplot(3, 1, 2)
    plt.plot(metrics["steps"], "g-", alpha=0.3)
    moving_avg = [
        np.mean(metrics["steps"][max(0, i - window_size) : i + 1])
        for i in range(len(metrics["steps"]))
    ]
    plt.plot(moving_avg, "r-")
    plt.title(f"{title} - Average Episode Length")
    plt.ylabel("Steps")
    plt.grid(True)

    # Success rate curve
    plt.subplot(3, 1, 3)
    plt.plot(metrics["success_rate"], "r-")
    plt.title(f"{title} - Success Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def visualize_trajectory(env, policy, start_position):
    """
    Visualize a trajectory following the given policy
    """
    state = env.reset(start_position)

    done = False
    total_reward = 0

    while not done:
        # Discretize state
        disc_state = discretize_state(state, env)

        # Get action from policy
        action = policy[disc_state]

        # Take step
        state, reward, done = env.step(action)
        total_reward += reward

        if env.current_steps > 500:  # Safety
            break

    # Visualize the trajectory
    env.visualize(show=True)

    # Print results
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps taken: {env.current_steps}")
    print(
        f"U-turn successful: {env._is_terminal(state) and env._is_valid_state(state)}"
    )

    return total_reward, env.current_steps


def visualize_detailed_trajectory(env, policy, start_position, save_path=None):
    """
    Detailed visualization of trajectory with action annotations
    """
    state = env.reset(start_position)

    # Track detailed information
    states = [state]
    actions = []
    rewards = []
    orientations = []

    done = False
    total_reward = 0

    while not done:
        # Discretize state
        disc_state = discretize_state(state, env)

        # Get action from policy
        action = policy[disc_state]
        wheel_angle, speed = env.decode_action(action)

        # Log detailed information
        orientations.append((
            env.state[2] * 180 / np.pi,  # Current orientation in degrees
            wheel_angle * 180 / np.pi,  # Wheel angle in degrees
            speed * 3.6,  # Speed in km/h
        ))

        # Take step
        next_state, reward, done = env.step(action)

        # Update tracking
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        state = next_state

        if env.current_steps > 500:
            break

    # Visualize the trajectory
    env.visualize(show=False)  # Prepare the visualization but don't show yet

    # Annotate key points
    plt.figure(figsize=(12, 10))

    # Plot trajectory
    xs = [state[0] for state in states]
    ys = [state[1] for state in states]
    plt.plot(xs, ys, "b-", linewidth=2, label="Trajectory")

    # Plot states at regular intervals
    interval = max(1, len(states) // 10)
    for i in range(0, len(states), interval):
        x, y, _ = states[i]
        plt.plot(x, y, "ro", markersize=5)
        if i < len(actions):
            wheel_angle, speed = env.decode_action(actions[i])
            plt.annotate(
                f"Step {i}\nAngle: {wheel_angle * 180 / np.pi:.0f}°\nSpeed: {speed * 3.6:.1f} km/h",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

    # Add street boundaries
    plt.plot([0, 0], [0, 20], "k--", label="Street boundary")
    plt.plot([env.street_width, env.street_width], [0, 20], "k--")

    # Add car positions
    # Initial car
    initial_corners = env._car_corners(states[0])
    initial_corners.append(initial_corners[0])  # Close the polygon
    initial_xs, initial_ys = zip(*initial_corners)
    plt.plot(initial_xs, initial_ys, "g-", linewidth=2, label="Initial position")

    # Final car
    final_corners = env._car_corners(states[-1])
    final_corners.append(final_corners[0])  # Close the polygon
    final_xs, final_ys = zip(*final_corners)
    plt.plot(final_xs, final_ys, "r-", linewidth=2, label="Final position")

    # Add information text
    info_text = (
        f"Total Steps: {len(actions)}\n"
        f"Total Reward: {total_reward:.2f}\n"
        f"Final Orientation: {states[-1][2] * 180 / np.pi:.1f}°\n"
        f"Target Orientation: {((states[0][2] + np.pi) % (2 * np.pi)) * 180 / np.pi:.1f}°\n"
        f"U-turn Successful: {env._is_terminal(states[-1])}"
    )
    plt.annotate(
        info_text,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=10,
    )

    plt.title(f"Detailed U-Turn Trajectory (Street Width: {env.street_width}m)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Also print detailed step information
    print("\nDetailed Step Information:")
    print("Step\tOrientation\tWheel Angle\tSpeed\tReward")
    for i in range(len(actions)):
        orient, wheel, spd = orientations[i]
        print(f"{i}\t{orient:.1f}°\t{wheel:.1f}°\t{spd:.1f} km/h\t{rewards[i]:.2f}")

    return total_reward, env.current_steps


def visualize_demonstration(demo_file):
    """
    Visualize a demonstration file to understand its structure
    """
    # Parse the demo file
    states = []
    actions = []
    with open(demo_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith("#") and "," in line:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    _ = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    gamma = float(parts[3].strip())
                    action = int(parts[4].strip())
                    states.append((x, y, gamma))
                    actions.append(action)

    # Create environment and simulate the demonstration
    env = CarEnvironment(street_width=6.5)
    env.reset(states[0])

    # Execute actions
    for action in actions:
        env.step(action)

    # Visualize the trajectory
    env.visualize(save_path=f"demo_visualization_{os.path.basename(demo_file)}.png")

    # Analyze the demonstration
    print(f"Demonstration: {demo_file}")
    print(f"Total steps: {len(actions)}")

    # Calculate turning metrics
    initial_gamma = states[0][2]
    final_gamma = states[-1][2]
    gamma_diff = min(
        abs(final_gamma - initial_gamma), 2 * np.pi - abs(final_gamma - initial_gamma)
    )
    print(f"Initial orientation: {initial_gamma * 180 / np.pi:.1f}°")
    print(f"Final orientation: {final_gamma * 180 / np.pi:.1f}°")
    print(f"Total turn: {gamma_diff * 180 / np.pi:.1f}°")

    # Calculate movement metrics
    initial_x, initial_y = states[0][0], states[0][1]
    final_x, final_y = states[-1][0], states[-1][1]
    distance = np.sqrt((final_x - initial_x) ** 2 + (final_y - initial_y) ** 2)
    print(f"Movement distance: {distance:.2f}m")

    max_y = max(state[1] for state in states)
    min_y = min(state[1] for state in states)
    print(f"Y-range: {min_y:.2f}m to {max_y:.2f}m (range: {max_y - min_y:.2f}m)")

    return states, actions


def replay_policy(env, policy, start_position=None, delay=100):
    """
    Replay a policy with controlled visualization speed
    """
    if not isinstance(env, InteractiveCarEnvironment):
        print("Error: Environment must be InteractiveCarEnvironment for visualization")
        return

    # Initialize environment
    state = env.reset(start_position)
    done = False
    total_reward = 0

    # Main replay loop
    while not done:
        # Display current state
        env.render()

        # Process window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Get action from policy
        disc_state = discretize_state(state, env)
        action = policy[disc_state]

        # Display action information
        wheel_angle, speed = env.decode_action(action)
        print(
            f"State: {state}, Action: wheel={wheel_angle * 180 / np.pi:.1f}°, speed={speed * 3.6:.1f}km/h"
        )

        # Take step
        state, reward, done = env.step(action)
        total_reward += reward

        # Controlled delay
        pygame.time.delay(delay)

    # Show final state longer
    env.render()
    pygame.time.delay(2000)
    print(f"Episode complete. Total reward: {total_reward:.2f}")
