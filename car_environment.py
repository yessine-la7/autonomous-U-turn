import numpy as np
import matplotlib.pyplot as plt


class CarEnvironment:
    """
    Car environment for U-Turn reinforcement learning problem
    """

    def __init__(self, street_width=6.5):
        # Car dimensions (in meters)
        self.car_width = 1.8  # b = 1.8m
        self.front_length = 0.8  # l1 = 0.8m
        self.body_length = 3.2  # l2 = 3.2m
        self.rear_length = 0.9  # l3 = 0.9m
        self.total_length = (
            self.front_length + self.body_length + self.rear_length
        )  # 4.9m

        # Street parameters
        self.street_width = street_width  # Width of the street in meters

        # State parameters (x, y, γ)
        self.state = None

        # Action parameters
        self.wheel_angles = (
            np.linspace(-45, 45, 19) * np.pi / 180
        )  # 19 angles from -45° to 45° (in radians)
        self.speeds = (
            np.array([-3, -2, -1, 0, 1, 2, 3]) / 3.6
        )  # 7 speeds from -3 to 3 km/h (converted to m/s)

        # Physics parameters
        self.dt = 0.1  # Time step in seconds
        self.wheelbase = self.body_length

        # Episode parameters
        self.max_steps = 400  # Maximum steps per episode
        self.current_steps = 0

        # Trajectory tracking
        self.trajectory = []

    def reset(self, start_position=None):
        """
        Reset the environment to a starting position
        """
        if start_position is None:
            # Default starting position: center of street, facing up
            x = self.street_width / 2
            y = self.front_length + self.body_length / 2
            gamma = np.pi / 2  # Facing up (positive y-axis)
            self.state = (x, y, gamma)
        else:
            self.state = start_position

        self.current_steps = 0
        self.trajectory = [self.state]
        return self.state

    def encode_action(self, wheel_idx, speed_idx):
        """
        Encode wheel angle index and speed index to a single action index
        """
        return wheel_idx * len(self.speeds) + speed_idx

    def decode_action(self, action):
        """
        Decode action index to wheel angle and speed
        """
        wheel_idx = action // len(self.speeds)
        speed_idx = action % len(self.speeds)
        return self.wheel_angles[wheel_idx], self.speeds[speed_idx]

    def step(self, action):
        """
        Take a step in the environment using the given action
        """
        wheel_angle, speed = self.decode_action(action)
        x, y, gamma = self.state

        # If speed is near zero, car doesn't move
        if abs(speed) < 1e-5:
            new_state = (x, y, gamma)
            reward = self._calculate_reward(new_state, action)
            done = self._is_terminal(new_state)
            self.current_steps += 1
            return new_state, reward, done

        # Calculate turning radius using bicycle model
        # For small angles, tan(wheel_angle) ≈ wheel_angle
        if abs(wheel_angle) < 1e-5:  # Straight line
            turning_radius = float("inf")
        else:
            turning_radius = self.wheelbase / np.tan(abs(wheel_angle))

        # Calculate change in heading angle
        if turning_radius != float("inf"):
            # Direction of turn depends on wheel angle sign and speed direction
            turn_direction = np.sign(wheel_angle) * np.sign(speed)
            delta_gamma = turn_direction * abs(speed) * self.dt / turning_radius
        else:
            delta_gamma = 0

        # Update heading
        new_gamma = (gamma + delta_gamma) % (2 * np.pi)

        # Update position
        new_x = x + speed * np.cos(gamma) * self.dt
        new_y = y + speed * np.sin(gamma) * self.dt

        new_state = (new_x, new_y, new_gamma)

        # Check if the car is within street boundaries
        is_valid = self._is_valid_state(new_state)

        # Calculate reward and check if terminal
        reward = self._calculate_reward(new_state, action, is_valid)
        done = (
            self._is_terminal(new_state)
            or not is_valid
            or self.current_steps >= self.max_steps
        )

        # Update state if valid
        if is_valid:
            self.state = new_state
            self.trajectory.append(new_state)
        else:
            # For invalid states, terminate episode
            done = True

        self.current_steps += 1

        return self.state, reward, done

    def _car_corners(self, state):
        """
        Calculate the four corners of the car given its state
        Returns corners in order: front-left, front-right, rear-right, rear-left
        """
        x, y, gamma = state

        # Half car width
        half_width = self.car_width / 2

        # Calculate corners relative to center point
        # Front-left corner
        fl_x = (self.front_length + self.body_length / 2) * np.cos(
            gamma
        ) - half_width * np.sin(gamma)
        fl_y = (self.front_length + self.body_length / 2) * np.sin(
            gamma
        ) + half_width * np.cos(gamma)

        # Front-right corner
        fr_x = (self.front_length + self.body_length / 2) * np.cos(
            gamma
        ) + half_width * np.sin(gamma)
        fr_y = (self.front_length + self.body_length / 2) * np.sin(
            gamma
        ) - half_width * np.cos(gamma)

        # Rear-right corner
        rr_x = -(self.rear_length + self.body_length / 2) * np.cos(
            gamma
        ) + half_width * np.sin(gamma)
        rr_y = -(self.rear_length + self.body_length / 2) * np.sin(
            gamma
        ) - half_width * np.cos(gamma)

        # Rear-left corner
        rl_x = -(self.rear_length + self.body_length / 2) * np.cos(
            gamma
        ) - half_width * np.sin(gamma)
        rl_y = -(self.rear_length + self.body_length / 2) * np.sin(
            gamma
        ) + half_width * np.cos(gamma)

        # Translate corners to absolute coordinates
        fl = (x + fl_x, y + fl_y)
        fr = (x + fr_x, y + fr_y)
        rr = (x + rr_x, y + rr_y)
        rl = (x + rl_x, y + rl_y)

        return [fl, fr, rr, rl]

    def _is_valid_state(self, state):
        """
        Check if all corners of the car are within street boundaries
        """
        corners = self._car_corners(state)

        # Check if any corner is outside street boundaries
        for corner_x, _ in corners:
            if corner_x < 0 or corner_x > self.street_width:
                return False

        return True

    def _calculate_reward(self, state, action, is_valid=True):
        """
        Reward function for U-turn learning.
        """
        if not is_valid:
            return -50  # Penalty for leaving the street

        x, y, gamma = state
        initial_x, initial_y, initial_gamma = self.trajectory[0]
        _, speed = self.decode_action(action)

        # Calculate target orientation (opposite of starting orientation)
        target_gamma = (initial_gamma + np.pi) % (2 * np.pi)
        gamma_diff = min(
            abs(gamma - target_gamma), 2 * np.pi - abs(gamma - target_gamma)
        )

        # === Primary Reward Components ===

        # 1. Orientation Progress - Reward getting closer to target orientation
        orientation_progress = 1 - (gamma_diff / np.pi)  # 0 to 1 scale
        orientation_reward = 10.0 * orientation_progress

        # 2. Position-based rewards - Different depending on street width
        # For narrower streets, reward more efficient use of space
        width_factor = max(0.5, min(1.0, self.street_width / 6.5))

        # Reward being on the opposite side of the street from where we started
        initial_side = 0 if initial_x < self.street_width / 2 else 1  # 0=left, 1=right
        target_side = 1 - initial_side  # Opposite side
        target_x_position = (
            target_side * self.street_width * 0.8
        )  # Target position on opposite side

        # Distance to target x-position, normalized
        x_progress = 1 - min(1.0, abs(x - target_x_position) / self.street_width)
        position_reward = 5.0 * x_progress * width_factor

        # 3. Y-position progress - Reward proper forward movement then return
        # Expect the car to move forward (higher y) then back down
        y_progress = 0.0
        if len(self.trajectory) > 5:
            max_y_so_far = max(state[1] for state in self.trajectory)
            if (
                max_y_so_far > initial_y + self.car_width
            ):  # If we've moved forward enough
                # Reward coming back down after moving up
                if y < max_y_so_far:
                    y_progress = min(1.0, (max_y_so_far - y) / (2 * self.car_width))
            else:
                # Initially reward moving forward
                y_progress = min(1.0, (y - initial_y) / (2 * self.car_width))
        y_reward = 2.0 * y_progress

        # 4. Action-based rewards - Encourage appropriate steering and speed
        # Reward appropriate speed (positive when moving away, negative when returning)
        speed_reward = 0.0
        if orientation_progress < 0.5:  # First half of the turn
            speed_reward = 0.5 * (speed > 0)  # Reward forward movement
        else:  # Second half of the turn
            # Potentially need to reverse to complete turn in tight spaces
            if self.street_width < 6.0:
                speed_reward = 0.5 * (
                    (speed < 0 and orientation_progress < 0.7)
                    or (speed > 0 and orientation_progress >= 0.7)
                )
            else:
                speed_reward = 0.5 * (speed > 0)  # Wider streets may not need reversing

        # 5. Safety margin - Penalty for getting too close to walls
        corners = self._car_corners(state)
        left_distances = [corner[0] for corner in corners]
        right_distances = [self.street_width - corner[0] for corner in corners]
        min_wall_distance = min(min(left_distances), min(right_distances))

        # Progressive penalty that increases as the car gets closer to walls
        safe_threshold = 0.3
        wall_proximity_penalty = 0
        if min_wall_distance < safe_threshold:
            wall_proximity_penalty = (
                -10 * ((safe_threshold - min_wall_distance) / safe_threshold) ** 2
            )

        # 6. Efficiency penalty - Encourage completing the task quickly
        step_penalty = -0.1

        # 7. Terminal state bonus - Large reward for completing the U-turn
        terminal_bonus = 0
        if self._is_terminal(state):
            terminal_bonus = 500.0

        # Combine all reward components
        total_reward = (
            orientation_reward
            + position_reward
            + y_reward
            + speed_reward
            + wall_proximity_penalty
            + step_penalty
            + terminal_bonus
        )

        return total_reward

    def _is_terminal(self, state):
        """
        Improved check for terminal state (U-turn completed)
        """
        x, y, gamma = state
        initial_x, initial_y, initial_gamma = self.trajectory[0]

        # 1. Car must be pointing in the opposite direction (±15°)
        target_gamma = (initial_gamma + np.pi) % (2 * np.pi)
        gamma_diff = min(
            abs(gamma - target_gamma), 2 * np.pi - abs(gamma - target_gamma)
        )
        correct_direction = gamma_diff < np.pi / 6  # Within ±15° of target

        # 2. Car must have moved a minimum distance
        distance_moved = np.sqrt((x - initial_x) ** 2 + (y - initial_y) ** 2)
        has_moved = distance_moved > self.total_length / 2

        # 3. Car must be on the opposite side of the street
        initial_side = initial_x < self.street_width / 2  # True if started on left side
        current_side = x < self.street_width / 2  # True if currently on left side
        on_opposite_side = initial_side != current_side

        # 4. Y-position indicator - the car should have moved up and then down
        # y_pattern_complete = False
        # if len(self.trajectory) > 10:
        #     max_y = max(state[1] for state in self.trajectory)
        #     y_pattern_complete = (max_y > initial_y + self.car_width / 2) and (
        #         y < max_y - self.car_width / 4
        #     )

        # All conditions must be met
        return correct_direction and has_moved and on_opposite_side

    def visualize(self, show=True, save_path=None):
        """
        Visualize the car trajectory
        """
        if not self.trajectory:
            return

        plt.figure(figsize=(10, 8))

        # Plot street boundaries
        plt.plot([0, 0], [0, 20], "k--", label="Street boundary")
        plt.plot([self.street_width, self.street_width], [0, 20], "k--")

        # Extract states
        xs = [state[0] for state in self.trajectory]
        ys = [state[1] for state in self.trajectory]

        # Plot trajectory
        plt.plot(xs, ys, "b-", label="Car trajectory")

        # Plot car at intervals
        step = max(1, len(self.trajectory) // 10)
        for i in range(0, len(self.trajectory), step):
            state = self.trajectory[i]
            corners = self._car_corners(state)
            corners.append(corners[0])  # Close the polygon
            xs, ys = zip(*corners)
            plt.plot(xs, ys, "r-")

        # Plot initial and final car positions
        if self.trajectory:
            # Initial car (green)
            corners = self._car_corners(self.trajectory[0])
            corners.append(corners[0])
            xs, ys = zip(*corners)
            plt.plot(xs, ys, "g-", linewidth=2, label="Initial position")

            # Final car (red)
            corners = self._car_corners(self.trajectory[-1])
            corners.append(corners[0])
            xs, ys = zip(*corners)
            plt.plot(xs, ys, "r-", linewidth=2, label="Final position")

        plt.title(f"Car U-Turn (Street Width: {self.street_width}m)")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def get_starting_positions(self):
        """
        Define four different starting positions for the car
        """
        w = self.street_width

        # Position 1: Bottom center, facing up
        pos1 = (w / 2, self.total_length / 2, np.pi / 2)

        # Position 2: Bottom left (0.2m from boundary), facing up
        pos2 = (self.car_width / 2 + 0.2, self.total_length / 2, np.pi / 2)

        # Position 3: Bottom right (0.2m from boundary), facing up
        pos3 = (w - self.car_width / 2 - 0.2, self.total_length / 2, np.pi / 2)

        # Position 4: Bottom center, slight angle to the right (5 degrees)
        pos4 = (w / 2, self.total_length / 2, np.pi / 2 + np.pi / 36)

        return [pos1, pos2, pos3, pos4]

# if __name__ == "__main__":
#     env = CarEnvironment(street_width=6.5)

#     num_episodes = 1  # Anzahl der Episoden
#     max_steps_per_episode = 100  # Maximale Schritte pro Episode

#     for episode in range(num_episodes):
#         print(f"\n=== Episode {episode + 1} ===")
#         state = env.reset()
#         print(f"Startzustand: {state}")

#         for step in range(max_steps_per_episode):
#             action = np.random.randint(0, len(env.wheel_angles) * len(env.speeds))  # Zufällige Aktion
#             new_state, reward, done = env.step(action)
#             print(f"Schritt {step + 1}: Aktion {action}, Neuer Zustand: {new_state}, Belohnung: {reward}, Fertig: {done}")

#             if done:
#                 print("Episode beendet.\n")
#                 break

#         env.visualize()

##############################################################################################

# if __name__ == "__main__":
#     env = CarEnvironment(street_width=6.5)

#     num_episodes = 1  # Anzahl der Episoden
#     max_steps_per_episode = 200  # Maximale Schritte pro Episode

#     # Definiere eine optimierte Aktionsfolge für einen U-Turn
#     forward_slow = env.encode_action(9, 4)  # Gerade lenken, langsame Vorwärtsfahrt
#     forward_fast = env.encode_action(9, 6)  # Gerade lenken, schnelle Vorwärtsfahrt
#     left_turn_moderate = env.encode_action(4, 5)  # Mäßig nach links lenken, mittlere Geschwindigkeit
#     left_turn_sharp = env.encode_action(0, 4)  # Stark nach links lenken, langsamer
#     right_turn_moderate = env.encode_action(14, 5)  # Mäßig nach rechts lenken
#     right_turn_sharp = env.encode_action(18, 4)  # Stark nach rechts lenken, langsamer
#     reverse_slow = env.encode_action(9, 1)  # Gerade lenken, langsame Rückwärtsfahrt

#     for episode in range(num_episodes):
#         print(f"\n=== Episode {episode + 1} ===")
#         state = env.reset()
#         print(f"Startzustand: {state}")

#         for step in range(max_steps_per_episode):
#             if step == 0:
#                 action = forward_slow  # Langsames Anfahren
#             elif step == 1:
#                 action = forward_fast  # Schnell vorwärts fahren
#             elif step == 2 or step == 3:
#                 action = left_turn_moderate  # Erst leicht nach links lenken
#             elif step == 4 or step == 5:
#                 action = left_turn_sharp  # Dann schärfer links lenken
#             elif step == 6:
#                 action = right_turn_moderate  # Korrektur nach rechts
#             elif step == 7:
#                 action = right_turn_sharp  # Falls nötig, noch stärker nach rechts
#             elif step == 8:
#                 action = forward_fast  # Falls möglich, nach vorne durchziehen
#             else:
#                 action = reverse_slow  # Nur falls notwendig, minimal rückwärts

#             new_state, reward, done = env.step(action)
#             print(f"Schritt {step + 1}: Aktion {action}, Neuer Zustand: {new_state}, Belohnung: {reward}, Fertig: {done}")

#             if done:
#                 print("Episode beendet.\n")
#                 break

#         env.visualize()
