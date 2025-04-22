import pygame
import numpy as np
from datetime import datetime
import os
from car_environment import CarEnvironment


class InteractiveCarEnvironment(CarEnvironment):
    """
    Interactive car environment with Pygame visualization
    """

    def __init__(self, street_width=6.5):
        # Initialize with increased max_steps
        super().__init__(street_width)
        self.max_steps = 500  # We need more steps for tighter street widths

        # Initialize Pygame with larger window
        if not pygame.get_init():
            pygame.init()
        self.scale = 70  # Increased scale for bigger visualization (pixels per meter)
        self.width = int(street_width * self.scale + 400)  # Extra space for info panel
        self.height = 900  # Taller window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"U-Turn Demo (Street Width: {street_width}m)")

        # Colors
        self.colors = {
            "background": (240, 240, 240),
            "street": (200, 200, 200),
            "car": (255, 100, 100),
            "trajectory": (100, 100, 255),
            "text": (0, 0, 0),
            "boundary": (180, 0, 0),
            "info_panel": (230, 230, 250),
        }

        # Fonts for text display
        self.font = pygame.font.SysFont("Arial", 18)
        self.title_font = pygame.font.SysFont("Arial", 22, bold=True)

        # Action tracking for demonstrations
        self.actions = []

        # Timing for physics updates
        self.clock = pygame.time.Clock()

    def reset(self, start_position=None):
        """Reset the environment and clear action history"""
        state = super().reset(start_position)
        self.actions = []
        return state

    def handle_input(self):
        """
        Get action from keyboard input
        """
        keys = pygame.key.get_pressed()

        # Default: no steering, no movement
        wheel_angle_idx = 9  # Middle of wheel_angles (0 degrees)
        speed_idx = 3  # Middle of speeds (0 km/h)

        # Steering control - map to discrete wheel angles
        # FIXED: LEFT key now turns LEFT (negative angle)
        if keys[pygame.K_LEFT]:
            wheel_angle_idx = 18  # -45 degrees (LEFT)
        elif keys[pygame.K_RIGHT]:
            wheel_angle_idx = 0  # +45 degrees (RIGHT)

        # For more granular control
        elif keys[pygame.K_a]:
            wheel_angle_idx = 14  # -25 degrees (slight LEFT)
        elif keys[pygame.K_d]:
            wheel_angle_idx = 4  # +25 degrees (slight RIGHT)

        # Speed control - map to discrete speeds
        if keys[pygame.K_UP]:
            speed_idx = 5  # +2 km/h (forward)
        elif keys[pygame.K_DOWN]:
            speed_idx = 1  # -2 km/h (backward)

        # For more granular control
        elif keys[pygame.K_w]:
            speed_idx = 4  # +1 km/h (slight forward)
        elif keys[pygame.K_s]:
            speed_idx = 2  # -1 km/h (slight backward)

        # Convert to single action index
        action = self.encode_action(wheel_angle_idx, speed_idx)

        return action

    def render(self):
        """
        Render the environment using Pygame with enhanced information display
        """
        # Clear screen
        self.screen.fill(self.colors["background"])

        # Calculate street area dimensions
        street_panel_width = int(self.street_width * self.scale)

        # Draw street
        pygame.draw.rect(
            self.screen,
            self.colors["street"],
            pygame.Rect(0, 0, street_panel_width, self.height),
        )

        # Draw street boundaries with thicker lines
        pygame.draw.line(
            self.screen,
            self.colors["boundary"],
            (0, 0),
            (0, self.height),
            4,  # Thicker line
        )
        pygame.draw.line(
            self.screen,
            self.colors["boundary"],
            (street_panel_width, 0),
            (street_panel_width, self.height),
            4,  # Thicker line
        )

        # Draw trajectory
        if len(self.trajectory) > 1:
            points = [
                (state[0] * self.scale, self.height - state[1] * self.scale)
                for state in self.trajectory
            ]
            pygame.draw.lines(
                self.screen,
                self.colors["trajectory"],
                False,
                points,
                3,  # Thicker line
            )

        # Draw car
        if self.state:
            # Get car corners
            corners = self._car_corners(self.state)

            # Convert to screen coordinates
            screen_corners = [
                (x * self.scale, self.height - y * self.scale) for x, y in corners
            ]

            # Draw car body as polygon
            pygame.draw.polygon(self.screen, self.colors["car"], screen_corners)
            pygame.draw.polygon(
                self.screen,
                (0, 0, 0),  # Black outline
                screen_corners,
                2,  # Line width
            )

            # Draw direction indicator (arrow from center in the direction of heading)
            x, y, gamma = self.state
            center = (x * self.scale, self.height - y * self.scale)
            direction_end = (
                center[0] + 30 * np.cos(gamma),
                center[1] - 30 * np.sin(gamma),
            )
            pygame.draw.line(self.screen, (0, 0, 0), center, direction_end, 3)

            # Draw arrow head
            arrow_head_length = 10
            arrow_head_width = 6
            angle = np.arctan2(
                center[1] - direction_end[1], direction_end[0] - center[0]
            )
            pygame.draw.polygon(
                self.screen,
                (0, 0, 0),
                [
                    direction_end,
                    (
                        direction_end[0]
                        - arrow_head_length * np.cos(angle)
                        - arrow_head_width * np.sin(angle),
                        direction_end[1]
                        + arrow_head_length * np.sin(angle)
                        - arrow_head_width * np.cos(angle),
                    ),
                    (
                        direction_end[0]
                        - arrow_head_length * np.cos(angle)
                        + arrow_head_width * np.sin(angle),
                        direction_end[1]
                        + arrow_head_length * np.sin(angle)
                        + arrow_head_width * np.cos(angle),
                    ),
                ],
            )

        # Information panel
        info_panel_x = street_panel_width + 10
        info_panel_width = self.width - street_panel_width - 10
        pygame.draw.rect(
            self.screen,
            self.colors["info_panel"],
            pygame.Rect(info_panel_x, 10, info_panel_width, self.height - 20),
        )

        # Display detailed information
        if self.state:
            x, y, gamma = self.state
            gamma_deg = (gamma * 180 / np.pi) % 360

            # Draw title
            title = self.title_font.render(
                "CAR AND ENVIRONMENT INFO", True, (0, 0, 100)
            )
            self.screen.blit(title, (info_panel_x + 10, 20))

            # Car position and orientation
            position_title = self.title_font.render("Car State:", True, (0, 0, 0))
            self.screen.blit(position_title, (info_panel_x + 10, 60))

            pos_info = [
                f"Position X: {x:.2f} m",
                f"Position Y: {y:.2f} m",
                f"Orientation: {gamma_deg:.1f}°",
            ]

            for i, info in enumerate(pos_info):
                text = self.font.render(info, True, self.colors["text"])
                self.screen.blit(text, (info_panel_x + 20, 90 + i * 25))

            # Car dimensions
            dim_title = self.title_font.render("Car Dimensions:", True, (0, 0, 0))
            self.screen.blit(dim_title, (info_panel_x + 10, 180))

            dim_info = [
                f"Width (b): {self.car_width:.2f} m",
                f"Front Length (l₁): {self.front_length:.2f} m",
                f"Body Length (l₂): {self.body_length:.2f} m",
                f"Rear Length (l₃): {self.rear_length:.2f} m",
                f"Total Length: {self.total_length:.2f} m",
            ]

            for i, info in enumerate(dim_info):
                text = self.font.render(info, True, self.colors["text"])
                self.screen.blit(text, (info_panel_x + 20, 210 + i * 25))

            # Environment information
            env_title = self.title_font.render("Environment:", True, (0, 0, 0))
            self.screen.blit(env_title, (info_panel_x + 10, 370))

            env_info = [
                f"Street Width: {self.street_width:.2f} m",
                f"Current Step: {self.current_steps}/{self.max_steps}",
                f"Total Trajectory Points: {len(self.trajectory)}",
            ]

            for i, info in enumerate(env_info):
                text = self.font.render(info, True, self.colors["text"])
                self.screen.blit(text, (info_panel_x + 20, 400 + i * 25))

            # Controls information
            controls_title = self.title_font.render("Controls:", True, (0, 0, 0))
            self.screen.blit(controls_title, (info_panel_x + 10, 500))

            controls_info = [
                "Arrow Keys: Move/Steer",
                "A/D: Slight Left/Right",
                "W/S: Slight Forward/Backward",
                "SPACE: Start/Stop recording",
                "R: Reset",
                "S: Save demonstration",
                "ESC: Quit",
            ]

            for i, info in enumerate(controls_info):
                text = self.font.render(info, True, self.colors["text"])
                self.screen.blit(text, (info_panel_x + 20, 530 + i * 25))

            # Status information
            status_title = self.title_font.render("Status:", True, (0, 0, 0))
            self.screen.blit(status_title, (info_panel_x + 10, 720))

            # Check if U-turn is complete
            uturn_complete = "Yes" if self._is_terminal(self.state) else "No"
            valid_state = "Yes" if self._is_valid_state(self.state) else "No"

            status_info = [
                f"U-Turn Complete: {uturn_complete}",
                f"Valid State: {valid_state}",
            ]

            for i, info in enumerate(status_info):
                color = (0, 128, 0) if "Yes" in info else (128, 0, 0)
                text = self.font.render(info, True, color)
                self.screen.blit(text, (info_panel_x + 20, 750 + i * 25))

        # Update display
        pygame.display.flip()

    def save_demonstration(self):
        """
        Save the current trajectory and actions to a text file
        """
        # Create demos directory if it doesn't exist
        os.makedirs("demos", exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demos/demo_width_{self.street_width}m_{timestamp}.txt"

        # Check if trajectory is complete
        is_successful = False
        if len(self.trajectory) > 1 and self.trajectory[-1] != self.trajectory[0]:
            is_successful = self._is_terminal(self.trajectory[-1])

        if not is_successful:
            x, y, gamma = self.trajectory[-1]
            initial_x, initial_y, initial_gamma = self.trajectory[0]

            # Calculate direction check
            target_gamma = (initial_gamma + np.pi) % (2 * np.pi)
            gamma_diff = min(
                abs(gamma - target_gamma), 2 * np.pi - abs(gamma - target_gamma)
            )
            angle_in_degrees = gamma_diff * 180 / np.pi

            # Calculate progress check
            dx = x - initial_x
            dy = y - initial_y
            init_dx = np.cos(initial_gamma)
            init_dy = np.sin(initial_gamma)
            progress = -(init_dx * dx + init_dy * dy)

            print("\nDemonstration not successful. Debug info:")
            print(
                f"Final angle difference: {angle_in_degrees:.2f}° (needs to be < {30:.2f}°)"
            )
            print(
                f"Progress in opposite direction: {progress:.2f}m (needs to be > {self.total_length / 2:.2f}m)"
            )
            print(f"Initial heading: {initial_gamma * 180 / np.pi:.2f}°")
            print(f"Final heading: {gamma * 180 / np.pi:.2f}°")
            print(f"Target heading: {target_gamma * 180 / np.pi:.2f}°")

        # Save to text file
        with open(filename, "w") as f:
            # Write header information
            f.write("# U-Turn Demonstration\n")
            f.write(f"# Street Width: {self.street_width}m\n")
            f.write(f"# Total Steps: {len(self.actions)}\n")
            f.write(f"# Success: {'Yes' if is_successful else 'No'}\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write initial state
            x0, y0, gamma0 = self.trajectory[0]
            f.write(f"# Initial State: {x0:.6f}, {y0:.6f}, {gamma0:.6f}\n\n")

            # Write column headers
            f.write("# step, x, y, gamma, action\n")

            # Write trajectory and actions
            for i, (state, action) in enumerate(
                zip(self.trajectory, self.actions + [0])
            ):
                if i < len(self.actions):  # Skip the last state which has no action
                    x, y, gamma = state
                    f.write(f"{i}, {x:.6f}, {y:.6f}, {gamma:.6f}, {action}\n")

        self.visualize(show=True)

        return filename, is_successful

    def run_demonstration(self):
        """
        Run the interactive demonstration
        """
        running = True
        recording = False

        current_position_index = 0
        starting_positions = self.get_starting_positions()

        print("\n===== Interactive U-Turn Demonstration =====")
        print(f"Street Width: {self.street_width}m")
        print("\nControls:")
        print("  Arrow Keys: Steer (Left/Right) and Move (Up/Down)")
        print("  A/D: Slight Left/Right")
        print("  W/S: Slight Forward/Backward")
        print("  SPACE: Start/Stop recording")
        print("  R: Reset the environment")
        print("  P: Change starting position")
        print("  S: Save current demonstration")
        print("  ESC: Quit")
        print("\nPress SPACE to start recording your demonstration!")

        # Reset environment with first starting position
        self.reset(starting_positions[current_position_index])

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        recording = not recording
                        if recording:
                            print("\nRecording started...")
                        else:
                            print("Recording paused.")
                    elif event.key == pygame.K_r:
                        self.reset(starting_positions[current_position_index])
                        recording = False
                        print("\nEnvironment reset.")
                    # Add this block to handle position cycling
                    elif event.key == pygame.K_p:
                        current_position_index = (current_position_index + 1) % len(
                            starting_positions
                        )
                        self.reset(starting_positions[current_position_index])
                        recording = False
                        print(
                            f"\nChanged to starting position {current_position_index + 1}"
                        )
                    elif event.key == pygame.K_s and len(self.trajectory) > 1:
                        filename, success = self.save_demonstration()
                        if success:
                            print(f"\nSuccessful demonstration saved to {filename}")
                        else:
                            print(f"\nIncomplete demonstration saved to {filename}")
                            print("(The U-turn was not completed successfully)")
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Get action from keyboard
            action = self.handle_input()

            # Update state if recording
            if recording:
                next_state, reward, done = self.step(action)
                self.actions.append(action)

                if done:
                    if self._is_terminal(next_state):
                        print("\nU-turn completed successfully!")
                        print("\ntotal reward:", reward)
                    else:
                        print("\nCar left the street or reached max steps.")
                    recording = False

            # Render current state
            self.render()
            self.clock.tick(10)  # 10 FPS (matches the 0.1s time step)

        pygame.quit()
        print("\nInteractive demonstration closed.")


# Function to launch interactive demonstration
def create_demonstrations(street_width=6.5):
    """
    Create interactive demonstrations for a given street width
    """
    env = InteractiveCarEnvironment(street_width=street_width)
    env.run_demonstration()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create interactive U-Turn demonstrations"
    )
    parser.add_argument(
        "--width", type=float, default=6.5, help="Street width in meters"
    )
    args = parser.parse_args()

    create_demonstrations(street_width=args.width)
