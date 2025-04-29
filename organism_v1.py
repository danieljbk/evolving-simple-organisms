# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Evolving Simple Organisms
#   2017-Nov.
#
#   Modified for live plotting (Qt5Agg), starvation, and robust window handling
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division, print_function
from collections import defaultdict
import sys  # Used for checking python version if needed
import time  # Added for pausing
import operator
import traceback  # For printing errors

# --- Force Matplotlib backend BEFORE importing pyplot ---
import matplotlib

try:
    matplotlib.use("Qt5Agg")  # Use Qt5 backend for stability on macOS/conda
except ImportError:
    print("Qt5Agg backend not available. Please install PyQt5:")
    print("  conda install pyqt")
    print("  or: pip install PyQt5")
    sys.exit(1)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

import numpy as np
from math import atan2, cos, degrees, floor, radians, sin, sqrt, inf
from random import randint, random, sample, uniform

# --- Attempt to import original plotting functions, provide fallbacks ---
try:
    # Assumes plotting.py is in the same directory
    from plotting import plot_food as plot_food_original
    from plotting import plot_organism as plot_organism_original

    print("Using plotting functions from plotting.py")

    # Adapt the original plot_organism to handle 'is_alive'
    # NOTE: This requires modifying the original plotting.py or accepting this override
    def plot_organism_adapted(x, y, r, ax, is_alive):
        if is_alive:
            # Call the original function if alive
            plot_organism_original(x, y, r, ax)
        else:
            # Plot dead organisms differently
            circle = Circle(
                [x, y],
                0.04,
                edgecolor="grey",
                facecolor="lightgrey",
                zorder=7,
                alpha=0.7,
            )
            ax.add_artist(circle)

    # Use the original food plotter directly
    plot_food_external = plot_food_original
    # Use the adapted organism plotter
    plot_organism_external = plot_organism_adapted

except ImportError:
    print("Warning: plotting.py not found. Using basic internal plotting functions.")

    # Define basic fallback plotting functions
    def plot_food_external(x, y, ax):
        ax.plot(x, y, "bo", markersize=4, alpha=0.6)  # Basic blue dot for food

    def plot_organism_external(x, y, r, ax, is_alive):  # Include is_alive flag
        if is_alive:
            ax.plot(x, y, "go", markersize=8)  # Basic green dot
            line_length = 0.1
            end_x = x + line_length * cos(radians(r))
            end_y = y + line_length * sin(radians(r))
            ax.plot(
                [x, end_x], [y, end_y], "k-", linewidth=1
            )  # Basic black direction line
        else:
            ax.plot(
                x, y, "o", color="grey", markersize=6, alpha=0.5
            )  # Grey dot for dead


# --- CONSTANTS / SETTINGS -----------------------------------------------------+

settings = {}

# EVOLUTION SETTINGS
settings["pop_size"] = 50  # number of organisms
settings["food_num"] = 100  # number of food particles
settings["gens"] = 50  # number of generations
settings["elitism"] = 0.20  # elitism (fraction of top living performers)
settings["mutate"] = 0.10  # mutation rate

# SIMULATION SETTINGS
settings["gen_time"] = 100  # generation length         (seconds)
settings["dt"] = 0.04  # simulation time step      (dt) => 2500 steps/gen
settings["dr_max"] = 720  # max rotational speed      (degrees per second)
settings["v_max"] = 0.5  # max velocity              (units per second)
settings["dv_max"] = 0.25  # max acceleration (+/-)    (units per second^2)
settings["collision_radius"] = 0.075  # Radius for eating food

# --- NEW SURVIVAL SETTING ---
# Represents number of time steps without food before dying.
# Example: 250 steps = 10 seconds (250 * 0.04)
settings["starvation_threshold"] = 250  # Adjust for difficulty

# ARENA BOUNDARIES
settings["x_min"] = -2.0
settings["x_max"] = 2.0
settings["y_min"] = -2.0
settings["y_max"] = 2.0

# --- Plotting Settings ---
settings["plot"] = True  # LIVE plot the simulation?
settings["plot_interval"] = 10  # Plot every N time steps

# ORGANISM NEURAL NET SETTINGS
settings["inodes"] = 1  # Input: Normalized heading to nearest food [-1, 1]
settings["hnodes"] = 5  # Hidden nodes
settings["onodes"] = 2  # Output: [dV scale, dR scale], both [-1, 1]

# --- GLOBAL PLOT VARIABLES ----------------------------------------------------+
fig, ax = None, None
simulation_stopped_early = False  # Flag to track if user closed plot window

# --- HELPER FUNCTIONS ---------------------------------------------------------+


def dist(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calc_heading(org, food):
    """Calculate normalized heading from organism to food (-1 to 1)."""
    d_x = food.x - org.x
    d_y = food.y - org.y
    # Angle from org's positive x-axis to food, in degrees
    absolute_angle_to_food = degrees(atan2(d_y, d_x))
    # Relative angle difference, accounting for org's current orientation 'r'
    relative_angle = absolute_angle_to_food - org.r

    # Normalize angle to be within [-180, 180]
    while relative_angle <= -180:
        relative_angle += 360
    while relative_angle > 180:
        relative_angle -= 360

    # Normalize to [-1, 1] for the neural network input
    return relative_angle / 180.0


def setup_live_plot(settings):
    """Initializes the plot figure and axes for live plotting."""
    global fig, ax
    print("Setting up plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()  # Turn on interactive mode
    ax.set_aspect("equal", adjustable="box")
    # Set limits with padding
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])
    plt.show()  # Show the plot window immediately
    # Need a tiny pause for plot window to actually appear sometimes
    plt.pause(0.1)
    print("Plot setup complete.")


def plot_frame_live(settings, organisms, foods, gen, time_step):
    """Updates the existing plot figure for live visualization."""
    global fig, ax
    # If the main figure doesn't exist anymore, do nothing.
    if fig is None or ax is None or not plt.fignum_exists(fig.number):
        return

    ax.clear()  # Clear previous drawings

    # Re-apply limits and title every frame (clearing axes removes them)
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])
    ax.set_title(f"Generation: {gen} | Time Step: {time_step}")

    # Plot food particles
    for food in foods:
        plot_food_external(food.x, food.y, ax)

    # Plot organisms (distinguishing alive/dead)
    for organism in organisms:
        plot_organism_external(
            organism.x, organism.y, organism.r, ax, organism.is_alive
        )

    # Re-apply aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Draw and pause for GUI update
    plt.draw()
    plt.pause(0.001)  # Minimal pause crucial for interactivity


# --- EVOLUTION FUNCTION -------------------------------------------------------+


def evolve(settings, organisms_old, gen):
    """Evolves the population using only *living* organisms from organisms_old."""

    # --- Filter out dead organisms ---
    living_organisms = [org for org in organisms_old if org.is_alive]
    num_living = len(living_organisms)
    print(
        f"  [Gen {gen-1}] Survivors: {num_living}/{settings['pop_size']}"
    )  # End of previous gen results

    # --- Handle Extinction ---
    if num_living == 0:
        print(
            f"  EXTINCTION in Gen {gen-1}! Creating new random population for Gen {gen}."
        )
        organisms_new = [
            organism(settings, name=f"gen[{gen}]-org[new{i}]")
            for i in range(settings["pop_size"])
        ]
        stats = defaultdict(float, {"BEST": 0, "WORST": 0, "AVG": 0, "COUNT": 0})
        return organisms_new, stats

    # --- Calculate selection numbers based on living population ---
    # Elitism based on fraction of *living* population
    elitism_num_actual = int(floor(settings["elitism"] * num_living))
    elitism_num_actual = min(
        elitism_num_actual, num_living
    )  # Cannot exceed number living

    # Number of offspring needed to refill population to pop_size
    num_offspring_needed = settings["pop_size"] - elitism_num_actual

    # --- GET STATS FROM *LIVING* SURVIVORS ----------------+
    stats = defaultdict(float)
    stats["WORST"] = float("inf")
    for org in living_organisms:  # Iterate over living only
        fitness = org.fitness  # Fitness accumulated during the simulation just ended
        if fitness > stats["BEST"]:
            stats["BEST"] = fitness
        if fitness < stats["WORST"]:
            stats["WORST"] = fitness
        stats["SUM"] += fitness

    stats["COUNT"] = num_living  # Store count of living organisms
    stats["AVG"] = stats["SUM"] / num_living if num_living > 0 else 0.0
    if stats["WORST"] == float("inf"):
        stats["WORST"] = 0.0  # Handle case if all survivors had 0 fitness

    # --- ELITISM (Keep best *living* organisms' weights) ---
    orgs_sorted = sorted(
        living_organisms, key=operator.attrgetter("fitness"), reverse=True
    )
    organisms_new = []
    # Create new organism objects for the next generation, copying elite weights
    for i in range(elitism_num_actual):
        elite_org = orgs_sorted[i]
        # The organism() constructor will reset position/velocity/alive status
        organisms_new.append(
            organism(
                settings,
                wih=elite_org.wih.copy(),
                who=elite_org.who.copy(),
                name=elite_org.name,
            )
        )

    # --- GENERATE NEW ORGANISMS (Breed from elites) --------+
    # Use the newly created elite organisms as the parent pool
    parent_pool = organisms_new[:]

    # Fallback if elite pool is empty (e.g., elitism=0, but some survived)
    if not parent_pool and orgs_sorted:
        print("  Warning: Elite pool empty, using top survivors as parents.")
        # Use top N survivors directly as parents (take their weights)
        # Select up to, say, 5 potential parents or all survivors if fewer
        potential_parents_data = [
            (org.wih.copy(), org.who.copy())
            for org in orgs_sorted[: min(5, len(orgs_sorted))]
        ]
        if not potential_parents_data:  # Should not happen if num_living > 0
            print("  Error: Could not identify any parents.")
            parent_pool = []  # Will trigger random fill below

    # If still no valid parents, fill with random
    if not parent_pool and not potential_parents_data:
        print(
            "  Error: Cannot select parents. Filling generation with random organisms."
        )
        while len(organisms_new) < settings["pop_size"]:
            organisms_new.append(
                organism(
                    settings, name=f"gen[{gen}]-org[rand_fill{len(organisms_new)}]"
                )
            )

    else:  # Normal breeding loop
        current_offspring_count = 0
        while len(organisms_new) < settings["pop_size"]:
            # Select parents
            if "potential_parents_data" in locals() and potential_parents_data:
                # If using fallback pool, sample weights directly
                p1_idx, p2_idx = (
                    sample(range(len(potential_parents_data)), 2)
                    if len(potential_parents_data) >= 2
                    else (0, 0)
                )
                p1_wih, p1_who = potential_parents_data[p1_idx]
                p2_wih, p2_who = potential_parents_data[p2_idx]
            elif parent_pool:
                # If using elite pool, sample organism objects
                parent1, parent2 = (
                    sample(parent_pool, 2)
                    if len(parent_pool) >= 2
                    else (parent_pool[0], parent_pool[0])
                )
                p1_wih, p1_who = parent1.wih, parent1.who
                p2_wih, p2_who = parent2.wih, parent2.who
            else:
                # Should be impossible state if logic above is correct, but safety break
                print("  Error in breeding loop parent selection.")
                break

            # CROSSOVER
            crossover_weight = random()
            wih_new = (crossover_weight * p1_wih) + ((1 - crossover_weight) * p2_wih)
            who_new = (crossover_weight * p1_who) + ((1 - crossover_weight) * p2_who)

            # MUTATION
            if random() <= settings["mutate"]:
                mat_pick = randint(0, 1)
                # Mutate WIH
                if mat_pick == 0 and settings["hnodes"] > 0 and settings["inodes"] > 0:
                    idx_row, idx_col = randint(0, settings["hnodes"] - 1), randint(
                        0, settings["inodes"] - 1
                    )
                    wih_new[idx_row, idx_col] *= uniform(0.9, 1.1)
                    wih_new[idx_row, idx_col] = max(
                        -1.0, min(wih_new[idx_row, idx_col], 1.0)
                    )
                # Mutate WHO
                elif (
                    mat_pick == 1 and settings["onodes"] > 0 and settings["hnodes"] > 0
                ):
                    idx_row, idx_col = randint(0, settings["onodes"] - 1), randint(
                        0, settings["hnodes"] - 1
                    )
                    who_new[idx_row, idx_col] *= uniform(0.9, 1.1)
                    who_new[idx_row, idx_col] = max(
                        -1.0, min(who_new[idx_row, idx_col], 1.0)
                    )

            # Add the offspring (constructor resets state)
            new_name = f"gen[{gen}]-org[{elitism_num_actual + current_offspring_count}]"
            organisms_new.append(
                organism(settings, wih=wih_new, who=who_new, name=new_name)
            )
            current_offspring_count += 1

    # Return the new generation list, ensuring correct size
    return organisms_new[: settings["pop_size"]], stats


# --- SIMULATION FUNCTION ------------------------------------------------------+


def simulate(settings, organisms, foods, gen):
    """Runs simulation physics and logic for one generation.
    Includes starvation checks.
    Returns: tuple (updated_organisms_list, was_window_closed_flag)
    """
    global fig  # To check if plot window exists

    total_time_steps = int(settings["gen_time"] / settings["dt"])
    plot_interval = settings.get("plot_interval", 1)
    starvation_threshold = settings["starvation_threshold"]

    # --- Reset organism state for the START of this generation's simulation ---
    # Important: Ensures organisms carried over by elitism don't retain
    # high fitness or dead status from the previous generation's *simulation*.
    # Their *weights* are preserved by 'evolve'.
    for org in organisms:
        org.fitness = 0
        org.is_alive = True
        org.time_since_last_meal = 0
        # Reset position/velocity to random for fair start each gen
        org.x = uniform(settings["x_min"], settings["x_max"])
        org.y = uniform(settings["y_min"], settings["y_max"])
        org.r = uniform(0, 360)
        org.v = uniform(0, settings["v_max"])
        # Sensory inputs will be updated each step
        org.d_food = float("inf")
        org.r_food = 0.0

    # --- SIMULATION TIME STEP LOOP -----------------------+
    for t_step in range(total_time_steps):

        # --- Check for window closure signal ---
        if settings["plot"]:
            if fig is None or not plt.fignum_exists(fig.number):
                print("\nPlot window closed detected inside simulate.")
                return organisms, True  # Signal closure

            # --- Plotting ---
            if t_step % plot_interval == 0:
                plot_frame_live(settings, organisms, foods, gen, t_step)

        # Track which living organisms ate this step
        org_ate_this_step = {
            i: False for i, org in enumerate(organisms) if org.is_alive
        }

        # --- Interaction & State Update Loop ---
        for i, org in enumerate(organisms):
            if not org.is_alive:
                continue  # Skip dead organisms

            # 1. Sensing: Find closest food
            org.d_food = float("inf")
            nearest_food_idx = -1
            for j, food in enumerate(foods):
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # Check for eating food first
                if food_org_dist <= settings["collision_radius"]:
                    org.fitness += food.energy
                    food.respawn(settings)
                    org.time_since_last_meal = 0  # Reset starvation timer!
                    org_ate_this_step[i] = True
                    # Assume can only eat one food particle per time step
                    break  # Move to next organism after eating

                # If not eaten, check if this food is the closest seen so far
                elif food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    nearest_food_idx = j

            # Calculate heading to closest food (if one was found and not eaten)
            if not org_ate_this_step[i] and nearest_food_idx != -1:
                org.r_food = calc_heading(org, foods[nearest_food_idx])
            else:
                # If ate, or no food nearby, heading input is zero
                org.r_food = 0.0

            # 2. Starvation Update (if didn't eat this step)
            if not org_ate_this_step[i]:
                org.time_since_last_meal += 1
                if org.time_since_last_meal > starvation_threshold:
                    org.is_alive = False
                    org.fitness = 0  # Dead organisms contribute no fitness
                    # Optional: Stop movement immediately upon death
                    # org.v = 0

            # 3. Thinking & Movement (only if still alive after starvation check)
            if org.is_alive:
                # Ensure weights are valid
                if (
                    org.wih is None
                    or org.who is None
                    or org.wih.shape != (settings["hnodes"], settings["inodes"])
                    or org.who.shape != (settings["onodes"], settings["hnodes"])
                ):
                    continue  # Skip if weights are malformed

                org.think()
                org.update_r(settings)
                org.update_vel(settings)
                org.update_pos(settings)

    # Simulation loop finished for this generation
    return organisms, False  # Return status: window was not closed


# --- CLASSES ------------------------------------------------------------------+


class food:
    """Represents a food particle."""

    def __init__(self, settings):
        self.settings = settings
        self.respawn(settings)
        self.energy = 1  # Amount of fitness gained when eaten

    def respawn(self, settings):
        """Place the food particle at a random location within bounds."""
        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])


class organism:
    """Represents an organism with NN brain, state, and survival status."""

    def __init__(self, settings, wih=None, who=None, name=None):
        self.settings = settings  # Store settings reference

        # --- NN Architecture ---
        self.inodes = settings["inodes"]
        self.hnodes = settings["hnodes"]
        self.onodes = settings["onodes"]

        # --- Weights ---
        # Initialize randomly if not provided or if shape is incorrect
        if wih is None or wih.shape != (self.hnodes, self.inodes):
            self.wih = np.random.uniform(-1, 1, (self.hnodes, self.inodes))
            if wih is not None:
                print(f"Warning: Correcting WIH shape for {name}")
        else:
            self.wih = wih.copy()  # Important: Copy arrays

        if who is None or who.shape != (self.onodes, self.hnodes):
            self.who = np.random.uniform(-1, 1, (self.onodes, self.hnodes))
            if who is not None:
                print(f"Warning: Correcting WHO shape for {name}")
        else:
            self.who = who.copy()  # Important: Copy arrays

        # --- State Variables (initialized/reset each generation by simulate) ---
        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])
        self.r = uniform(0, 360)  # Orientation [0, 360] degrees
        self.v = uniform(0, settings["v_max"])  # Velocity [0, v_max]
        self.fitness = 0  # Accumulated score for the current gen
        self.is_alive = True  # Survival status
        self.time_since_last_meal = 0  # Starvation counter

        # --- Sensory Input (updated each time step) ---
        self.d_food = float("inf")  # Distance to nearest food
        self.r_food = 0.0  # Normalized heading [-1, 1]

        # --- NN Outputs (calculated by think) ---
        self.nn_dv = 0.0  # Scaler for velocity change [-1, 1]
        self.nn_dr = 0.0  # Scaler for rotation change [-1, 1]

        # --- Identity ---
        self.name = name if name else f"org_{randint(1000,9999)}"

    def think(self):
        """Use NN to determine desired change in velocity and rotation."""
        if not self.is_alive:
            return  # Dead organisms don't think

        # Input: Normalized heading [-1, 1]
        nn_input = np.array([[self.r_food]])  # Shape: (inodes, 1) = (1, 1)

        # Activation function
        af = np.tanh  # Use numpy's tanh directly for arrays

        # Hidden layer: h = act( W_ih * input )
        h1 = af(np.dot(self.wih, nn_input))  # Shape: (hnodes, 1)

        # Output layer: out = act( W_ho * h )
        out = af(np.dot(self.who, h1))  # Shape: (onodes, 1)

        # Assign outputs to control variables
        self.nn_dv = float(out[0, 0])  # Velocity change scaler
        self.nn_dr = float(out[1, 0])  # Rotation change scaler

    def update_r(self, settings):
        """Update orientation based on NN output nn_dr."""
        if not self.is_alive:
            return
        # Calculate change in rotation
        delta_r = self.nn_dr * settings["dr_max"] * settings["dt"]
        # Update and wrap angle
        self.r = (self.r + delta_r) % 360

    def update_vel(self, settings):
        """Update velocity based on NN output nn_dv."""
        if not self.is_alive:
            return
        # Calculate change in velocity (acceleration * time)
        delta_v = self.nn_dv * settings["dv_max"] * settings["dt"]
        # Update velocity
        self.v += delta_v
        # Clamp velocity to [0, v_max]
        self.v = max(0.0, min(self.v, settings["v_max"]))

    def update_pos(self, settings):
        """Update position based on velocity, orientation. Apply wrap-around."""
        if not self.is_alive:
            return
        # Calculate position change
        dx = self.v * cos(radians(self.r)) * settings["dt"]
        dy = self.v * sin(radians(self.r)) * settings["dt"]
        # Update position
        self.x += dx
        self.y += dy

        # Apply toroidal world wrap-around
        world_width = settings["x_max"] - settings["x_min"]
        world_height = settings["y_max"] - settings["y_min"]
        if self.x < settings["x_min"]:
            self.x += world_width
        elif self.x >= settings["x_max"]:
            self.x -= world_width
        if self.y < settings["y_min"]:
            self.y += world_height
        elif self.y >= settings["y_max"]:
            self.y -= world_height


# --- MAIN RUN FUNCTION --------------------------------------------------------+


def run(settings):
    """Sets up and runs the evolutionary simulation."""
    global fig, ax, simulation_stopped_early  # Allow modification
    simulation_stopped_early = False  # Reset at start

    # Setup plotting window if enabled
    if settings["plot"]:
        setup_live_plot(settings)

    # Initialize environment
    print("Initializing food...")
    foods = [food(settings) for _ in range(settings["food_num"])]
    print("Initializing organisms (Generation 0)...")
    organisms = [
        organism(settings, name=f"gen[0]-org[{i}]") for i in range(settings["pop_size"])
    ]

    # Data storage and timing
    stats_history = []
    start_total_time = time.time()

    # --- GENERATION LOOP ---
    print(
        f"\nStarting simulation: {settings['gens']} generations, {int(settings['gen_time'] / settings['dt'])} steps/gen"
    )
    generations_completed = 0
    for gen in range(settings["gens"]):
        gen_start_time = time.time()

        # 1. SIMULATE the current population
        #    - Resets organism states (pos, vel, fitness, alive, starvation)
        #    - Runs time steps: sense, think, act, move, eat, starve, die
        #    - Returns updated organism list and window closure status
        organisms, was_closed = simulate(settings, organisms, foods, gen)

        # Check if user closed the plot window during simulation
        if was_closed:
            simulation_stopped_early = True
            print("Stopping simulation loop due to window closure.")
            break  # Exit generation loop

        # 2. EVOLVE the population for the next generation
        #    - Filters for survivors (is_alive == True)
        #    - Calculates stats based on survivors' fitness
        #    - Selects elites (copies weights)
        #    - Breeds offspring (crossover, mutation of weights)
        #    - Returns a *new* list of organism objects for the next generation
        organisms, stats = evolve(settings, organisms, gen + 1)  # Evolve for gen+1
        stats_history.append(stats)
        generations_completed = gen + 1

        # Print generation summary stats
        gen_end_time = time.time()
        living_count = stats.get("COUNT", "N/A")  # Get survivor count from evolve stats
        print(
            f"> GEN: {gen} | Survivors: {living_count:>3} | BEST_fit: {stats['BEST']:<6.1f} | AVG_fit: {stats['AVG']:<6.2f} | WORST_fit: {stats['WORST']:<6.1f} | Duration: {gen_end_time - gen_start_time:.2f}s"
        )

    # --- POST-SIMULATION ---
    end_total_time = time.time()
    print(f"\n--- Simulation Loop Finished ---")
    print(f"Generations run: {generations_completed}")
    if simulation_stopped_early:
        print("(Simulation was stopped early by user action)")
    print(f"Total wall time: {end_total_time - start_total_time:.2f} seconds.")

    # --- Optional: Plot overall statistics graph ---
    if stats_history and not simulation_stopped_early:
        print("\nDisplaying overall fitness statistics graph...")
        try:
            # Create a new, separate figure for the stats plot
            stats_fig, stats_ax = plt.subplots(figsize=(10, 5))
            generations_axis = list(range(len(stats_history)))
            best_fitness = [s["BEST"] for s in stats_history]
            avg_fitness = [s["AVG"] for s in stats_history]
            worst_fitness = [s["WORST"] for s in stats_history]
            survivor_counts = [s["COUNT"] for s in stats_history]  # Get survivor counts

            # Plot fitness lines
            stats_ax.plot(
                generations_axis, best_fitness, label="Best Fitness", color="green"
            )
            stats_ax.plot(
                generations_axis, avg_fitness, label="Average Fitness", color="blue"
            )
            stats_ax.plot(
                generations_axis,
                worst_fitness,
                label="Worst Fitness",
                color="red",
                linestyle=":",
            )
            stats_ax.set_xlabel("Generation")
            stats_ax.set_ylabel("Fitness (Food Eaten)", color="black")
            stats_ax.tick_params(axis="y", labelcolor="black")
            stats_ax.legend(loc="upper left")
            stats_ax.grid(True)

            # Add a secondary y-axis for survivor count
            ax2 = stats_ax.twinx()
            ax2.plot(
                generations_axis,
                survivor_counts,
                label="Survivors",
                color="purple",
                linestyle="--",
            )
            ax2.set_ylabel("Number of Survivors", color="purple")
            ax2.tick_params(axis="y", labelcolor="purple")
            # Set survivor axis limits appropriately (0 to pop_size + buffer)
            ax2.set_ylim(0, settings["pop_size"] * 1.1)
            ax2.legend(loc="upper right")

            stats_fig.tight_layout()  # Adjust plot to prevent overlap
            stats_ax.set_title("Evolution of Fitness and Survivor Count")

            # Turn interactive mode off before showing blocking plot
            plt.ioff()
            plt.show()  # This blocks until the stats window is closed
        except Exception as e_stats:
            print(f"Could not display stats graph: {e_stats}")
            traceback.print_exc()  # Print full error for stats plot issue

    elif simulation_stopped_early:
        print("\nSkipping statistics graph because simulation was stopped early.")

    print("\nRun function finished.")


# --- SCRIPT ENTRY POINT -------------------------------------------------------+

if __name__ == "__main__":
    print("Executing script...")
    try:
        run(settings)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted by user (Ctrl+C) ---")
        simulation_stopped_early = True  # Ensure cleanup knows it was stopped
    except Exception as e:
        print(f"\n--- An unhandled error occurred during execution ---")
        traceback.print_exc()  # Print the full error stack trace
        simulation_stopped_early = True  # Assume stop on error for cleanup
    finally:
        # --- Cleanup ---
        print("\nFinal cleanup: Closing any open Matplotlib figures...")
        # Turning interactive mode off can sometimes help closing behavior
        plt.ioff()
        plt.close("all")  # Close all figures managed by pyplot state machine
        print("Cleanup complete. Exiting script.")
        # Optional: Set exit code based on whether it finished normally
        # exit_code = 1 if simulation_stopped_early else 0
        # sys.exit(exit_code)

# --- END ----------------------------------------------------------------------+
