# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Evolving Simple Organisms
#   2017-Nov.
#
#   Modified for live plotting with robust window handling (using Qt5Agg)
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division, print_function
from collections import defaultdict
import sys  # Used for checking python version if needed

# --- Force Matplotlib backend BEFORE importing pyplot ---
import matplotlib

# Ensure compatibility check isn't strictly necessary but good practice
# print(f"Using Matplotlib backend: {matplotlib.get_backend()}")
try:
    matplotlib.use("Qt5Agg")  # Use Qt5 backend for stability on macOS/conda
except ImportError:
    print("Qt5Agg backend not available. Please install PyQt5.")
    print("conda install pyqt")
    # Or pip install PyQt5
    sys.exit(1)


# Keep plotting imports separate for clarity
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

# Assuming plotting.py is in the same directory for original plot functions
# These functions will be called by plot_frame_live
try:
    from plotting import plot_food as plot_food_external
    from plotting import plot_organism as plot_organism_external
except ImportError:
    print("Warning: plotting.py not found. Using basic internal plotting functions.")

    # Define basic fallback functions if plotting.py is missing
    def plot_food_external(x, y, ax):
        ax.plot(x, y, "bo", markersize=4, alpha=0.6)  # Basic blue dot for food

    def plot_organism_external(x, y, r, ax):
        ax.plot(x, y, "go", markersize=8)  # Basic green dot for organism
        line_length = 0.1
        end_x = x + line_length * cos(radians(r))
        end_y = y + line_length * sin(radians(r))
        ax.plot(
            [x, end_x], [y, end_y], "k-", linewidth=1
        )  # Basic black line for direction


import numpy as np
import operator
import time  # Added for pausing

from math import atan2, cos, degrees, floor, radians, sin, sqrt
from random import randint, random, sample, uniform

# --- CONSTANTS ----------------------------------------------------------------+

settings = {}

# EVOLUTION SETTINGS
settings["pop_size"] = 50  # number of organisms
settings["food_num"] = 100  # number of food particles
settings["gens"] = 50  # number of generations
settings["elitism"] = 0.20  # elitism (selection bias)
settings["mutate"] = 0.10  # mutation rate

# SIMULATION SETTINGS
settings["gen_time"] = 100  # generation length         (seconds)
settings["dt"] = 0.04  # simulation time step      (dt)
settings["dr_max"] = 720  # max rotational speed      (degrees per second)
settings["v_max"] = 0.5  # max velocity              (units per second)
settings["dv_max"] = 0.25  # max acceleration (+/-)    (units per second^2)
settings["collision_radius"] = 0.075  # Radius for eating food

settings["x_min"] = -2.0  # arena western border
settings["x_max"] = 2.0  # arena eastern border
settings["y_min"] = -2.0  # arena southern border
settings["y_max"] = 2.0  # arena northern border

# --- Plotting Settings ---
settings["plot"] = True  # LIVE plot the simulation? (can be slow!)
settings["plot_interval"] = (
    10  # Plot every N time steps (1 = plot every step, >1 = faster)
)

# ORGANISM NEURAL NET SETTINGS
settings["inodes"] = 1  # number of input nodes (normalized heading)
settings["hnodes"] = 5  # number of hidden nodes
settings["onodes"] = 2  # number of output nodes (dv_scale, dr_scale)

# --- GLOBAL PLOT VARIABLES ---
fig, ax = None, None
simulation_stopped_early = False  # Flag to track if user closed plot window

# --- FUNCTIONS ----------------------------------------------------------------+


def dist(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calc_heading(org, food):
    """Calculate normalized heading to food (-1 to 1)."""
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r

    # Normalize angle to [-180, 180]
    while theta_d <= -180:
        theta_d += 360
    while theta_d > 180:
        theta_d -= 360

    return theta_d / 180.0  # Return normalized value


def setup_live_plot(settings):
    """Initializes the plot figure and axes for live plotting."""
    global fig, ax
    print("Setting up plot...")
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust size as needed
    plt.ion()  # Turn on interactive mode
    # Set aspect ratio and limits once
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])  # Add padding
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])  # Add padding
    plt.show()  # Show the plot window immediately
    print("Plot setup complete.")


def plot_frame_live(settings, organisms, foods, gen, time_step):
    """Updates the existing plot figure for live visualization."""
    global fig, ax
    # If the main figure doesn't exist anymore, do nothing further.
    if fig is None or not plt.fignum_exists(fig.number):
        return

    if ax is None:  # Should not happen if setup_live_plot was called
        print("Error: Plot axes not initialized.")
        return

    ax.clear()  # Clear previous drawings

    # Re-apply limits and title every frame
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])
    ax.set_title(f"Generation: {gen} | Time Step: {time_step}")

    # PLOT ORGANISMS (using function from plotting.py or fallback)
    for organism in organisms:
        plot_organism_external(organism.x, organism.y, organism.r, ax)

    # PLOT FOOD PARTICLES (using function from plotting.py or fallback)
    for food in foods:
        plot_food_external(food.x, food.y, ax)

    # Re-apply aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # --- Crucial steps for live update ---
    plt.draw()
    # Pause allows the plot to update and makes animation visible
    plt.pause(0.001)  # Minimal pause for GUI event loop


def evolve(settings, organisms_old, gen):
    """Evolves the population using elitism, crossover, and mutation."""

    elitism_num = int(floor(settings["elitism"] * settings["pop_size"]))
    new_orgs_count = settings["pop_size"] - elitism_num

    # --- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(float)  # Use float for average
    stats["BEST"] = 0.0
    stats["WORST"] = float("inf")

    if not organisms_old:  # Handle empty population case
        stats["AVG"] = 0.0
        stats["WORST"] = 0.0
        return [], stats

    for org in organisms_old:
        fitness = org.fitness
        if fitness > stats["BEST"]:
            stats["BEST"] = fitness
        if fitness < stats["WORST"]:
            stats["WORST"] = fitness
        stats["SUM"] += fitness

    stats["COUNT"] = len(organisms_old)
    stats["AVG"] = stats["SUM"] / stats["COUNT"] if stats["COUNT"] > 0 else 0.0
    if stats["WORST"] == float("inf"):  # Handle case where all fitnesses are 0
        stats["WORST"] = 0.0

    # --- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    orgs_sorted = sorted(
        organisms_old, key=operator.attrgetter("fitness"), reverse=True
    )
    organisms_new = []
    for i in range(min(elitism_num, len(orgs_sorted))):  # Copy best organisms
        elite_org = orgs_sorted[i]
        # Create a new organism instance copying the weights and properties
        # Resetting position/velocity/etc. for the new generation start
        organisms_new.append(
            organism(
                settings,
                wih=elite_org.wih.copy(),
                who=elite_org.who.copy(),
                name=elite_org.name,
            )
        )

    # --- GENERATE NEW ORGANISMS (CROSSOVER & MUTATION) ----+
    elite_pool = organisms_new[:]  # Use the already copied elites as parents

    if not elite_pool:  # Handle edge case: Elitism is 0 or pop_size is very small
        # Fill remaining slots with random organisms if no elites to breed from
        print(
            f"Warning: Elite pool empty for Gen {gen}. Filling with random organisms."
        )
        while len(organisms_new) < settings["pop_size"]:
            wih_init = np.random.uniform(
                -1, 1, (settings["hnodes"], settings["inodes"])
            )
            who_init = np.random.uniform(
                -1, 1, (settings["onodes"], settings["hnodes"])
            )
            organisms_new.append(
                organism(
                    settings,
                    wih_init,
                    who_init,
                    name=f"gen[{gen}]-org[rand{len(organisms_new)}]",
                )
            )

    else:  # Normal breeding process
        for w in range(new_orgs_count):
            # SELECTION (Select 2 parents randomly from the elite pool)
            if len(elite_pool) >= 2:
                parent1, parent2 = sample(elite_pool, 2)
            else:  # If only 1 elite, use it as both parents
                parent1 = parent2 = elite_pool[0]

            # CROSSOVER
            crossover_weight = random()
            wih_new = (crossover_weight * parent1.wih) + (
                (1 - crossover_weight) * parent2.wih
            )
            who_new = (crossover_weight * parent1.who) + (
                (1 - crossover_weight) * parent2.who
            )

            # MUTATION
            if random() <= settings["mutate"]:
                mat_pick = randint(0, 1)  # Pick which weight matrix to mutate

                # MUTATE: WIH WEIGHTS (Input -> Hidden)
                if mat_pick == 0 and settings["hnodes"] > 0 and settings["inodes"] > 0:
                    idx_row = randint(0, settings["hnodes"] - 1)
                    idx_col = randint(0, settings["inodes"] - 1)
                    mutation_factor = uniform(0.9, 1.1)
                    wih_new[idx_row, idx_col] *= mutation_factor
                    # Clamp weight
                    wih_new[idx_row, idx_col] = max(
                        -1.0, min(wih_new[idx_row, idx_col], 1.0)
                    )

                # MUTATE: WHO WEIGHTS (Hidden -> Output)
                elif (
                    mat_pick == 1 and settings["onodes"] > 0 and settings["hnodes"] > 0
                ):
                    idx_row = randint(0, settings["onodes"] - 1)
                    idx_col = randint(0, settings["hnodes"] - 1)
                    mutation_factor = uniform(0.9, 1.1)
                    who_new[idx_row, idx_col] *= mutation_factor
                    # Clamp weight
                    who_new[idx_row, idx_col] = max(
                        -1.0, min(who_new[idx_row, idx_col], 1.0)
                    )

            # Add the offspring
            new_name = f"gen[{gen}]-org[{elitism_num + w}]"
            organisms_new.append(
                organism(settings, wih=wih_new, who=who_new, name=new_name)
            )

    # Ensure population size is maintained, adding random if necessary
    while len(organisms_new) < settings["pop_size"]:
        print(
            f"Warning: Population size low after breeding Gen {gen}. Adding random organism."
        )
        wih_init = np.random.uniform(-1, 1, (settings["hnodes"], settings["inodes"]))
        who_init = np.random.uniform(-1, 1, (settings["onodes"], settings["hnodes"]))
        organisms_new.append(
            organism(
                settings,
                wih_init,
                who_init,
                name=f"gen[{gen}]-org[filler{len(organisms_new)}]",
            )
        )

    # Return the new generation, ensuring exactly pop_size
    return organisms_new[: settings["pop_size"]], stats


def simulate(settings, organisms, foods, gen):
    """Runs simulation for one generation.
    Returns: tuple (updated_organisms, was_closed)
             was_closed is True if plot window was detected closed.
    """
    global fig  # Need access to main figure object to check existence

    total_time_steps = int(settings["gen_time"] / settings["dt"])
    plot_interval = settings.get("plot_interval", 1)

    # --- Reset organism state for simulation (except weights) ---
    for org in organisms:
        org.fitness = 0
        org.x = uniform(settings["x_min"], settings["x_max"])
        org.y = uniform(settings["y_min"], settings["y_max"])
        org.r = uniform(0, 360)
        org.v = uniform(0, settings["v_max"])
        org.d_food = float("inf")
        org.r_food = 0.0

    # --- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(total_time_steps):

        # --- Check for window closure *before* attempting to plot ---
        if settings["plot"]:
            # Check if the figure object exists and if its window number still exists in pyplot's manager
            if fig is None or not plt.fignum_exists(fig.number):
                print("\nPlot window closed detected inside simulate.")
                return organisms, True  # Signal that the simulation should stop

            # --- Plotting Condition ---
            if t_step % plot_interval == 0:
                plot_frame_live(settings, organisms, foods, gen, t_step)

        # --- Organism/Food Interaction ---
        for i, org in enumerate(organisms):
            # Reset sensory input for this step
            org.d_food = float("inf")
            nearest_food_idx = -1

            for j, food in enumerate(foods):
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # Check for eating food
                if food_org_dist <= settings["collision_radius"]:
                    org.fitness += food.energy
                    food.respawn(settings)  # Respawn eaten food
                    # No need to check this eaten food as closest

                # Check if this food is the closest *so far*
                # (only consider it if it wasn't just eaten)
                elif food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    nearest_food_idx = j

            # If a closest food was found, calculate heading
            if nearest_food_idx != -1:
                org.r_food = calc_heading(org, foods[nearest_food_idx])
            else:
                org.r_food = 0.0  # No food found nearby, heading is 0

        # --- Organism Thinking and Movement ---
        for org in organisms:
            # Ensure weights are valid before thinking/moving
            if (
                org.wih is None
                or org.who is None
                or org.wih.shape != (settings["hnodes"], settings["inodes"])
                or org.who.shape != (settings["onodes"], settings["hnodes"])
            ):
                # print(f"Warning: Organism {org.name} has invalid weights! Skipping.")
                continue  # Skip this organism if weights are bad

            org.think()  # NN calculates nn_dv, nn_dr
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)  # Includes boundary wrap/clamp

    # If the loop completes without the window being closed
    return organisms, False  # Return False for was_closed


# --- CLASSES ------------------------------------------------------------------+


class food:
    """Represents a food particle."""

    def __init__(self, settings):
        self.settings = settings
        self.respawn(settings)
        self.energy = 1

    def respawn(self, settings):
        """Place the food particle at a random location."""
        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])


class organism:
    """Represents an organism with position, velocity, orientation, and NN."""

    def __init__(self, settings, wih=None, who=None, name=None):
        self.settings = settings

        # Initial physical state (randomized)
        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])
        self.r = uniform(0, 360)  # Orientation [0, 360] degrees
        self.v = uniform(0, settings["v_max"])  # Velocity    [0, v_max]

        # Sensory input / State (reset each simulation/step)
        self.d_food = float("inf")  # Distance to nearest food
        self.r_food = 0.0  # Normalized heading [-1, 1] to nearest food

        # Fitness (accumulated during simulation)
        self.fitness = 0

        # Neural Network weights & architecture
        self.hnodes = settings["hnodes"]
        self.inodes = settings["inodes"]
        self.onodes = settings["onodes"]

        # Initialize or copy weights (ensure valid shapes)
        if wih is None or wih.shape != (self.hnodes, self.inodes):
            self.wih = np.random.uniform(-1, 1, (self.hnodes, self.inodes))
        else:
            self.wih = wih.copy()  # Copy to prevent shared references

        if who is None or who.shape != (self.onodes, self.hnodes):
            self.who = np.random.uniform(-1, 1, (self.onodes, self.hnodes))
        else:
            self.who = who.copy()  # Copy to prevent shared references

        # NN outputs (updated by think())
        self.nn_dv = 0.0  # Scaler for velocity change
        self.nn_dr = 0.0  # Scaler for rotation change

        # Identity
        self.name = name if name else f"org_{randint(1000,9999)}"

    def think(self):
        """Calculate NN output based on sensory input (r_food)."""
        # Input is self.r_food (already normalized between -1 and 1)
        # Reshape input for matrix multiplication (1 input node)
        nn_input = np.array([[self.r_food]])  # Shape (inodes, 1) which is (1, 1)

        # Activation function (tanh)
        af = lambda x: np.tanh(x)

        # Hidden layer calculation: h = act(W_ih * input)
        # W_ih shape: (hnodes, inodes), input shape: (inodes, 1)
        # Result h1 shape: (hnodes, 1)
        h1 = af(np.dot(self.wih, nn_input))

        # Output layer calculation: out = act(W_ho * h)
        # W_ho shape: (onodes, hnodes), h1 shape: (hnodes, 1)
        # Result out shape: (onodes, 1)
        out = af(np.dot(self.who, h1))

        # Update control outputs based on NN result
        # Outputs are in range [-1, 1] due to tanh
        self.nn_dv = float(out[0][0])  # First output controls velocity change scale
        self.nn_dr = float(out[1][0])  # Second output controls rotation change scale

    def update_r(self, settings):
        """Update orientation based on NN output."""
        # nn_dr is [-1, 1], dr_max is max change per second
        delta_r = self.nn_dr * settings["dr_max"] * settings["dt"]
        self.r = (self.r + delta_r) % 360  # Keep angle in [0, 360)

    def update_vel(self, settings):
        """Update velocity based on NN output."""
        # nn_dv is [-1, 1], dv_max is max acceleration per second^2
        delta_v = self.nn_dv * settings["dv_max"] * settings["dt"]
        self.v += delta_v
        # Clamp velocity
        self.v = max(0.0, min(self.v, settings["v_max"]))

    def update_pos(self, settings):
        """Update position based on velocity, orientation, and wrap-around."""
        # Calculate change in position
        dx = self.v * cos(radians(self.r)) * settings["dt"]
        dy = self.v * sin(radians(self.r)) * settings["dt"]
        self.x += dx
        self.y += dy

        # Toroidal world wrap-around logic
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


# --- MAIN EXECUTION -----------------------------------------------------------+


def run(settings):
    """Main function to run the simulation and evolution."""
    global fig, ax, simulation_stopped_early  # Allow modification of global flag
    simulation_stopped_early = False  # Reset flag at the start

    # --- SETUP PLOTTING (if enabled) ----------------------+
    if settings["plot"]:
        setup_live_plot(settings)

    # --- POPULATE ENVIRONMENT -----------------------------+
    print("Initializing food...")
    foods = [food(settings) for _ in range(settings["food_num"])]
    print("Initializing organisms...")
    organisms = []
    for i in range(settings["pop_size"]):
        organisms.append(
            organism(settings, name=f"gen[0]-org[{i}]")
        )  # Let __init__ create random weights

    # --- Store stats history ---
    stats_history = []
    start_total_time = time.time()

    # --- CYCLE THROUGH EACH GENERATION --------------------+
    print(f"\nStarting simulation for {settings['gens']} generations...")
    generations_completed = 0
    for gen in range(settings["gens"]):
        gen_start_time = time.time()

        # SIMULATE - This now resets organism state internally
        organisms, was_closed = simulate(settings, organisms, foods, gen)

        # --- Check if window was closed during simulate ---
        if was_closed:
            simulation_stopped_early = True
            print("Stopping simulation loop due to window closure.")
            break  # Exit the generation loop

        # EVOLVE (only if loop didn't break)
        # Evolve returns a *new* list of organisms with new weights
        organisms, stats = evolve(
            settings, organisms, gen + 1
        )  # Pass next gen num for naming
        stats_history.append(stats)
        generations_completed = gen + 1  # Track completed generations

        gen_end_time = time.time()
        print(
            f"> GEN: {gen} | BEST: {stats['BEST']:.1f} | AVG: {stats['AVG']:.2f} | WORST: {stats['WORST']:.1f} | Duration: {gen_end_time - gen_start_time:.2f}s"
        )

    # --- POST-SIMULATION SUMMARY --------------------------+
    end_total_time = time.time()
    print(f"\nSimulation loop finished after {generations_completed} generations.")
    if simulation_stopped_early:
        print("(Simulation was stopped early by closing the plot window)")
    print(f"Total time: {end_total_time - start_total_time:.2f} seconds.")

    # --- Optional: Plot overall statistics *only if run completed normally* ---
    if stats_history and not simulation_stopped_early:
        print("Displaying fitness statistics graph...")
        try:
            # Create a *new* figure specifically for stats
            stats_fig, stats_ax = plt.subplots(figsize=(10, 5))
            generations = list(range(len(stats_history)))
            best_fitness = [s["BEST"] for s in stats_history]
            avg_fitness = [s["AVG"] for s in stats_history]
            worst_fitness = [s["WORST"] for s in stats_history]

            stats_ax.plot(generations, best_fitness, label="Best Fitness")
            stats_ax.plot(generations, avg_fitness, label="Average Fitness")
            stats_ax.plot(
                generations, worst_fitness, label="Worst Fitness", linestyle="--"
            )
            stats_ax.set_xlabel("Generation")
            stats_ax.set_ylabel("Fitness (Food Eaten)")
            stats_ax.set_title("Evolution of Fitness over Generations")
            stats_ax.legend()
            stats_ax.grid(True)

            # Turn interactive mode OFF before showing blocking stats plot
            plt.ioff()
            plt.show()  # Show the stats plot (this blocks until closed)
        except Exception as e_stats:
            print(f"Could not display stats graph: {e_stats}")

    elif simulation_stopped_early:
        print("Skipping statistics graph because simulation was stopped early.")

    print("Exiting run function.")


# --- SCRIPT ENTRY POINT -------------------------------------------------------+

if __name__ == "__main__":
    try:
        run(settings)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted by user (Ctrl+C) ---")
        simulation_stopped_early = True  # Treat as early stop
    except Exception as e:
        print(f"\n--- An error occurred during execution ---")
        import traceback

        traceback.print_exc()
        simulation_stopped_early = True  # Treat as early stop to ensure cleanup
    finally:
        # --- Ensure all plot windows are closed on exit ---
        print("\nFinal cleanup: Closing any open Matplotlib figures.")
        # Turning interactive mode off might help close behave better sometimes
        plt.ioff()
        plt.close("all")  # Close all figures managed by pyplot
        print("Cleanup complete.")
        # Optional: Exit code indicates success or failure/interruption
        exit_code = 1 if simulation_stopped_early else 0
        # sys.exit(exit_code) # Uncomment if specific exit code is desired

# --- END ----------------------------------------------------------------------+
