# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Evolving Simple Organisms
#   2017-Nov.
#   Modified for live plotting
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division, print_function
from collections import defaultdict

import matplotlib

matplotlib.use("Qt5Agg")  # <--- USE THIS BACKEND

# Keep plotting imports separate for clarity
import matplotlib.pyplot as plt  # Modified import
from matplotlib.patches import Circle
import matplotlib.lines as lines

# Assuming plotting.py is in the same directory and contains
# plot_food and plot_organism as provided previously.
from plotting import plot_food
from plotting import plot_organism

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

settings["x_min"] = -2.0  # arena western border
settings["x_max"] = 2.0  # arena eastern border
settings["y_min"] = -2.0  # arena southern border
settings["y_max"] = 2.0  # arena northern border

# --- Plotting Settings ---
settings["plot"] = True  # LIVE plot the simulation? (can be slow!)
settings["plot_interval"] = (
    5  # Plot every N time steps (1 = plot every step, >1 = faster)
)

# ORGANISM NEURAL NET SETTINGS
settings["inodes"] = 1  # number of input nodes
settings["hnodes"] = 5  # number of hidden nodes
settings["onodes"] = 2  # number of output nodes


# --- GLOBAL PLOT VARIABLES (for live plotting) ---
fig, ax = None, None

# --- FUNCTIONS ----------------------------------------------------------------+


def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r

    # Normalize angle to [-180, 180]
    while theta_d <= -180:
        theta_d += 360
    while theta_d > 180:
        theta_d -= 360

    return theta_d / 180.0  # Return normalized value directly


def setup_live_plot(settings):
    """Initializes the plot figure and axes for live plotting."""
    global fig, ax
    print("Setting up plot...")
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust size as needed
    plt.ion()  # Turn on interactive mode
    plt.show()  # Show the plot window immediately
    print("Plot setup complete.")


def plot_frame_live(settings, organisms, foods, gen, time_step):
    """Updates the existing plot figure for live visualization."""
    global fig, ax
    if ax is None:  # Should not happen if setup_live_plot was called
        print("Error: Plot axes not initialized.")
        return

    ax.clear()  # Clear previous drawings

    # Set plot limits for every frame
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])  # Add padding
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])  # Add padding

    # PLOT ORGANISMS (using function from plotting.py)
    for organism in organisms:
        plot_organism(organism.x, organism.y, organism.r, ax)

    # PLOT FOOD PARTICLES (using function from plotting.py)
    for food in foods:
        plot_food(food.x, food.y, ax)

    # MISC PLOT SETTINGS
    ax.set_aspect("equal", adjustable="box")
    # Optional: remove ticks for cleaner look
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Update plot title/text
    ax.set_title(f"Generation: {gen} | Time Step: {time_step}")
    # Removed plt.figtext as it behaves differently with interactive plots

    # --- Crucial steps for live update ---
    plt.draw()
    # Pause allows the plot to update and makes animation visible
    # A very small pause is needed for the GUI event loop
    plt.pause(0.001)


# (evolve function remains the same as in your original code)
def evolve(settings, organisms_old, gen):

    elitism_num = int(floor(settings["elitism"] * settings["pop_size"]))
    new_orgs = settings["pop_size"] - elitism_num

    # --- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(float)  # Use float for average
    stats["BEST"] = 0.0
    stats["WORST"] = float("inf")

    # Handle potential division by zero if pop_size is 0
    if not organisms_old:
        stats["AVG"] = 0.0
        stats["WORST"] = 0.0
        # Return empty list and stats if initial population is empty
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
    for i in range(min(elitism_num, len(orgs_sorted))):  # Ensure index is valid
        # Create a new organism instance copying the weights
        elite_org = orgs_sorted[i]
        organisms_new.append(
            organism(
                settings,
                wih=elite_org.wih.copy(),
                who=elite_org.who.copy(),
                name=elite_org.name,
            )
        )

    # --- GENERATE NEW ORGANISMS ---------------------------+
    # Pool of parents for crossover comes from the elite group
    elite_pool = orgs_sorted[:elitism_num]

    if not elite_pool:  # Handle edge case: Elitism is 0 or pop_size is very small
        print(
            f"Warning: Elite pool is empty for Gen {gen}. Filling with random organisms."
        )
        while len(organisms_new) < settings["pop_size"]:
            # Create new random organisms if no elites to breed from
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
        for w in range(new_orgs):
            # SELECTION (TRUNCATION SELECTION from elite pool)
            # Ensure we have at least 2 parents if possible, otherwise reuse the single elite
            if len(elite_pool) >= 2:
                parent1, parent2 = sample(elite_pool, 2)
            else:
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
                # PICK WHICH WEIGHT MATRIX TO MUTATE (0: wih, 1: who)
                mat_pick = randint(0, 1)

                # MUTATE: WIH WEIGHTS (Input -> Hidden)
                if mat_pick == 0 and settings["hnodes"] > 0:  # Check hnodes > 0
                    index_row = randint(0, settings["hnodes"] - 1)
                    index_col = randint(
                        0, settings["inodes"] - 1
                    )  # WIH is (hnodes, inodes)
                    mutation_factor = uniform(0.9, 1.1)
                    wih_new[index_row, index_col] *= mutation_factor
                    # Clamp weights to [-1, 1]
                    wih_new[index_row, index_col] = max(
                        -1.0, min(wih_new[index_row, index_col], 1.0)
                    )

                # MUTATE: WHO WEIGHTS (Hidden -> Output)
                elif (
                    mat_pick == 1 and settings["onodes"] > 0 and settings["hnodes"] > 0
                ):  # Check nodes > 0
                    index_row = randint(0, settings["onodes"] - 1)
                    index_col = randint(0, settings["hnodes"] - 1)
                    mutation_factor = uniform(0.9, 1.1)
                    who_new[index_row, index_col] *= mutation_factor
                    # Clamp weights to [-1, 1]
                    who_new[index_row, index_col] = max(
                        -1.0, min(who_new[index_row, index_col], 1.0)
                    )

            new_name = f"gen[{gen}]-org[{elitism_num + w}]"
            organisms_new.append(
                organism(settings, wih=wih_new, who=who_new, name=new_name)
            )

    # Ensure population size is maintained, potentially adding random if needed
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

    return organisms_new[: settings["pop_size"]], stats  # Ensure correct size


def simulate(settings, organisms, foods, gen):

    total_time_steps = int(settings["gen_time"] / settings["dt"])
    plot_interval = settings.get("plot_interval", 1)  # Get interval, default to 1

    # --- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(0, total_time_steps):

        # --- Live Plotting Condition ---
        # Plot only if 'plot' is True AND it's the right time step interval
        if settings["plot"] and (t_step % plot_interval == 0):
            plot_frame_live(settings, organisms, foods, gen, t_step)

        # --- Organism/Food Interaction ---
        for i, org in enumerate(organisms):
            # Reset sensory input for this step
            org.d_food = float("inf")  # Use infinity for comparison
            nearest_food_idx = -1

            for j, food in enumerate(foods):
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # Check for eating food (use a reasonable collision radius)
                collision_radius = 0.075  # Make this a setting?
                if food_org_dist <= collision_radius:
                    org.fitness += food.energy
                    food.respawn(settings)  # Respawn eaten food
                    # No need to check this food as closest now

                # Check if this food is the closest *so far*
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
            # Ensure wih/who are initialized (should happen in __init__ or evolve)
            if org.wih is None or org.who is None:
                print(f"Warning: Organism {org.name} has no weights!")
                # Handle this case: maybe skip thinking/moving or assign random weights?
                # For now, let's skip movement if weights are missing.
                continue

            org.think()  # NN calculates nn_dv, nn_dr
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)  # Includes boundary wrap/clamp

    return organisms  # Return updated organisms with new fitness


# --- CLASSES ------------------------------------------------------------------+


class food:
    def __init__(self, settings):
        self.settings = settings  # Store settings if needed later
        self.respawn(settings)  # Initial placement
        self.energy = 1

    def respawn(self, settings):
        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])
        # self.energy = 1 # Reset energy if it could change


class organism:
    def __init__(self, settings, wih=None, who=None, name=None):
        self.settings = settings  # Store settings

        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])
        self.r = uniform(0, 360)  # orientation [0, 360] degrees
        self.v = uniform(0, settings["v_max"])  # velocity    [0, v_max]
        # self.dv not directly used in updates, nn_dv drives change
        # self.dv = uniform(-settings['dv_max'], settings['dv_max'])

        # Sensory input / State
        self.d_food = float("inf")  # Distance to nearest food
        self.r_food = 0.0  # Normalized heading [-1, 1] to nearest food

        # Fitness
        self.fitness = 0  # fitness (food count)

        # Neural Network weights
        self.hnodes = settings["hnodes"]
        self.inodes = settings["inodes"]
        self.onodes = settings["onodes"]

        if wih is None:
            # Initialize weights if not provided
            self.wih = np.random.uniform(-1, 1, (self.hnodes, self.inodes))
        else:
            self.wih = wih.copy()  # Make sure to copy arrays

        if who is None:
            self.who = np.random.uniform(-1, 1, (self.onodes, self.hnodes))
        else:
            self.who = who.copy()  # Make sure to copy arrays

        # NN outputs (initialized, updated by think())
        self.nn_dv = 0.0
        self.nn_dr = 0.0

        # Identity
        self.name = name if name else f"org_{np.random.randint(1000)}"

    # NEURAL NETWORK
    def think(self):
        # Input is self.r_food (already normalized between -1 and 1 by calc_heading)
        nn_input = np.array([[self.r_food]])  # Shape (1, 1)

        # Activation function
        af = lambda x: np.tanh(x)

        # Check if weights exist and have correct shapes
        # Wih: (hnodes, inodes), Input: (inodes, 1) -> Result (hnodes, 1)
        if self.wih is None or self.wih.shape != (self.hnodes, self.inodes):
            print(
                f"Error: Invalid wih shape for {self.name}. Expected {(self.hnodes, self.inodes)}, Got {self.wih.shape if self.wih is not None else 'None'}"
            )
            # Provide default outputs or handle error
            self.nn_dv, self.nn_dr = 0.0, 0.0
            return

        # Who: (onodes, hnodes), Hidden: (hnodes, 1) -> Result (onodes, 1)
        if self.who is None or self.who.shape != (self.onodes, self.hnodes):
            print(
                f"Error: Invalid who shape for {self.name}. Expected {(self.onodes, self.hnodes)}, Got {self.who.shape if self.who is not None else 'None'}"
            )
            self.nn_dv, self.nn_dr = 0.0, 0.0
            return

        # Hidden layer calculation
        # Ensure input is treated as column vector if inodes > 1 (here inodes=1)
        h1 = af(np.dot(self.wih, nn_input))

        # Output layer calculation
        out = af(np.dot(self.who, h1))

        # UPDATE dv AND dr WITH MLP RESPONSE
        # Output ranges from [-1, +1] due to tanh
        self.nn_dv = float(out[0][0])  # Output 1: change in velocity scaler
        self.nn_dr = float(out[1][0])  # Output 2: change in rotation scaler

    # UPDATE HEADING (Orientation)
    def update_r(self, settings):
        # nn_dr is [-1, 1], dr_max is max change per second
        # Change in angle = nn_dr * max_rate * dt
        delta_r = self.nn_dr * settings["dr_max"] * settings["dt"]
        self.r += delta_r
        self.r = self.r % 360  # Keep angle in [0, 360)

    # UPDATE VELOCITY
    def update_vel(self, settings):
        # nn_dv is [-1, 1], dv_max is max change per second^2
        delta_v = self.nn_dv * settings["dv_max"] * settings["dt"]
        self.v += delta_v
        # Clamp velocity
        self.v = max(0.0, min(self.v, settings["v_max"]))

    # UPDATE POSITION (Implement wrap-around boundary)
    def update_pos(self, settings):
        # Calculate change in position
        dx = self.v * cos(radians(self.r)) * settings["dt"]
        dy = self.v * sin(radians(self.r)) * settings["dt"]
        self.x += dx
        self.y += dy

        # Toroidal wrap-around
        if self.x < settings["x_min"]:
            self.x = settings["x_max"] - (settings["x_min"] - self.x) % (
                settings["x_max"] - settings["x_min"]
            )
        elif self.x >= settings["x_max"]:
            self.x = settings["x_min"] + (self.x - settings["x_min"]) % (
                settings["x_max"] - settings["x_min"]
            )

        if self.y < settings["y_min"]:
            self.y = settings["y_max"] - (settings["y_min"] - self.y) % (
                settings["y_max"] - settings["y_min"]
            )
        elif self.y >= settings["y_max"]:
            self.y = settings["y_min"] + (self.y - settings["y_min"]) % (
                settings["y_max"] - settings["y_min"]
            )


# --- MAIN ---------------------------------------------------------------------+


def run(settings):
    global fig, ax  # Allow run to potentially close the plot

    # --- SETUP PLOTTING (if enabled) ----------------------+
    if settings["plot"]:
        setup_live_plot(settings)

    # --- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    print("Initializing food...")
    foods = [food(settings) for _ in range(settings["food_num"])]

    # --- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    print("Initializing organisms...")
    organisms = []
    for i in range(settings["pop_size"]):
        # Ensure initial weights are created correctly
        wih_init = np.random.uniform(-1, 1, (settings["hnodes"], settings["inodes"]))
        who_init = np.random.uniform(-1, 1, (settings["onodes"], settings["hnodes"]))
        organisms.append(
            organism(settings, wih=wih_init, who=who_init, name=f"gen[0]-org[{i}]")
        )

    # --- Store stats history ---
    stats_history = []
    start_total_time = time.time()

    # --- CYCLE THROUGH EACH GENERATION --------------------+
    print(f"\nStarting simulation for {settings['gens']} generations...")
    for gen in range(settings["gens"]):
        gen_start_time = time.time()

        # --- Reset fitness for the new generation ---
        # (Fitness is reset implicitly by how simulate/evolve works,
        # but good practice to ensure it if needed)
        # for org in organisms:
        #     org.fitness = 0 # Done by evolve creating new orgs or simulation starting fresh

        # SIMULATE
        # Organisms list is updated in place by simulate and evolve returns a new list
        organisms = simulate(settings, organisms, foods, gen)

        # EVOLVE
        organisms, stats = evolve(settings, organisms, gen + 1)  # Pass next gen num
        stats_history.append(stats)

        gen_end_time = time.time()
        print(
            f"> GEN: {gen} | BEST: {stats['BEST']:.1f} | AVG: {stats['AVG']:.2f} | WORST: {stats['WORST']:.1f} | Duration: {gen_end_time - gen_start_time:.2f}s"
        )

        # Optional: Add a check to stop if plot window is closed
        if settings["plot"] and not plt.fignum_exists(fig.number):
            print("Plot window closed, stopping simulation.")
            break

    end_total_time = time.time()
    print(f"\nSimulation finished after {len(stats_history)} generations.")
    print(f"Total time: {end_total_time - start_total_time:.2f} seconds.")

    # --- Keep plot open at the end ---
    if settings["plot"] and plt.fignum_exists(fig.number):
        print("Simulation ended. Close the plot window to exit.")
        plt.ioff()  # Turn off interactive mode
        # No need to call plt.show() again if plt.ion() was used,
        # unless you want to ensure it blocks here.
        # plt.show() # This would block until the window is closed
    elif settings["plot"]:
        print("Plot window was closed during simulation.")

    # --- Optional: Plot overall statistics ---
    if stats_history:
        plt.figure(figsize=(10, 5))  # Create a new figure for stats
        generations = list(range(len(stats_history)))
        best_fitness = [s["BEST"] for s in stats_history]
        avg_fitness = [s["AVG"] for s in stats_history]
        worst_fitness = [s["WORST"] for s in stats_history]

        plt.plot(generations, best_fitness, label="Best Fitness")
        plt.plot(generations, avg_fitness, label="Average Fitness")
        plt.plot(generations, worst_fitness, label="Worst Fitness", linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Food Eaten)")
        plt.title("Evolution of Fitness over Generations")
        plt.legend()
        plt.grid(True)
        plt.show()  # Show the stats plot (will block until closed)

    print("Exiting.")
    pass


# --- RUN ----------------------------------------------------------------------+

if __name__ == "__main__":
    # Wrap run in a try block to ensure plot closes if error occurs
    try:
        run(settings)
    except Exception as e:
        print(f"\n--- An error occurred ---")
        import traceback

        traceback.print_exc()
        # Ensure interactive mode is off if an error happens mid-plot
        if settings.get("plot", False):  # Use .get for safety
            plt.ioff()
            plt.close("all")  # Close any open figures
    finally:
        # Optional: Any final cleanup
        print("Cleanup complete.")


# --- END ----------------------------------------------------------------------+
