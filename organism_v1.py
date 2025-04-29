# --- START OF FILE organism_v3.py ---

# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Evolving Simple Organisms
#   2017-Nov.
#
#   Modified for:
#     - Live plotting (Qt5Agg)
#     - Energy-based survival mechanic
#     - Corpse creation on death (corpses are food)
#     - Visual indicator for corpse eaters
#     - Robust window handling
#     - Food respawning on consumption
#     - Energy-based organism collision (REVISED)
#     - Organism sensing (NEW)
#     - Limited food per generation (respawned)
#     - Generation ends on death threshold or max time
#     - Survivors carry over full state (pos, vel, etc.)
#     - No explicit elitism; population filled with offspring of survivors.
#   (Python 3.6 Compatible Syntax - CORRECTED)
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+
from __future__ import division, print_function
from collections import defaultdict
import sys
import time
import operator
import traceback
import matplotlib

try:
    matplotlib.use("Qt5Agg")
except ImportError:
    print("Qt5Agg backend needed: conda install pyqt OR pip install PyQt5")
    sys.exit(1)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines
import numpy as np
from math import atan2, cos, degrees, floor, radians, sin, sqrt, inf
from random import randint, random, sample, uniform

# --- Plotting Setup (Using external/fallback plotting from v1/v2) ---
try:
    from plotting import plot_food as plot_food_original
    from plotting import plot_organism as plot_organism_original

    print("Using plotting functions from plotting.py")

    # --- Adapted plotting functions ---
    def plot_food_adapted(x, y, ax, is_corpse):
        if is_corpse:
            circle = Circle([x, y], 0.035, ec="orange", fc="yellow", zorder=6)
            ax.add_artist(circle)
        else:
            plot_food_original(x, y, ax)

    def plot_organism_adapted(x, y, r, ax, is_alive, ate_corpse_recently):
        if is_alive:
            if ate_corpse_recently:
                circle = Circle([x, y], 0.05, ec="darkred", fc="salmon", zorder=8)
                ax.add_artist(circle)
                edge = Circle([x, y], 0.05, facecolor="None", ec="darkred", zorder=8)
                ax.add_artist(edge)
                x2, y2 = x + cos(radians(r)) * 0.075, y + sin(radians(r)) * 0.075
                ax.add_line(
                    lines.Line2D(
                        [x, x2], [y, y2], color="darkred", linewidth=1, zorder=10
                    )
                )
            else:
                plot_organism_original(x, y, r, ax)
        else:
            ax.add_artist(
                Circle([x, y], 0.04, ec="grey", fc="lightgrey", zorder=7, alpha=0.7)
            )

    plot_food_external = plot_food_adapted
    plot_organism_external = plot_organism_adapted
except ImportError:
    print("Warning: plotting.py not found. Using basic internal plotting functions.")

    def plot_food_external(x, y, ax, is_corpse):
        color, edgec, size = (
            ("yellow", "orange", 5) if is_corpse else ("blue", "darkblue", 4)
        )
        ax.plot(x, y, "o", mfc=color, mec=edgec, ms=size, alpha=0.8)

    def plot_organism_external(x, y, r, ax, is_alive, ate_corpse_recently):
        if is_alive:
            color, edgec, linec = (
                ("salmon", "darkred", "darkred")
                if ate_corpse_recently
                else ("lightgreen", "darkgreen", "black")
            )
            ax.plot(x, y, "o", mfc=color, mec=edgec, ms=8)
            ex, ey = x + 0.1 * cos(radians(r)), y + 0.1 * sin(radians(r))
            ax.plot([x, ex], [y, ey], "-", color=linec, lw=1)
        else:
            ax.plot(x, y, "o", color="grey", ms=6, alpha=0.5)


# --- CONSTANTS / SETTINGS -----------------------------------------------------+
settings = {}
# EVOLUTION
settings["pop_size"] = 50
settings["food_num"] = 50
settings["gens"] = 10000
settings["elitism"] = 0.0
settings["mutate"] = 0.05
# SIMULATION
settings["gen_time"] = 500
settings["dt"] = 0.04
settings["dr_max"] = 720  # Max Rotational speed change (degrees/sec)
settings["v_max"] = 0.5  # Max Velocity (units/sec)
settings["dv_max"] = 0.05  # Max Velocity change (units/sec^2)
settings["collision_radius"] = 0.075  # Radius for food eating and organism collision
settings["sensor_radius"] = 0.8  # <<< NEW: Radius for sensing other organisms
settings["bounce_velocity_multiplier"] = (
    1.0  # Velocity multiplier after losing collision
)
settings["debug_collisions"] = False  # <<< NEW: Set True to print collision details
# ENERGY & SURVIVAL
_base_survival_steps = 100
settings["initial_energy"] = _base_survival_steps
settings["energy_per_food"] = _base_survival_steps
settings["corpse_energy_multiplier"] = 3.0
settings["corpse_fitness_value"] = 3
settings["energy_per_step"] = 1
settings["max_energy"] = _base_survival_steps * 5
settings["death_threshold_percent"] = 0.75
# ARENA & PLOTTING
settings["x_min"], settings["x_max"] = -2.0, 2.0
settings["y_min"], settings["y_max"] = -2.0, 2.0
settings["plot"] = True
settings["plot_interval"] = 10
# NN
settings["inodes"] = 3  # <<< UPDATED: Input Nodes (r_food, r_org, d_org_norm)
settings["hnodes"] = 5  # Hidden Nodes
settings["onodes"] = 2  # Output Nodes (dv, dr)

# --- GLOBAL PLOT VARIABLES ----------------------------------------------------+
fig, ax = None, None
simulation_stopped_early = False


# --- HELPER FUNCTIONS (dist, calc_heading, setup_live_plot are unchanged) -----+
def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Calculates relative angle normalized to [-1, 1]
def calc_heading(source_org, target_x, target_y):
    d_x = target_x - source_org.x
    d_y = target_y - source_org.y
    target_angle_rad = atan2(d_y, d_x)
    relative_angle_deg = degrees(target_angle_rad) - source_org.r

    # Normalize angle to [-180, 180]
    while relative_angle_deg <= -180:
        relative_angle_deg += 360
    while relative_angle_deg > 180:
        relative_angle_deg -= 360

    # Normalize to [-1, 1] for NN input
    return relative_angle_deg / 180.0


def setup_live_plot(settings):
    global fig, ax
    print("Setting up plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])
    plt.show()
    plt.pause(0.1)
    print("Plot setup complete.")


# --- Modified plot_frame_live (Unchanged from v1/v2) ---
def plot_frame_live(settings, organisms, foods, gen, time_step):
    global fig, ax
    if fig is None or ax is None or not plt.fignum_exists(fig.number):
        return
    ax.clear()
    ax.set_xlim([settings["x_min"] - 0.5, settings["x_max"] + 0.5])
    ax.set_ylim([settings["y_min"] - 0.5, settings["y_max"] + 0.5])
    ax.set_title(f"Generation: {gen} | Time Step: {time_step}")
    for food in foods:
        if food.is_active:
            plot_food_external(food.x, food.y, ax, food.is_corpse)
    for org in organisms:
        plot_organism_external(
            org.x, org.y, org.r, ax, org.is_alive, org.ate_corpse_recently
        )
    ax.set_aspect("equal", adjustable="box")
    plt.draw()
    plt.pause(0.001)


# --- EVOLUTION FUNCTION (Handles potential NN shape change) ---
def evolve(settings, organisms_old, gen):
    survivors = [org for org in organisms_old if org.is_alive]
    num_survivors = len(survivors)
    print(f"  [Gen {gen-1}] Survivors: {num_survivors}/{settings['pop_size']}")

    if num_survivors == 0:
        print(f"  ##### EXTINCTION in Gen {gen-1}! Simulation will end. #####")
        return [], defaultdict(float, {"COUNT": 0})

    stats = defaultdict(float, {"WORST": float("inf"), "BEST": 0.0, "SUM": 0.0})
    for org in survivors:
        fitness = org.fitness
        stats["BEST"] = max(stats["BEST"], fitness)
        stats["WORST"] = min(stats["WORST"], fitness)
        stats["SUM"] += fitness
    stats["COUNT"] = num_survivors
    stats["AVG"] = stats["SUM"] / num_survivors if num_survivors > 0 else 0.0
    if stats["WORST"] == float("inf"):
        stats["WORST"] = 0.0

    organisms_new = []
    # Keep survivors - they retain their state (pos, vel, energy, weights)
    for survivor in survivors:
        # Check if survivor's weights match current NN settings
        # If inodes changed, the old weights are incompatible. Reinitialize.
        # This simulates a "mutation" forced by changing sensor morphology.
        if survivor.wih.shape[1] != settings["inodes"]:
            print(
                f"  Survivor {survivor.name} has incompatible input weights ({survivor.wih.shape}). Reinitializing wih."
            )
            survivor.wih = np.random.uniform(
                -1, 1, (settings["hnodes"], settings["inodes"])
            )
            # Optionally reinitialize who as well, or just let evolution adapt it.
            # survivor.who = np.random.uniform(-1, 1, (settings["onodes"], settings["hnodes"]))

        # Reset fitness, alive status etc for the *next* generation simulation
        # (This is now done at the start of simulate(), keep state here)
        organisms_new.append(survivor)

    # Breed offspring to fill remaining spots
    spots_to_fill = settings["pop_size"] - num_survivors
    parent_pool = survivors
    if spots_to_fill > 0:
        for i in range(spots_to_fill):
            parent1, parent2 = (
                sample(parent_pool, 2)
                if num_survivors >= 2
                else (parent_pool[0], parent_pool[0])
            )

            # Crossover weights
            c_weight = random()
            wih_new = (c_weight * parent1.wih) + ((1 - c_weight) * parent2.wih)
            who_new = (c_weight * parent1.who) + ((1 - c_weight) * parent2.who)

            # Mutation
            if random() <= settings["mutate"]:
                mat, h, inn, o = (
                    randint(0, 1),
                    settings["hnodes"],
                    settings["inodes"],
                    settings["onodes"],
                )
                if mat == 0 and h > 0 and inn > 0 and wih_new.size > 0:
                    r, c = randint(0, h - 1), randint(0, inn - 1)
                    wih_new[r, c] = max(
                        -1.0, min(1.0, wih_new[r, c] + uniform(-0.1, 0.1))
                    )  # Additive mutation
                elif mat == 1 and o > 0 and h > 0 and who_new.size > 0:
                    r, c = randint(0, o - 1), randint(0, h - 1)
                    who_new[r, c] = max(
                        -1.0, min(1.0, who_new[r, c] + uniform(-0.1, 0.1))
                    )  # Additive mutation

            # Ensure offspring weights match current settings (should happen naturally if parents match)
            if wih_new.shape != (settings["hnodes"], settings["inodes"]):
                wih_new = np.random.uniform(
                    -1, 1, (settings["hnodes"], settings["inodes"])
                )
            if who_new.shape != (settings["onodes"], settings["hnodes"]):
                who_new = np.random.uniform(
                    -1, 1, (settings["onodes"], settings["hnodes"])
                )

            offspring_name = f"gen[{gen}]-offspring[{i}]"
            # Offspring start with default state (random pos, vel, initial energy)
            organisms_new.append(
                organism(
                    settings,
                    wih=wih_new,
                    who=who_new,
                    name=offspring_name,
                    initial_state=None,
                )
            )

    return organisms_new[: settings["pop_size"]], stats


# --- SIMULATION FUNCTION (Revised: Organism sensing, Revised collision) -------+
def simulate(settings, organisms, foods, gen):
    global fig

    max_time_steps = int(settings["gen_time"] / settings["dt"])
    plot_interval = settings.get("plot_interval", 1)
    initial_energy = settings["initial_energy"]
    energy_per_food = settings["energy_per_food"]
    energy_per_step = settings["energy_per_step"]
    max_energy = settings["max_energy"]
    corpse_energy_gain = energy_per_food * settings["corpse_energy_multiplier"]
    corpse_fitness = settings["corpse_fitness_value"]
    death_limit_count = int(settings["pop_size"] * settings["death_threshold_percent"])
    collision_radius_sq = settings["collision_radius"] ** 2
    sensor_radius = settings["sensor_radius"]
    sensor_radius_sq = sensor_radius**2  # For efficiency
    bounce_vel_mult = settings["bounce_velocity_multiplier"]
    debug_collisions = settings["debug_collisions"]

    # Respawn any inactive food/corpses from previous generation
    for food_item in foods:
        if not food_item.is_active:
            food_item.respawn(settings)

    # Reset Organism State for the new generation simulation
    # Survivors keep their position, velocity, orientation, and weights from last gen
    for org in organisms:
        # Only reset simulation-specific state, not physical state if it survived
        org.fitness = 0
        org.is_alive = True  # Assume alive at start of gen
        org.energy = initial_energy  # Reset energy for all
        org.ate_corpse_recently = False
        # Reset sensor inputs for the first step
        org.d_food = inf
        org.r_food = 0.0
        org.d_org = inf
        org.r_org = 0.0
        org.d_org_norm = 1.0  # Default normalized distance (max)
        org.nn_dv = 0.0
        org.nn_dr = 0.0

    gen_ended_early_death = False
    actual_steps = 0
    for t_step in range(max_time_steps):
        actual_steps = t_step + 1

        if settings["plot"]:
            if fig is None or not plt.fignum_exists(fig.number):
                return organisms, True  # Signal window closed
            if t_step % plot_interval == 0:
                plot_frame_live(settings, organisms, foods, gen, t_step)

        current_dead_count = 0
        living_org_indices = [i for i, org in enumerate(organisms) if org.is_alive]

        # --- Step 1: Sensing, Metabolism, Death Check, Eating, Thinking ---
        for i in living_org_indices:
            org = organisms[i]

            # Metabolism and Death Check
            org.energy -= energy_per_step
            if org.energy <= 0:
                org.is_alive = False
                org.fitness = 0  # Lose fitness on death
                # Create Corpse
                for food_slot in foods:
                    if not food_slot.is_active:
                        food_slot.become_corpse(org.x, org.y, corpse_fitness)
                        break
                continue  # Skip rest for this newly dead organism

            # --- Sensing ---
            # Food Sensing
            org.d_food = float("inf")
            nearest_food_idx = -1
            ate_this_step = False
            for j, food in enumerate(foods):
                if not food.is_active:
                    continue
                dx, dy = food.x - org.x, food.y - org.y
                dist_sq = dx * dx + dy * dy

                # Eating Check
                if dist_sq <= collision_radius_sq:
                    org.fitness += food.energy
                    energy_gain = (
                        corpse_energy_gain if food.is_corpse else energy_per_food
                    )
                    org.energy = min(org.energy + energy_gain, max_energy)
                    if food.is_corpse:
                        org.ate_corpse_recently = True
                    food.respawn(settings)  # Respawn food
                    ate_this_step = True
                    # Considered food eaten, reset food sensor for NN this step
                    org.d_food = 0.0  # Indicate food present/eaten
                    org.r_food = 0.0
                    break  # Ate one item

                # Find Closest Food (if not eaten)
                elif dist_sq < org.d_food**2:
                    org.d_food = sqrt(dist_sq)
                    nearest_food_idx = j

            # Calculate food heading if food sensed and not eaten
            if not ate_this_step and nearest_food_idx != -1:
                if foods[nearest_food_idx].is_active:  # Check if still there
                    org.r_food = calc_heading(
                        org, foods[nearest_food_idx].x, foods[nearest_food_idx].y
                    )
                else:  # Food gone
                    org.d_food = float("inf")
                    org.r_food = 0.0
            elif not ate_this_step:  # No food seen
                org.d_food = float("inf")
                org.r_food = 0.0

            # Organism Sensing
            org.d_org = float("inf")
            org.r_org = 0.0
            org.d_org_norm = 1.0  # Default: no organism nearby
            nearest_org_dist_sq = sensor_radius_sq  # Start search within sensor range

            for k in living_org_indices:
                if i == k:
                    continue  # Don't sense self
                other_org = organisms[k]
                dx, dy = other_org.x - org.x, other_org.y - org.y
                dist_sq = dx * dx + dy * dy

                if dist_sq < nearest_org_dist_sq:
                    nearest_org_dist_sq = dist_sq
                    org.d_org = sqrt(dist_sq)  # Actual distance
                    org.r_org = calc_heading(
                        org, other_org.x, other_org.y
                    )  # Relative angle [-1, 1]
                    org.d_org_norm = (
                        org.d_org / sensor_radius
                    )  # Normalized distance [0, 1)

            # --- Thinking ---
            org.think()  # Uses org.r_food, org.r_org, org.d_org_norm

        # --- Update living list after potential deaths ---
        living_org_indices = [i for i, org in enumerate(organisms) if org.is_alive]
        current_dead_count = settings["pop_size"] - len(living_org_indices)

        # --- Step 2: Calculate Intended Movement & Check Collisions ---
        collision_pairs = set()  # Avoid double checks/effects
        bounce_applied = {
            i: False for i in living_org_indices
        }  # Track who bounced this step

        for i_idx, i in enumerate(living_org_indices):
            org1 = organisms[i]

            # Calculate intended rotation and velocity change for this step
            org1.calculate_dv_dr()  # Store intended dv, dr based on NN output

            # Check for collisions with organisms later in the list
            for j in living_org_indices[i_idx + 1 :]:
                org2 = organisms[j]

                pair = tuple(sorted((i, j)))  # Unique identifier for the pair
                if pair in collision_pairs:
                    continue  # Already processed this pair

                dx, dy = org1.x - org2.x, org1.y - org2.y
                dist_sq = dx * dx + dy * dy

                if dist_sq <= collision_radius_sq:
                    collision_pairs.add(pair)  # Mark pair as checked

                    # Determine winner/loser based on energy
                    loser = None
                    winner = None
                    if org1.energy < org2.energy:
                        loser, winner = org1, org2
                        loser_idx, winner_idx = i, j
                    elif org2.energy < org1.energy:
                        loser, winner = org2, org1
                        loser_idx, winner_idx = j, i
                    else:  # Equal energy, no bounce
                        if debug_collisions:
                            print(
                                f"  Collision: {org1.name} (E:{org1.energy:.0f}) vs {org2.name} (E:{org2.energy:.0f}). EQUAL ENERGY - No bounce."
                            )
                        continue  # Skip to next pair

                    # Apply bounce effect only if loser hasn't bounced already this step
                    if not bounce_applied[loser_idx]:
                        if debug_collisions:
                            print(
                                f"  Collision: {winner.name} (E:{winner.energy:.0f}) wins vs {loser.name} (E:{loser.energy:.0f}). Applying bounce."
                            )

                        # Loser bounces: reverse direction, reduce velocity
                        loser.r = (loser.r + 180.0) % 360.0
                        loser.v *= bounce_vel_mult
                        # Ensure velocity doesn't go negative (shouldn't with multiplier)
                        loser.v = max(0.0, loser.v)

                        # Mark loser as having bounced this step
                        bounce_applied[loser_idx] = True

                    # Note: Winner continues with their calculated move (no change here)

        # --- Step 3: Apply Movement (Update r, v, then position) ---
        for i in living_org_indices:
            org = organisms[i]
            # Update rotation and velocity based on stored NN outputs (or direct modification if bounced)
            org.update_r(settings)  # Applies the stored delta_r
            org.update_vel(settings)  # Applies the stored delta_v

            # Update position using potentially modified r and v
            org.update_pos(settings)

        # Check Generation End Condition (Death Threshold)
        if current_dead_count >= death_limit_count:
            print(
                f"  Gen {gen} ended early: step {t_step+1}, deaths {current_dead_count}/{settings['pop_size']}."
            )
            gen_ended_early_death = True
            break

    # --- End of Generation ---
    if not gen_ended_early_death:
        final_dead_count = sum(1 for org in organisms if not org.is_alive)
        print(
            f"  Gen {gen} finished max time ({actual_steps} steps). Deaths: {final_dead_count}"
        )

    return organisms, False  # Return False for window status (not closed)


# --- CLASSES (Food unchanged, Organism updated for sensing/NN) --------+
class food:
    # (Identical to v2)
    def __init__(self, settings):
        self.settings = settings
        self.respawn(settings)  # Initial spawn
        self.energy = 1

    def respawn(self, settings):
        self.x = uniform(settings["x_min"], settings["x_max"])
        self.y = uniform(settings["y_min"], settings["y_max"])
        self.is_active = True
        self.is_corpse = False
        self.energy = 1

    def become_corpse(self, x, y, fitness_value):
        self.x = x
        self.y = y
        self.is_active = True
        self.is_corpse = True
        self.energy = fitness_value


class organism:
    def __init__(self, settings, wih=None, who=None, name=None, initial_state=None):
        self.settings = settings
        self.inodes = settings["inodes"]  # Now 3
        self.hnodes = settings["hnodes"]
        self.onodes = settings["onodes"]  # Now 2

        # Weights - Initialize randomly if None or shape mismatch
        if wih is None or wih.shape != (self.hnodes, self.inodes):
            self.wih = np.random.uniform(-1, 1, (self.hnodes, self.inodes))
            if wih is not None:
                print(
                    f"Warning: {name} received incompatible wih shape {wih.shape}, expected {(self.hnodes, self.inodes)}. Reinitialized."
                )
        else:
            self.wih = wih.copy()

        if who is None or who.shape != (self.onodes, self.hnodes):
            self.who = np.random.uniform(-1, 1, (self.onodes, self.hnodes))
            if who is not None:
                print(
                    f"Warning: {name} received incompatible who shape {who.shape}, expected {(self.onodes, self.hnodes)}. Reinitialized."
                )
        else:
            self.who = who.copy()

        # State Variables
        if initial_state:  # Carry over state for survivors
            self.x, self.y, self.r, self.v = (
                initial_state["x"],
                initial_state["y"],
                initial_state["r"],
                initial_state["v"],
            )
            # Energy is reset in simulate()
        else:  # Default state for new offspring
            self.x = uniform(settings["x_min"], settings["x_max"])
            self.y = uniform(settings["y_min"], settings["y_max"])
            self.r = uniform(0, 360)  # Degrees
            self.v = uniform(0, settings["v_max"])

        # Simulation Variables (Reset each generation in simulate())
        self.fitness = 0
        self.is_alive = True
        self.energy = settings["initial_energy"]
        self.ate_corpse_recently = False
        # Sensor Inputs
        self.d_food = inf
        self.r_food = 0.0  # [-1, 1]
        self.d_org = inf
        self.r_org = 0.0  # [-1, 1]
        self.d_org_norm = 1.0  # [0, 1]
        # NN Outputs / Deltas
        self.nn_dv_signal = 0.0  # Raw NN output for dv scaler
        self.nn_dr_signal = 0.0  # Raw NN output for dr scaler
        self.delta_v = 0.0  # Calculated change in v for the step
        self.delta_r = 0.0  # Calculated change in r for the step
        # ---
        self.name = name if name else f"org_{randint(1000, 9999)}"

    def think(self):
        """Use NN to determine desired change scales (dv, dr) based on sensor inputs."""
        if not self.is_alive:
            return

        # --- Prepare Input Vector ---
        # Inputs: [relative_food_angle, relative_org_angle, normalized_org_dist]
        nn_input = np.array(
            [[self.r_food], [self.r_org], [self.d_org_norm]]
        )  # Shape: (inodes, 1) = (3, 1)

        # --- Define Activation Function ---
        af = np.tanh  # Using tanh for hidden and output layers

        # --- Forward Pass ---
        # Hidden layer: h = act( W_ih * input )
        h1 = af(np.dot(self.wih, nn_input))  # Shape: (hnodes, 1)

        # Output layer: out = act( W_ho * h )
        out = af(np.dot(self.who, h1))  # Shape: (onodes, 1) = (2, 1)

        # --- Store NN Output Signals ---
        self.nn_dv_signal = float(out[0, 0])  # Scaler for velocity change [-1, 1]
        self.nn_dr_signal = float(out[1, 0])  # Scaler for rotation change [-1, 1]

    def calculate_dv_dr(self):
        """Calculate the actual delta_v and delta_r for this step based on NN signals."""
        if not self.is_alive:
            return
        # Calculate velocity change based on NN output signal
        self.delta_v = self.nn_dv_signal * self.settings["dv_max"] * self.settings["dt"]
        # Calculate rotation change based on NN output signal
        self.delta_r = self.nn_dr_signal * self.settings["dr_max"] * self.settings["dt"]

    def update_r(self, settings):
        """Apply the calculated rotation change."""
        if not self.is_alive:
            return
        # Note: self.r might have been directly modified by collision logic before this
        self.r = (self.r + self.delta_r) % 360.0

    def update_vel(self, settings):
        """Apply the calculated velocity change, respecting limits."""
        if not self.is_alive:
            return
        # Note: self.v might have been directly modified by collision logic before this
        self.v = self.v + self.delta_v
        # Clamp velocity to [0, v_max]
        self.v = max(0.0, min(self.v, settings["v_max"]))

    def update_pos(self, settings):
        """Update position based on current velocity and orientation. Apply wrap-around."""
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


# --- MAIN RUN FUNCTION (Minor change for debug message) ----------------------
def run(settings):
    global fig, ax, simulation_stopped_early
    simulation_stopped_early = False
    if settings["plot"]:
        setup_live_plot(settings)

    print(f"Initializing food pool (size {settings['food_num']})...")
    foods = [food(settings) for _ in range(settings["food_num"])]
    print(f"Initializing organisms (Gen 0, size {settings['pop_size']})...")
    print(
        f"NN Architecture: Inputs={settings['inodes']}, Hidden={settings['hnodes']}, Outputs={settings['onodes']}"
    )
    organisms = [
        organism(settings, name=f"gen[0]-org[{i}]") for i in range(settings["pop_size"])
    ]

    stats_history = []
    start_total_time = time.time()
    print(
        f"\nStarting simulation: {settings['gens']} generations OR until {settings['death_threshold_percent']*100:.0f}% death per gen."
    )
    if settings["debug_collisions"]:
        print(">>> Collision Debugging Enabled <<<")
    generations_completed = 0
    extinction_occurred = False

    for gen in range(settings["gens"]):
        gen_start_time = time.time()
        # Reset states inside simulate() now handles initial state better
        organisms, was_closed = simulate(settings, organisms, foods, gen)
        if was_closed:
            simulation_stopped_early = True
            print("Stopping loop due to plot window closure.")
            break

        organisms_next, stats = evolve(
            settings, organisms, gen + 1
        )  # Evolve uses survivors
        generations_completed = gen + 1

        if not organisms_next:  # Extinction
            extinction_occurred = True
            stats_history.append(stats)  # Record stats of the last gen
            print(f"> GEN: {gen} | Survivors: 0 | EXTINCTION")
            break

        organisms = organisms_next  # Prepare for next generation
        stats_history.append(stats)
        gen_end_time = time.time()
        living_count = stats.get("COUNT", "N/A")
        print(
            f"> GEN: {gen:<3} | Survivors: {living_count:>3} | BEST_fit: {stats['BEST']:<6.1f} | AVG_fit: {stats['AVG']:<6.2f} | WORST_fit: {stats['WORST']:<6.1f} | Duration: {gen_end_time - gen_start_time:.2f}s"
        )

    # --- POST-SIMULATION ---
    end_total_time = time.time()
    print("\n--- Simulation Loop Finished ---")
    print(f"Generations run: {generations_completed}")
    if extinction_occurred:
        print("   ##### Population went extinct! #####")
    if simulation_stopped_early:
        print("(Simulation was stopped early by user action or plot closure)")
    print(f"Total wall time: {end_total_time - start_total_time:.2f} seconds.")

    # Plot stats (same as before)
    if stats_history and not simulation_stopped_early:
        print("\nDisplaying overall fitness statistics graph...")
        try:
            stats_fig, stats_ax = plt.subplots(figsize=(10, 5))
            gens_axis = list(range(len(stats_history)))
            best = [s["BEST"] for s in stats_history]
            avg = [s["AVG"] for s in stats_history]
            worst = [s["WORST"] for s in stats_history]
            survivors = [s["COUNT"] for s in stats_history]
            stats_ax.plot(gens_axis, best, label="Best", c="g")
            stats_ax.plot(gens_axis, avg, label="Avg", c="b")
            stats_ax.plot(gens_axis, worst, label="Worst", c="r", ls=":")
            stats_ax.set_xlabel("Generation")
            stats_ax.set_ylabel("Fitness")
            stats_ax.legend(loc="upper left")
            stats_ax.grid(True)
            ax2 = stats_ax.twinx()
            ax2.plot(gens_axis, survivors, label="Survivors", c="purple", ls="--")
            ax2.set_ylabel("Survivors", c="purple")
            ax2.tick_params(axis="y", labelcolor="purple")
            ax2.set_ylim(0, settings["pop_size"] * 1.1)
            ax2.legend(loc="upper right")
            stats_fig.tight_layout()
            stats_ax.set_title("Fitness & Survivor Count")
            plt.ioff()
            plt.show()
        except Exception as e_stats:
            print(f"Stats plot error: {e_stats}")
            traceback.print_exc()
    # ... rest of post-simulation messages ...


# --- SCRIPT ENTRY POINT (Unchanged) --------------------------+
if __name__ == "__main__":
    print("Executing script (organism_v3.py)...")
    try:
        run(settings)
    except KeyboardInterrupt:
        print("\n--- Interrupted by user (Ctrl+C) ---")
        simulation_stopped_early = True
    except Exception as e:
        print("\n--- Unhandled error ---")
        traceback.print_exc()
        simulation_stopped_early = True
    finally:
        print("\nFinal cleanup: Closing Matplotlib figures.")
        plt.ioff()
        plt.close("all")
        print("Cleanup complete.")
# --- END ----------------------------------------------------------------------+
