"""
problem_generator.py
--------------------
Generates a PDDL problem file from an abstract state produced by
WorldAbstraction.compute().
"""

import os


def generate_problem(abstract_state, domain_name="shepherding",
                     problem_name="shepherding-problem",
                     output_path="pddl/problem.pddl"):
    """
    Build a PDDL problem string from the abstract state dict and write it
    to *output_path*.

    Parameters
    ----------
    abstract_state : dict
        Output of WorldAbstraction.compute().
    domain_name : str
    problem_name : str
    output_path : str
        File path where the problem.pddl will be written.

    Returns
    -------
    str
        The full PDDL problem text.
    """
    lines = []
    lines.append(f"(define (problem {problem_name})")
    lines.append(f"  (:domain {domain_name})")

    # --- Objects ---
    zone_objs = " ".join(abstract_state["all_zones"])
    lines.append("  (:objects")
    lines.append(f"    {zone_objs} - zone")

    outlier_ids = abstract_state["outlier_ids"]
    if outlier_ids:
        sheep_objs = " ".join(f"sheep{i}" for i in outlier_ids)
        lines.append(f"    {sheep_objs} - sheep")

    lines.append("  )")

    # --- Init ---
    lines.append("  (:init")

    # Robot location
    lines.append(f"    (robot-at-zone {abstract_state['robot_zone']})")

    # Flock location
    lines.append(f"    (flock-at-zone {abstract_state['flock_zone']})")

    # Flock state flag (avoid negative preconditions in the domain)
    if abstract_state["flock_dispersed"]:
        lines.append("    (flock-dispersed)")
    else:
        lines.append("    (flock-compact)")

    # Outlier sheep zones
    for sheep_id, zone in abstract_state["sheep_zones"].items():
        lines.append(f"    (sheep-at-zone sheep{sheep_id} {zone})")

    # Adjacency (bidirectional)
    for z1, z2 in abstract_state["adjacent_pairs"]:
        lines.append(f"    (zone-adjacent {z1} {z2})")
        lines.append(f"    (zone-adjacent {z2} {z1})")

    # Goal zone marker
    lines.append(f"    (zone-is-goal {abstract_state['goal_zone']})")

    # Behind-flock spatial relationships
    for robot_z, flock_z, goal_z in abstract_state["behind_triples"]:
        lines.append(f"    (behind-flock {robot_z} {flock_z} {goal_z})")

    lines.append("  )")

    # --- Goal ---
    lines.append("  (:goal (flock-at-goal))")
    lines.append(")")

    problem_text = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(problem_text)

    return problem_text
