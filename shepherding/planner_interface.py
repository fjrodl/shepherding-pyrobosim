"""
planner_interface.py
--------------------
Thin wrapper around planner backends (pyperplan or POPF) that solves
 a PDDL planning problem and
returns the plan as a list of parsed action dicts.

Each action dict has the form:
  {
      "name": "drive-flock",          # action name
      "args": ["z_2_2", "z_2_3", ...] # ground arguments
  }
"""

import re
import shlex
import shutil
import subprocess


def solve(domain_path, problem_path, backend="pyperplan", popf_command="popf", timeout_s=None):
    """
    Solve a planning task using a selected backend and return a parsed plan.

    Parameters
    ----------
    domain_path : str
        Path to the PDDL domain file.
    problem_path : str
        Path to the PDDL problem file.

    Returns
    -------
    list[dict]  — ordered plan, empty if no plan found.

    Raises
    ------
    RuntimeError
        If the selected planner backend is unavailable or fails.
    """
    backend = backend.lower().strip()

    if backend == "pyperplan":
        return _solve_with_pyperplan(domain_path, problem_path)

    if backend == "popf":
        return _solve_with_popf(
            domain_path,
            problem_path,
            popf_command=popf_command,
            timeout_s=timeout_s,
        )

    raise RuntimeError(
        f"Unsupported planner backend '{backend}'. Use 'pyperplan' or 'popf'."
    )


def _solve_with_pyperplan(domain_path, problem_path):
    """Solve using pyperplan's Python API."""
    if not _pyperplan_available():
        raise RuntimeError("pyperplan is not installed. Run: pip install pyperplan")

    from pyperplan.planner import HEURISTICS, SEARCHES, search_plan

    try:
        solution = search_plan(
            domain_path,
            problem_path,
            SEARCHES["astar"],
            HEURISTICS["hadd"],
        )
    except Exception as exc:
        raise RuntimeError(f"pyperplan error: {exc}") from exc

    if solution is None:
        return []

    return _parse_solution(solution)


def _solve_with_popf(domain_path, problem_path, popf_command="popf", timeout_s=None):
    """Solve using an external POPF executable and parse action lines."""
    cmd_parts = shlex.split(popf_command)
    if not cmd_parts:
        raise RuntimeError("POPF command is empty.")

    popf_bin = cmd_parts[0]
    if shutil.which(popf_bin) is None:
        raise RuntimeError(
            f"POPF executable '{popf_bin}' not found on PATH. "
            "Install POPF or set planner.popf_command in YAML."
        )

    cmd = cmd_parts + [domain_path, problem_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"POPF timed out after {timeout_s} seconds") from exc

    output = f"{result.stdout}\n{result.stderr}"
    plan = _parse_popf_output(output)

    if result.returncode != 0 and not plan:
        raise RuntimeError(
            f"POPF error (exit {result.returncode}):\n{result.stderr}\n{result.stdout}"
        )

    return plan


def solve_from_files(domain_path, problem_path):
    """Alias kept for clarity when calling from experiments."""
    return solve(domain_path, problem_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pyperplan_available():
    try:
        import pyperplan  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_solution(solution):
    """
    Parse pyperplan's internal operator solution list into action dicts.

    Operators are rendered as strings such as:
        "(move-robot z_0_0 z_0_1)"
    """
    plan = []
    pattern = re.compile(r"^\s*\(([a-zA-Z0-9_\-]+)((?:\s+[a-zA-Z0-9_\-]+)*)\)\s*$")

    for op in solution:
        m = pattern.match(getattr(op, "name", ""))
        if m:
            name = m.group(1).lower()
            args = m.group(2).split() if m.group(2).strip() else []
            plan.append({"name": name, "args": args})

    return plan


def _parse_popf_output(output):
    """
    Parse POPF textual output into action dicts.

    Expected action-line examples:
      0.000: (move-robot z_0_0 z_0_1) [1.000]
      (drive-flock z_1_0 z_2_0 z_3_0 z_4_4)
    """
    plan = []
    timed_pattern = re.compile(r"^\s*[0-9]+(?:\.[0-9]+)?:\s*\(([^)]+)\)")
    plain_pattern = re.compile(r"^\s*\(([^)]+)\)\s*$")

    for line in output.splitlines():
        m = timed_pattern.match(line)
        if not m:
            m = plain_pattern.match(line)
        if not m:
            continue

        parts = m.group(1).strip().split()
        if not parts:
            continue

        plan.append({
            "name": parts[0].lower(),
            "args": [a.lower() for a in parts[1:]],
        })

    return plan
