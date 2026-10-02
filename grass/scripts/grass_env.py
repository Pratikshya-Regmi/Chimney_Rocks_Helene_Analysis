#!/usr/bin/env python3
"""
grass_env.py — one place to open a GRASS session.

Mirrors the environment setup used in the project notebooks so scripts behave
identically whether run from Jupyter, a terminal, or Claude Code.

Usage:
    from grass_env import grass_session, log
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    log("starting stage 1", run="coreg")
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- config
GRASSBIN = os.environ.get("GRASSBIN", "/usr/local/bin/grass85")
GISDBASE = os.environ.get(
    "GISDBASE",
    "/home/pregmi3/Desktop/helene_nov_2025/data_now_17_2025/Notebooks",
)
LOG_DIR = Path(os.environ.get("HELENE_LOGS", "results/logs"))

PROJECTS = {
    "watershed": "DEM_generation",
    "lure": "DEM_generation_lure",
}


def grass_session(project="DEM_generation", mapset="PERMANENT",
                  gisdbase=None, grassbin=None):
    """Initialise GRASS and return (grass.script, grass.jupyter).

    Accepts either a real project name or one of the aliases in PROJECTS.
    """
    gisdbase = gisdbase or GISDBASE
    grassbin = grassbin or GRASSBIN
    project = PROJECTS.get(project, project)

    if not Path(gisdbase).exists():
        sys.exit(f"GISDBASE not found: {gisdbase}\n"
                 f"Set the GISDBASE environment variable.")

    py_path = subprocess.check_output(
        [grassbin, "--config", "python_path"], text=True).strip()
    dist_dir = str(Path(py_path).parents[1])

    os.environ["GISBASE"] = dist_dir
    os.environ["PATH"] = f"{dist_dir}/bin:{dist_dir}/scripts:" + os.environ["PATH"]
    os.environ["PYTHONPATH"] = py_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{dist_dir}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    sys.path.insert(0, py_path)

    import grass.script as gs
    import grass.jupyter as gj

    gj.init(gisdbase, location=project, mapset=mapset, grass_path=grassbin)
    print(f"GRASS session: {project}/{mapset}")
    return gs, gj


def log(message, run="run", echo=True):
    """Append a timestamped line to results/logs/<run>_<date>.log.

    Every script must log its inputs, parameters and outputs -- if a number
    changes between runs, the logs are what explain why.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = LOG_DIR / f"{run}_{datetime.now():%Y%m%d}.log"
    line = f"[{stamp}] {message}"
    with open(path, "a") as fh:
        fh.write(line + "\n")
    if echo:
        print(line)


def region_summary(gs):
    """Return the current computational region as a dict, and log it."""
    reg = gs.parse_command("g.region", flags="g")
    log(f"region: n={reg['n']} s={reg['s']} e={reg['e']} w={reg['w']} "
        f"nsres={reg['nsres']} ewres={reg['ewres']} "
        f"rows={reg['rows']} cols={reg['cols']}", run="region")
    return reg


def require_maps(gs, names, mapset=None):
    """Abort with a clear message if any required map is missing.

    Prevents a script from silently producing numbers off the wrong surface.
    """
    missing = []
    for n in names:
        found = gs.find_file(name=n, element="cell", mapset=mapset)
        if not found or not found.get("name"):
            missing.append(n)
    if missing:
        sys.exit("Missing required raster(s): " + ", ".join(missing) +
                 "\nRun scripts/00_inventory.py to see what is available.")
    return True
