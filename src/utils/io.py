import os
import json
from typing import Dict, List


def read_json(json_path: str) -> Dict:
    """
    Read a JSON file and return its contents as a dictionary.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Dict: Contents of the JSON file.
    """
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    return json_dict


def get_run_directories(run_dir: str) -> List[str]:
    """
    Get a list of run directories in the specified directory.

    Args:
        run_dir (str): The directory containing the run directories.

    Returns:
        List[str]: List of run directories.
    """
    if not os.path.exists(run_dir):
        raise ValueError(f"Directory {run_dir} does not exist.")

    return [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]


def filter_run_directories(
    run_dirs: List[str], run_base_name: str, run_dir="results/runs"
) -> List[str]:
    """
    Filter run directories that start with the specified base name.

    Args:
        run_dirs (List[str]): List of run directories.
        run_base_name (str): The base name to filter run directories.

    Returns:
        List[str]: Filtered list of run directories.
    """
    relevant_run_dirs = [d for d in run_dirs if run_base_name in d]
    for d in relevant_run_dirs:
        if not os.path.exists(os.path.join(run_dir, d, "model")):
            relevant_run_dirs.remove(d)
    if len(relevant_run_dirs) == 0:
        raise ValueError(f"No run directories found for run_base_name: {run_base_name}")
    return relevant_run_dirs


def get_latest_run_dir(run_base_name: str, run_dir: str = "results/runs") -> str:
    """
    Get the latest run directory.

    Args:
        run_base_name (str): The base name of the run directories.
        run_dir (str, optional): The directory containing the run directories. Defaults to "results/runs".

    Returns:
        str: The path to the latest run directory.

    Raises:
        ValueError: If no run directories are found for the specified base name.
    """
    run_dirs = get_run_directories(run_dir)
    relevant_run_dirs = filter_run_directories(run_dirs, run_base_name)
    try:
        latest_run_dir = max(relevant_run_dirs)
        return os.path.join(run_dir, latest_run_dir)
    except ValueError:
        raise ValueError(f"No run directories found for run_base_name: {run_base_name}")


def get_model_path_from_run_dir(run_dir: str) -> str:
    """
    Get the model path from the specified run directory.

    Args:
        run_dir (str): The run directory.

    Returns:
        str: The path to the model directory.

    Raises:
        ValueError: If multiple directories are found in the model base path.
    """
    model_base_path = os.path.join(run_dir, "model")
    subdirs = os.listdir(model_base_path)
    if len(subdirs) > 1:
        raise ValueError(
            f"Multiple directories found in {model_base_path}. Expected only one."
        )
    model_path = os.path.join(model_base_path, subdirs[0])
    return model_path


def get_model_run_dir(run_base_dir: str) -> str:
    """
    Get the model path for the latest run directory with the specified run name.

    Args:
        run_base_dir (str): The base name of the run directory

    Returns:
        str: The path to the model directory.
    """
    run_dir = get_latest_run_dir(run_base_dir)
    model_path = get_model_path_from_run_dir(run_dir)
    return model_path
