"""
env.py

Environment utilities and the EnvManager class to create and manage isolated
venvs for models with difficult dependencies.
"""

import os
import random
import numpy as np
import torch
import subprocess, json, sys, os
from typing import Dict, Any
from pathlib import Path
import tempfile

from proto_language.utils.compute import determine_visible_devices

from logging import getLogger

logger = getLogger(__name__)

def seed_everything(seed: int):
    """
    Seeds everything

    Args:
        seed (int): The seed to use
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnvManager:
    """
    Class that manages the creation and management of venvs for different models.
    """

    def __init__(self, model_name: str, refresh: bool = False):
        """
        Initialize a EnvManager for a given model.

        Args:
            model_name: The name of the model to manage
            refresh: Whether to refresh the venv if it already exists
        """
        self.model_name = self._determine_valid_model_name(model_name)
        self.env_path = Path(".venvs") / f"{model_name}_env"
        self.setup_script = self._find_setup_script(model_name)

        # auto-create/refresh venv if needed
        if (
            not self.env_path.exists()
            or refresh
            or not self._is_venv_setup_successful()
        ):
            if self.env_path.exists() and not self._is_venv_setup_successful():
                logger.info(
                    f"Venv for {model_name} exists but setup was not successful. Attempting to recreate..."
                )
            else:
                logger.info(f"Setting up venv for {model_name}...")
            self._create_env()

        else:
            logger.debug(
                f"Venv for {model_name} already exists and setup was successful at {self.env_path}"
            )

    def _determine_valid_model_name(self, model_name: str):
        """
        Helper function to determine if a provided model is a model that contains
        a 'standalone' subdirectory.
        """
        # Get project root (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "proto_language" / "tools" / "models"
        available_models = []

        # Find all directories that contain a "standalone" subdirectory
        for item in models_dir.rglob("*"):
            if item.is_dir() and (item / "standalone").exists():
                # Use just the final directory name
                available_models.append(item.name)

        if model_name not in available_models:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models: {available_models}"
            )
        return model_name

    def _find_setup_script(self, model_name: str):
        """
        Helper function to find the setup.sh script for a given model
        """
        # Get project root (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "proto_language" / "tools" / "models"

        # Find the directory that contains a "standalone" subdirectory for this model
        for item in models_dir.rglob("*"):
            if item.is_dir() and (item / "standalone").exists():
                # Match on the final directory name
                if item.name == model_name:
                    return item / "standalone" / "setup.sh"

        raise ValueError(f"Could not find standalone directory for model: {model_name}")

    def _is_venv_setup_successful(self):
        """
        Helper function to check if the venv setup was successful by reading STATUS.txt
        """
        status_file = self.env_path / "STATUS.txt"
        if not status_file.exists():
            return False

        try:
            with open(status_file, "r") as f:
                status = f.read().strip()
            return status == "SUCCESS"
        except Exception:
            return False

    def _create_env(self):
        """
        Helper function to create a venv for a given model using the setup.sh script.
        """
        import datetime

        status_file = self.env_path / "STATUS.txt"

        try:
            # create venv
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.env_path)], check=True
            )

            # Check if setup script exists
            if not self.setup_script.exists():
                error_msg = f"No setup.sh script found for {self.model_name} at {self.setup_script}"
                with open(status_file, "w") as f:
                    f.write(
                        f"FAILED\n\nError: {error_msg}\nTimestamp: {datetime.datetime.now()}\n"
                    )
                raise ValueError(error_msg)

            # Make setup script executable
            subprocess.run(["chmod", "+x", str(self.setup_script)], check=True)

            # Set up environment variables for the setup script
            env = os.environ.copy()
            env["VENV_PATH"] = str(self.env_path)
            env["PYTHON_EXE"] = str(self.env_path / "bin" / "python")
            env["PIP_EXE"] = str(self.env_path / "bin" / "pip")

            # Run the setup script from its directory with venv activated
            activate_script = self.env_path.absolute() / "bin" / "activate"
            result = subprocess.run(
                ["bash", "-c", f"source {activate_script} && {self.setup_script}"],
                cwd=self.setup_script.parent,
                env=env,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero return code
            )

            # Create STATUS.txt based on result
            if result.returncode == 0:
                # Success
                with open(status_file, "w") as f:
                    f.write("SUCCESS")
            else:
                # Failure
                with open(status_file, "w") as f:
                    f.write(f"FAILED\n\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Command: {self.setup_script}\n")
                    f.write(f"Timestamp: {datetime.datetime.now()}\n\n")
                    if result.stdout:
                        f.write(f"STDOUT:\n{result.stdout}\n\n")
                    if result.stderr:
                        f.write(f"STDERR:\n{result.stderr}\n")

                raise subprocess.CalledProcessError(
                    result.returncode,
                    str(self.setup_script),
                    result.stdout,
                    result.stderr,
                )

        except Exception as e:
            # Handle any other exceptions (like venv creation failure)
            if not status_file.exists():
                with open(status_file, "w") as f:
                    f.write(f"FAILED\n\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {datetime.datetime.now()}\n")
            raise

    def call_standalone_script_in_venv(
        self,
        script_path: Path,
        input_dict: Dict[str, Any],
        device: str = "cuda:0",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Helper function to call a standalone script utilizing the current venv

        Args:
            script_path: The path to the script to call with the venv activated
            input_dict: A dictionary of json-serializable input parameters to
                pass to the script
            device: The device to utilize for script execution for the model
            verbose: Whether to print verbose output from the script

        Returns:
            The output of the script as a dictionary
        """

        with tempfile.TemporaryDirectory() as temp_dir:

            # Create a temp input file location
            input_json_path = Path(temp_dir) / "input.json"
            with open(input_json_path, "w") as f:
                json.dump(input_dict, f)

            # Create a temp output file location
            output_json_path = Path(temp_dir) / "output.json"

            # Set up environment variables
            env = os.environ.copy()

            # Set CUDA_VISIBLE_DEVICES to the specified device number
            env["CUDA_VISIBLE_DEVICES"] = determine_visible_devices(device=device)

            try:
                if verbose:
                    logger.debug(
                        f"Running {script_path} with input: {input_dict} and device: {device}"
                    )
                subprocess.run(
                    [
                        str(self.env_path.absolute() / "bin" / "python"),
                        str(script_path),
                        str(input_json_path),
                        str(output_json_path),
                    ],
                    env=env,
                    text=True,
                    check=True,
                    stdout=None if verbose else subprocess.PIPE,
                    stderr=None if verbose else subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running {script_path}: {e}")
                if e.stderr:
                    logger.error(f"STDERR: {e.stderr}")
                if e.stdout:
                    logger.error(f"STDOUT: {e.stdout}")
                raise e

            # Read in the output file
            with open(output_json_path, "r") as f:
                output_data = json.load(f)

            return output_data
