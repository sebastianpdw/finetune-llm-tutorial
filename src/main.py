import os
import sys
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from loguru import logger as lg

from deployment.commands import (
    get_deploy_chatbot_app_command,
    get_inference_command,
    get_setup_vm_cmd,
    get_run_finetuning_command,
)
from deployment.hyperstack import HyperstackManager
from deployment.vms import SSHManager, VMManager
from utils.io import (
    get_latest_run_dir,
    get_model_path_from_run_dir,
    get_model_run_dir,
    read_json,
)

# Constants
DEFAULT_ENV_FILE = "src/.env"
LOCAL_LOGS_DIR = "./logs/local"

# Load environment variables if .env file exists
if os.path.exists(DEFAULT_ENV_FILE):
    lg.warning(
        f"Loading environment variables from {DEFAULT_ENV_FILE}. "
        f"Will override variables like HYPERSTACK_API_KEY."
    )
    load_dotenv(DEFAULT_ENV_FILE, override=True)

# Create logs directory
os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)


def deploy_vm(
    environment_name: str,
    vm_name: str,
    flavor_name: str,
    use_fixed_ips: bool = False,
    image_name: Optional[str] = None,
    keypair_name: Optional[str] = None,
) -> Tuple[VMManager, HyperstackManager]:
    """
    Deploys a VM using HyperstackManager and initializes VMManager.

    Args:
        environment_name (str): The environment name.
        vm_name (str): The VM name.
        flavor_name (str): The flavor name.
        use_fixed_ips (bool, optional): Whether to use fixed IPs. Defaults to False.
        image_name (Optional[str], optional): The image name. Defaults to None.
        keypair_name (Optional[str], optional): The keypair name. Defaults to None.

    Returns:
        Tuple[VMManager, HyperstackManager]: A tuple containing the VMManager and HyperstackManager instances.

    Raises:
        Exception: If VM deployment fails or SSH connection cannot be established.
    """
    try:
        hyperstack_manager = HyperstackManager(environment_name=environment_name)
        vm_id = hyperstack_manager.get_or_create_vm(
            vm_name=vm_name,
            flavor_name=flavor_name,
            enable_public_ip=not use_fixed_ips,
            image_name=image_name,
            keypair_name=keypair_name,
        )
        hyperstack_manager.wait_for_vm_active(vm_id)
        ip_key = "fixed_ip" if use_fixed_ips else "floating_ip"
        hyperstack_manager.wait_for_ips_available(vm_id, ip_key)

        vm_details = hyperstack_manager.get_vm_from_id(vm_id)
        lg.debug(f"VM details: {vm_details}")
        private_key, private_key_path = hyperstack_manager.get_private_ssh_key(
            vm_details["keypair"]["name"]
        )

        # Verify SSH connectivity
        SSHManager.verify_connection(vm_details[ip_key], private_key)

        # Initialize VM manager
        vm_manager = VMManager(
            vm_details, private_key_path, use_fixed_ips=use_fixed_ips
        )

        return vm_manager, hyperstack_manager
    except Exception as e:
        lg.error(f"Failed to deploy VM: {str(e)}")
        raise


def setup_vm(vm_manager: VMManager) -> None:
    """
    Sets up the VM by copying necessary files and running setup commands.

    Args:
        vm_manager (VMManager): The VMManager instance.

    Raises:
        Exception: If VM setup fails.
    """
    try:
        # Copy source code, excluding unnecessary directories
        vm_manager.copy_to_remote("src/", "src/", exclude=[".git", ".venv", "data"])

        # Run setup commands
        vm_manager.run_command(get_setup_vm_cmd())

        # Copy datasets and configs
        vm_manager.copy_to_remote(
            "data/datasets/", "data/datasets/", exclude=[".git", ".venv", "data"]
        )
        vm_manager.copy_to_remote(
            "configs/", "configs/", exclude=[".git", ".venv", "data"]
        )
    except Exception as e:
        lg.error(f"Failed to set up VM: {str(e)}")
        raise


def run_finetuning(
    vm_manager: VMManager, finetuning_config: Dict, config_path: str
) -> None:
    """
    Runs the finetuning process on the VM.

    Args:
        vm_manager (VMManager): The VMManager instance.
        finetuning_config (Dict): The finetuning configuration.
        config_path (str): The path to the configuration file.

    Raises:
        Exception: If finetuning fails.
    """
    try:
        lg.info("Running finetuning...")
        env_vars = None

        if finetuning_config.get("use_host_hf_token", False):
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                lg.warning(
                    "HF_TOKEN environment variable not found but use_host_hf_token is True"
                )
            else:
                lg.debug("Using host HF_TOKEN from environment")
                env_vars = {"HF_TOKEN": hf_token}

        finetuning_command = get_run_finetuning_command(
            config_path=config_path, environment_vars_dict=env_vars
        )
        vm_manager.run_command(finetuning_command)

        # Create results directory and copy results from remote
        os.makedirs("results", exist_ok=True)
        vm_manager.copy_from_remote("results/", "results/")
    except Exception as e:
        lg.error(f"Failed to run finetuning: {str(e)}")
        raise


def run_inference(
    vm_manager: VMManager, inference_config: Dict, config_path: str
) -> None:
    """
    Runs the inference process on the VM.

    Args:
        vm_manager (VMManager): The VMManager instance.
        inference_config (Dict): The inference configuration.
        config_path (str): The path to the configuration file.

    Raises:
        Exception: If inference fails.
    """
    try:
        lg.info("Running inference...")
        input_path = (
            inference_config.get("mode_configs", {}).get("csv", {}).get("input_path")
        )
        if input_path:
            vm_manager.copy_to_remote(input_path, input_path)

        vm_manager.run_command(get_inference_command(config_path))

        vm_manager.copy_from_remote("results/", "results/")
        lg.debug("Inference results copied to local")
    except Exception as e:
        lg.error(f"Failed to run inference: {str(e)}")
        raise


def deploy_chatbot_app(
    vm_manager: VMManager, app_deployment_config: Dict, run_name: str
) -> None:
    """
    Deploys the chatbot application on the VM.

    Args:
        vm_manager (VMManager): The VMManager instance.
        app_deployment_config (Dict): The app deployment configuration.
        run_name (str): The run name.

    Raises:
        Exception: If app deployment fails.
    """
    try:
        lg.info("Running app deployment...")
        env_vars = None

        if app_deployment_config.get("use_host_hf_token", False):
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                lg.warning(
                    "HF_TOKEN environment variable not found but use_host_hf_token is True"
                )
            else:
                lg.debug("Using host HF_TOKEN from environment")
                env_vars = {"HF_TOKEN": hf_token}

        # Determine model to deploy
        model_to_deploy = None
        model_run_dir = app_deployment_config.get("model_run_dir")
        if model_run_dir:
            if model_run_dir.lower() == "latest":
                run_dir = get_latest_run_dir(run_name)
                model_to_deploy = get_model_path_from_run_dir(run_dir)
            else:
                model_to_deploy = get_model_run_dir(model_run_dir)

        base_model = app_deployment_config.get("base_model")

        lg.debug("Copying app files to VM...")
        # Copy model files if needed
        if model_to_deploy:
            if not model_to_deploy.endswith("/"):
                model_to_deploy += "/"
            model_to_deploy = os.path.join("./", model_to_deploy)
            vm_manager.create_dir(model_to_deploy)
            vm_manager.copy_to_remote(model_to_deploy, model_to_deploy)

        # Copy frontend and configuration files
        vm_manager.copy_to_remote("frontend/", "frontend/")
        vm_manager.copy_to_remote("docker-compose.yaml", "docker-compose.yaml")
        vm_manager.copy_to_remote("nginx.conf", "nginx.conf")

        # Deploy the app
        deploy_cmd = get_deploy_chatbot_app_command(
            model_to_deploy=model_to_deploy,
            base_model=base_model,
            environment_vars_dict=env_vars,
            additional_vllm_args=app_deployment_config.get("additional_vllm_args"),
        )
        vm_manager.run_command(deploy_cmd)
    except Exception as e:
        lg.error(f"Failed to deploy chatbot app: {str(e)}")
        raise


def deploy_and_setup_vm(deployment_config: Dict) -> Tuple[VMManager, HyperstackManager]:
    """
    Deploys and sets up the VM based on the deployment configuration.

    Args:
        deployment_config (Dict): The deployment configuration.

    Returns:
        Tuple[VMManager, HyperstackManager]: A tuple containing the VMManager and HyperstackManager instances.

    Raises:
        KeyError: If required configuration keys are missing.
        Exception: If VM deployment or setup fails.
    """
    try:
        if deployment_config.get("enabled", False):
            lg.info("Deploying VM...")
            # Check for required configuration keys
            required_keys = ["environment_name", "vm_name", "flavor_name"]
            for key in required_keys:
                if key not in deployment_config:
                    raise KeyError(f"Required key '{key}' missing in deployment_config")

            vm_manager, hyperstack_manager = deploy_vm(
                deployment_config["environment_name"],
                deployment_config["vm_name"],
                deployment_config["flavor_name"],
                use_fixed_ips=deployment_config.get("use_fixed_ips", False),
                image_name=deployment_config.get("image_name"),
                keypair_name=deployment_config.get("keypair_name"),
            )
            setup_vm(vm_manager)
        else:
            lg.info("Using existing VM...")
            # Check for required configuration keys
            required_keys = ["environment_name", "vm_name"]
            for key in required_keys:
                if key not in deployment_config:
                    raise KeyError(f"Required key '{key}' missing in deployment_config")

            hyperstack_manager = HyperstackManager(
                environment_name=deployment_config["environment_name"]
            )
            vm_details = hyperstack_manager.get_vm_from_name(
                deployment_config["vm_name"]
            )
            _, private_key_path = hyperstack_manager.get_private_ssh_key(
                vm_details["keypair"]["name"]
            )
            vm_manager = VMManager(
                vm_details,
                private_key_path,
                use_fixed_ips=deployment_config.get("use_fixed_ips", False),
            )

        return vm_manager, hyperstack_manager
    except Exception as e:
        lg.error(f"Failed to deploy and set up VM: {str(e)}")
        raise


def main(config_path: str) -> None:
    """
    Main function to run the deployment, finetuning, inference, and app deployment processes.

    Args:
        config_path (str): The path to the configuration file.

    Raises:
        ValueError: If run_name is 'latest' or if the config file cannot be read.
        Exception: If any step in the process fails.
    """
    try:
        config = read_json(config_path)
        if not config:
            raise ValueError(f"Failed to read or parse config file: {config_path}")

        run_name = config.get("run_name")
        if not run_name:
            raise ValueError("run_name not specified in config")
        if run_name.lower() == "latest":
            raise ValueError("run_name cannot be 'latest'. This is a reserved keyword.")

        deployment_config = config.get("hyperstack_deployment", {})
        finetuning_config = config.get("finetuning", {})
        inference_config = config.get("inference", {})
        app_deployment_config = config.get("app_deployment", {})

        # Deploy and set up VM
        vm_manager, hyperstack_manager = deploy_and_setup_vm(deployment_config)

        # Run finetuning if enabled
        if finetuning_config.get("enabled", False):
            run_finetuning(vm_manager, finetuning_config, config_path)

        # Run inference if enabled
        if inference_config.get("enabled", False):
            run_inference(vm_manager, inference_config, config_path)

        # Deploy chatbot app if enabled
        if app_deployment_config.get("enabled", False):
            deploy_chatbot_app(vm_manager, app_deployment_config, run_name)

        # Delete VM if required
        if deployment_config.get("delete_after_completion", False):
            lg.info("Deleting VM...")
            hyperstack_manager.delete_vm(vm_manager.vm_details["id"])
        else:
            lg.info("VM will not be deleted because 'delete_after_completion' is False")

    except Exception as e:
        lg.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    # Set up logging
    lg.add(os.path.join(LOCAL_LOGS_DIR, "{time}.log"))

    # Define default config path
    DEFAULT_CONFIG_PATH = "./configs/config_000.json"
    config_paths = sys.argv[1:]

    if not config_paths:
        lg.warning(
            f"No config paths provided. "
            f"Using default config path: {DEFAULT_CONFIG_PATH}"
        )
        config_paths = [DEFAULT_CONFIG_PATH]

    # Process each config file
    for path in config_paths:
        lg.info(f"Running with config: {path}")
        try:
            main(path)
        except Exception as e:
            lg.error(f"Failed to process config {path}: {str(e)}")
            # Continue with next config instead of stopping execution
            continue
