import os
import sys
from typing import Tuple, Dict
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

DEFAULT_ENV_FILE = "src/.env"
if os.path.exists(DEFAULT_ENV_FILE):
    lg.warning(
        f"Loading environment variables from {DEFAULT_ENV_FILE}. Will override variables like HYPERSTACK_API_KEY."
    )
    load_dotenv("src/.env", override=True)

LOCAL_LOGS_DIR = "./logs/local"
os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)


def deploy_vm(
    environment_name: str,
    vm_name: str,
    flavor_name: str,
    use_fixed_ips: bool = False,
    image_name: str = None,
) -> Tuple[VMManager, HyperstackManager]:
    """
    Deploys a VM using HyperstackManager and initializes VMManager.

    Args:
        environment_name (str): The environment name.
        vm_name (str): The VM name.
        flavor_name (str): The flavor name.
        use_fixed_ips (bool): Whether to use fixed IPs. Defaults to False.
        image_name (str): The image name. Defaults to None.

    Returns:
        Tuple[VMManager, HyperstackManager]: The VMManager and HyperstackManager instances.
    """
    hyperstack_manager = HyperstackManager(environment_name=environment_name)
    vm_id = hyperstack_manager.get_or_create_vm(
        vm_name=vm_name,
        flavor_name=flavor_name,
        enable_public_ip=not use_fixed_ips,
        image_name=image_name,
    )
    hyperstack_manager.wait_for_vm_active(vm_id)
    ip_key = "fixed_ip" if use_fixed_ips else "floating_ip"
    hyperstack_manager.wait_for_ips_available(vm_id, ip_key)

    vm_details = hyperstack_manager.get_vm_from_id(vm_id)
    lg.debug(vm_details)
    private_key, private_key_path = hyperstack_manager.get_private_ssh_key(
        vm_details["keypair"]["name"]
    )

    # Verify SSH connectivity
    SSHManager.verify_connection(vm_details[ip_key], private_key)

    # Initialize VM manager
    vm_manager = VMManager(vm_details, private_key_path, use_fixed_ips=use_fixed_ips)

    return vm_manager, hyperstack_manager


def setup_vm(vm_manager: VMManager):
    """
    Sets up the VM by copying necessary files and running setup commands.

    Args:
        vm_manager (VMManager): The VMManager instance.
    """
    vm_manager.copy_to_remote("src/", "src/", exclude=[".git", ".venv", "data"])
    vm_manager.run_command(get_setup_vm_cmd())
    vm_manager.copy_to_remote(
        "data/datasets/", "data/datasets/", exclude=[".git", ".venv", "data"]
    )
    vm_manager.copy_to_remote("configs/", "configs/", exclude=[".git", ".venv", "data"])


def run_finetuning(vm_manager: VMManager, finetuning_config: Dict, config_path: str):
    """
    Runs the finetuning process on the VM.

    Args:
        vm_manager (VMManager): The VMManager instance.
        finetuning_config (Dict): The finetuning configuration.
        config_path (str): The path to the configuration file.
    """
    lg.info("Running finetuning...")
    if finetuning_config.get("use_host_hf_token", False):
        lg.debug("Using host HF_TOKEN from environment")
        env_vars = {"HF_TOKEN": os.getenv("HF_TOKEN")}
    else:
        env_vars = None

    finetuning_command = get_run_finetuning_command(
        config_path=config_path, environment_vars_dict=env_vars
    )
    vm_manager.run_command(finetuning_command)
    os.makedirs("results", exist_ok=True)
    vm_manager.copy_from_remote("results/", "results/")


def run_inference(vm_manager: VMManager, inference_config: Dict, config_path: str):
    """
    Runs the inference process on the VM.

    Args:
        vm_manager (VMManager): The VMManager instance.
        inference_config (Dict): The inference configuration.
        config_path (str): The path to the configuration file.
    """
    lg.info("Running inference...")
    input_path = (
        inference_config.get("mode_configs", {}).get("csv", {}).get("input_path")
    )
    if input_path:
        vm_manager.copy_to_remote(input_path, input_path)

    vm_manager.run_command(get_inference_command(config_path))

    vm_manager.copy_from_remote("results/", "results/")
    lg.debug("Inference results copied to local")


def deploy_chatbot_app(
    vm_manager: VMManager, app_deployment_config: Dict, run_name: str
):
    """
    Deploys the chatbot application on the VM.

    Args:
        vm_manager (VMManager): The VMManager instance.
        app_deployment_config (Dict): The app deployment configuration.
        run_name (str): The run name.
    """
    lg.info("Running app deployment...")
    if app_deployment_config.get("use_host_hf_token", False):
        lg.debug("Using host HF_TOKEN from environment")
        env_vars = {"HF_TOKEN": os.getenv("HF_TOKEN")}
    else:
        env_vars = None

    model_run_dir = app_deployment_config.get("model_run_dir")
    if model_run_dir.lower() == "latest":
        run_dir = get_latest_run_dir(run_name)
        model_to_deploy = get_model_path_from_run_dir(run_dir)
    elif model_run_dir:
        model_to_deploy = get_model_run_dir(model_run_dir)
    else:
        model_to_deploy = None
    base_model = app_deployment_config.get("base_model")

    lg.debug("Copying app files to VM...")
    if model_to_deploy:
        if not model_to_deploy.endswith("/"):
            model_to_deploy += "/"
        model_to_deploy = os.path.join("./", model_to_deploy)
        vm_manager.create_dir(model_to_deploy)
        vm_manager.copy_to_remote(model_to_deploy, model_to_deploy)
    vm_manager.copy_to_remote("frontend/", "frontend/")
    vm_manager.copy_to_remote("docker-compose.yaml", "docker-compose.yaml")
    vm_manager.copy_to_remote("nginx.conf", "nginx.conf")

    deploy_cmd = get_deploy_chatbot_app_command(
        model_to_deploy=model_to_deploy,
        base_model=base_model,
        environment_vars_dict=env_vars,
        additional_vllm_args=app_deployment_config.get("additional_vllm_args"),
    )
    vm_manager.run_command(deploy_cmd)


def deploy_and_setup_vm(deployment_config: Dict) -> Tuple[VMManager, HyperstackManager]:
    """
    Deploys and sets up the VM based on the deployment configuration.

    Args:
        deployment_config (Dict): The deployment configuration.

    Returns:
        Tuple[VMManager, HyperstackManager]: The VMManager and HyperstackManager instances.
    """
    if deployment_config.get("enabled", False):
        lg.info("Deploying VM...")
        vm_manager, hyperstack_manager = deploy_vm(
            deployment_config["environment_name"],
            deployment_config["vm_name"],
            deployment_config["flavor_name"],
            image_name=deployment_config.get("image_name", None),
        )
        setup_vm(vm_manager)
    else:
        lg.info("Using existing VM...")
        hyperstack_manager = HyperstackManager(
            environment_name=deployment_config["environment_name"]
        )
        vm_details = hyperstack_manager.get_vm_from_name(deployment_config["vm_name"])
        _, private_key_path = hyperstack_manager.get_private_ssh_key(
            vm_details["keypair"]["name"]
        )
        vm_manager = VMManager(vm_details, private_key_path)

    return vm_manager, hyperstack_manager


def main(config_path: str):
    """
    Main function to run the deployment, finetuning, inference, and app deployment processes.

    Args:
        config_path (str): The path to the configuration file.
    """
    config = read_json(config_path)

    run_name = config["run_name"]
    if run_name.lower() == "latest":
        raise ValueError("run_name cannot be 'latest'. This is a reserved keyword.")

    deployment_config = config.get("hyperstack_deployment")
    finetuning_config = config.get("finetuning")
    inference_config = config.get("inference")
    app_deployment_config = config.get("app_deployment")

    # Deploy and set up VM
    vm_manager, hyperstack_manager = deploy_and_setup_vm(deployment_config)

    # Run finetuning
    if finetuning_config.get("enabled", False):
        run_finetuning(vm_manager, finetuning_config, config_path)

    # Run inference
    if inference_config.get("enabled", False):
        run_inference(vm_manager, inference_config, config_path)

    # Deploy chatbot app
    if app_deployment_config.get("enabled", False):
        deploy_chatbot_app(vm_manager, app_deployment_config, run_name)

    # Finally delete VM if required
    if deployment_config.get("delete_after_completion", False):
        lg.info("Deleting VM...")
        hyperstack_manager.delete_vm(vm_manager.vm_details["id"])
    else:
        lg.info("VM will not be deleted because 'delete_after_completion' is False")


if __name__ == "__main__":
    lg.add(LOCAL_LOGS_DIR + "/{time}.log")
    DEFAULT_CONFIG_PATH = "./configs/config_000.json"
    config_paths = sys.argv[1:]

    if len(config_paths) == 0:
        lg.warning(
            f"No config paths provided. "
            f"Using default config path: {DEFAULT_CONFIG_PATH}"
        )
        config_paths = [DEFAULT_CONFIG_PATH]

    for path in config_paths:
        lg.info(f"Running with config: {path}")
        main(path)
