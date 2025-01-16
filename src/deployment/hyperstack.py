import os
import json
import time
from typing import List, Optional, Tuple
from pathlib import Path
import hyperstack
from hyperstack.rest import ApiException
from hyperstack.models import (
    CreateInstancesPayload,
    CreateSecurityRulePayload,
    ImportKeypairPayload,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from loguru import logger as lg

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union


DEFAULT_HYPERSTACK_API_URL = "https://infrahub-api.nexgencloud.com/v1"
DEFAULT_ENVIRONMENT_NAME = "default-CANADA-1"

DEFAULT_KEY_DIR = "keys"
DEFAULT_KEYPAIR_NAME = "finetuning-keypair"

DEFAULT_IMAGE_NAME = "Ubuntu Server 22.04 LTS R535 CUDA 12.2 with Docker"
DEFAULT_SECURITY_RULES = [
    CreateSecurityRulePayload(
        direction="ingress",
        ethertype="IPv4",
        protocol="tcp",
        remote_ip_prefix="0.0.0.0/0",
        port_range_min=22,
        port_range_max=22,
    ),
    CreateSecurityRulePayload(
        direction="ingress",
        ethertype="IPv6",
        protocol="tcp",
        remote_ip_prefix="::/0",
        port_range_min=22,
        port_range_max=22,
    ),
]


class HyperstackManager:
    """
    Manages Hyperstack operations for infrastructure deployment and destruction.
    Uses threading for concurrent operations.
    """

    def __init__(
        self,
        environment_name: str = None,
        key_dir=None,
    ):
        """
        Initialize HyperstackManager with API client.

        Args:
            environment_name (str): Name of the environment to deploy resources in
        """
        lg.debug(f"Initializing HyperstackManager with environment: {environment_name}")

        # Get default values if not provided
        if key_dir is None:
            key_dir = DEFAULT_KEY_DIR
        if environment_name is None:
            environment_name = DEFAULT_ENVIRONMENT_NAME
        if os.environ.get("HYPERSTACK_API_ADDRESS") is None:
            lg.warning(
                "No HYPERSTACK_API_ADDRESS environment variable found. "
                "Taking default from Hyperstack Python SDK"
            )

        self.environment_name = environment_name
        configuration = hyperstack.Configuration(
            host=os.environ.get("HYPERSTACK_API_ADDRESS"),
        )

        self.key_dir = Path(key_dir)

        try:
            api_key = os.environ["HYPERSTACK_API_KEY"]
            configuration.api_key["apiKey"] = api_key
            self.api_client = hyperstack.ApiClient(configuration)
            lg.debug("Successfully initialized Hyperstack API client")
        except KeyError:
            lg.error("API_KEY environment variable not found")
            raise ValueError("API_KEY environment variable must be set")

    def _generate_ssh_keypair(self, key_path: Path) -> Tuple[str, str]:
        """
        Generate a new SSH keypair and save to filesystem.

        Args:
            key_path (Path): Path to save the private key

        Returns:
            Tuple[str, str]: Tuple of (private key OpenSSH string, public key OpenSSH string)

        Raises:
            Exception: If key generation fails
        """
        lg.debug(f"Generating new SSH keypair at: {key_path}")
        # Generate private key
        private_key = ed25519.Ed25519PrivateKey.generate()

        # Generate public key
        public_key = private_key.public_key()

        # Serialize private key in OpenSSH format
        private_openssh = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        # Serialize public key in OpenSSH format
        public_openssh = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        ).decode("utf-8")

        try:
            # Save private key to file
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_text(private_openssh)
            key_path.chmod(0o600)  # Set proper permissions
            lg.debug(
                f"Successfully saved private key to {key_path} with permissions 600"
            )

            return private_openssh, public_openssh
        except Exception as e:
            lg.error(f"Failed to save keypair: {e}")
            raise

    def _get_or_create_keypair(self, keypair_name: str) -> Tuple[str, Path]:
        """
        Get existing keypair or create new one if it doesn't exist.

        Args:
            keypair_name (str): Name of the keypair to get or create
            key_dir (Optional[str]): Directory to store keypair files. Defaults to ~/.ssh/hyperstack/

        Returns:
            Tuple[str, Path]: Tuple of (keypair name, path to private key file)

        Raises:
            Exception: If keypair operations fail or if keypair exists in API but was not created by this script
        """
        key_path = self.key_dir / f"{keypair_name}.pem"

        with self.api_client as api_client:
            keypair_api = hyperstack.KeypairApi(api_client)

            # Check if keypair exists in API
            try:
                keypairs = keypair_api.list_key_pairs()
                existing_keypair = next(
                    (kp for kp in keypairs.keypairs if kp.name == keypair_name), None
                )
                if existing_keypair:
                    # If keypair exists in API, check for local key file
                    if key_path.exists():
                        lg.debug(
                            f"Found existing keypair in API and local key at {key_path}"
                        )
                        return keypair_name, key_path
                    else:
                        raise Exception(
                            f"Keypair '{keypair_name}' exists in API but no corresponding "
                            f"local key file found at {key_path}. This keypair may not have "
                            "been created through this script."
                        )
            except ApiException as e:
                raise Exception(f"Failed to list keypairs: {e}")

            # Create new keypair since it doesn't exist in API
            lg.debug(f"Checking for existing keypair at {key_path}")

            if key_path.exists():
                lg.debug("Found existing private key")
                # If local key exists but not in Hyperstack, import existing key
                try:
                    public_key = Path(str(key_path) + ".pub").read_text().strip()
                    lg.debug("Successfully loaded existing public key")
                except FileNotFoundError:
                    lg.warning("Public key missing, regenerating keypair")
                    # Regenerate keypair if public key is missing
                    _, public_key = self._generate_ssh_keypair(key_path)
            else:
                lg.debug("No existing keypair found, generating new one")
                # Generate new keypair
                _, public_key = self._generate_ssh_keypair(key_path)

            try:
                payload = ImportKeypairPayload(
                    name=keypair_name,
                    public_key=public_key,
                    environment_name=self.environment_name,
                )
                response = keypair_api.import_key_pair(payload)
                return response.keypair.name, key_path
            except ApiException as e:
                raise Exception(f"Failed to create keypair: {e}")

    def get_private_ssh_key(self, keypair_name=None) -> Tuple[str, str]:
        """
        Fetch private SSH key from local filesystem.

        Returns:
            str: Private SSH key
        """
        if keypair_name is None:
            keypair_name = DEFAULT_KEYPAIR_NAME

        lg.debug(f"Fetching private SSH key for keypair: {keypair_name}")
        private_key_path = self.key_dir / f"{keypair_name}.pem"

        try:
            private_key_str = private_key_path.read_text()

            return private_key_str, private_key_path
        except FileNotFoundError as e:
            lg.error(f"Private key not found: {e}")
            raise

    def list_vms(self) -> List[dict]:
        """
        List all virtual machines.

        Returns:
            List[dict]: List of VM details including name, id, and status

        Raises:
            Exception: If VM listing fails
        """
        # lg.debug("Fetching list of all virtual machines")
        with self.api_client as api_client:
            api_instance = hyperstack.VirtualMachineApi(api_client)
            try:
                response = api_instance.list_virtual_machines(
                    environment=self.environment_name
                )
                vm_list = [
                    {
                        "name": vm.name,
                        "id": vm.id,
                        "status": vm.status,
                        "floating_ip": vm.floating_ip,
                        "fixed_ip": vm.fixed_ip,
                    }
                    for vm in response.instances
                ]
                # lg.debug(f"Found {len(vm_list)} virtual machines")
                return vm_list
            except ApiException as e:
                raise Exception(f"Failed to list VMs: {e}")

    def get_vm_from_id(self, vm_id: str) -> dict:
        """
        Get details of a single virtual machine.

        Args:
            vm_id (str): ID of the virtual machine.

        Returns:
            dict: Details of the VM including name, id, status, and floating IP.

        Raises:
            Exception: If VM details retrieval fails.
        """
        # lg.debug(f"Fetching details for VM with ID: {vm_id}")
        with self.api_client as api_client:
            api_instance = hyperstack.VirtualMachineApi(api_client)
            try:
                response = api_instance.retrieve_virtual_machine_details(vm_id)
                return response.model_dump()["instance"]
            except ApiException as e:
                raise Exception(f"Failed to get VM details: {e}")

    def get_vm_from_name(self, vm_name: str) -> dict:
        """
        Get details of a single virtual machine.

        Args:
            vm_name (str): Name of the virtual machine.

        Returns:
            dict: Details of the VM including name, id, status, and floating IP.

        Raises:
            Exception: If VM details retrieval fails.
        """
        lg.debug(f"Fetching details for VM with name: {vm_name}")

        vms = self.list_vms()
        vm_ids = [vm["id"] for vm in vms if vm["name"] == vm_name]
        if len(vm_ids) == 0:
            lg.warning(f"No VMs found with name {vm_name}")
            return None
        elif len(vm_ids) > 1:
            lg.warning(f"Found multiple VMs with name {vm_name}, returning first one")
        vm_id = vm_ids[0]

        with self.api_client as api_client:
            api_instance = hyperstack.VirtualMachineApi(api_client)
            try:
                response = api_instance.retrieve_virtual_machine_details(vm_id)
                return response.model_dump()["instance"]
            except ApiException as e:
                raise Exception(f"Failed to get VM details: {e}")

    def get_vm_details_dict(self, vm_ids: List[str]) -> List[dict]:
        """
        Get details of multiple virtual machines.

        Args:
            vm_ids (List[str]): List of VM IDs to get details for.

        Returns:
            List[dict]: List of VM details including name, id, status, and floating IP.

        Raises:
            Exception: If VM details retrieval fails.
        """
        lg.debug(f"Fetching details for {len(vm_ids)} virtual machines")
        vm_details = [self.get_vm_from_id(vm_id) for vm_id in vm_ids]

        vm_details_dict = {}
        for vm in vm_details:
            vm_details_dict[vm["name"]] = {
                "flavor": vm["flavor"]["name"],
                "fixed_ip": vm["fixed_ip"],
                "floating_ip": vm["floating_ip"],
                "keypair_name": vm["keypair"]["name"],
            }

        return vm_details_dict

    def get_or_create_vm(
        self,
        vm_name: str,
        flavor_name: str,
        keypair_name: str = None,
        image_name: str = None,
        enable_public_ip: bool = True,
        labels: Optional[List[str]] = None,
        security_rules: Optional[List[CreateSecurityRulePayload]] = None,
        max_retries: int = 10,
        retry_interval: int = 10,
    ) -> str:
        """
        Create a single VM with specified parameters.

        Args:
            vm_name (str): Name of the VM to create.
            flavor_name (str): Flavor name for the VM.
            keypair_name (str): Name of the keypair to use.
            image_name (str): Name of the VM image to use.
            enable_public_ip (bool): Flag to enable public IP for the VM.
            labels (Optional[List[str]]): Labels to attach to the VM.
            security_rules (List[CreateSecurityRulePayload]): Security rules for the VM.
            max_retries (int): Maximum number of retries for VM creation.
            retry_interval (int): Interval between retries in seconds.

        Returns:
            str: ID of the created VM.

        Raises:
            Exception: If VM creation fails.
        """
        lg.debug(f"Getting or creating VM with flavor: {flavor_name}")
        if keypair_name is None:
            keypair_name = DEFAULT_KEYPAIR_NAME
        if image_name is None:
            image_name = DEFAULT_IMAGE_NAME
        if security_rules is None:
            security_rules = DEFAULT_SECURITY_RULES

        lg.debug(f"Getting or creating keypair: {keypair_name}")
        self._get_or_create_keypair(keypair_name)

        lg.debug(f"Checking if VM already exists with name: '{vm_name}'")
        existing_vm = self.get_vm_from_name(vm_name)
        if existing_vm:
            lg.debug(f"VM with name '{vm_name}' already exists")
            return existing_vm["id"]

        # Create VM payload
        payload = CreateInstancesPayload(
            name=vm_name,
            environment_name=self.environment_name,
            key_name=keypair_name,
            image_name=image_name,
            flavor_name=flavor_name,
            count=1,
            assign_floating_ip=enable_public_ip,
            labels=labels or [],
            security_rules=security_rules,
        )
        with self.api_client as api_client:
            api_instance = hyperstack.VirtualMachineApi(api_client)
            for attempt in range(max_retries):
                try:
                    response = api_instance.create_virtual_machines(payload)
                    vm_id = response.instances[0].id
                    lg.debug(f"Successfully created VM with ID: {vm_id}")
                    return vm_id
                except ApiException as e:
                    body_json = json.loads(e.body)
                    lg.error(
                        f"Attempt {attempt + 1} failed: ({e.status} {e.reason}): {body_json.get('message')}"
                    )
                    if attempt < max_retries - 1:
                        lg.debug(f"Retrying in {retry_interval} seconds...")
                        time.sleep(retry_interval)
                    else:
                        raise Exception(
                            f"VM creation failed after {max_retries} attempts"
                        )

    def wait_for_vm_active(
        self, vm_id: str, max_status_retries: int = 60, retry_interval: int = 10
    ) -> bool:
        """
        Wait for a VM to become active.

        Args:
            vm_id (str): ID of the virtual machine.
            max_status_retries (int): Maximum number of retries for checking VM status.
            retry_interval (int): Interval between retries in seconds.

        Returns:
            bool: True if the VM becomes active, False otherwise.
        """
        for status_attempt in range(max_status_retries):
            details = self.get_vm_from_id(vm_id)
            status = details["status"]

            if status == "ACTIVE":
                lg.debug(f"VM {vm_id} successfully created and ACTIVE")
                return True
            elif status == "ERROR":
                lg.error(f"VM {vm_id} entered ERROR state")
                return False

            lg.debug(f"VM {vm_id} status: {status}, waiting...")
            time.sleep(retry_interval)

        return False

    def get_or_create_and_wait_for_vm(
        self,
        vm_name: str,
        flavor_name: str,
        keypair_name: str = None,
        image_name: str = None,
        enable_public_ip: bool = True,
        labels: Optional[List[str]] = None,
        security_rules: Optional[List[CreateSecurityRulePayload]] = None,
        max_creation_retries: int = 3,
        max_status_retries: int = 60,
        retry_interval: int = 10,
    ) -> Optional[str]:
        """
        Create a single VM and wait for it to become active, with retries.
        """
        # Set default values
        if keypair_name is None:
            keypair_name = DEFAULT_KEYPAIR_NAME
        if image_name is None:
            image_name = DEFAULT_IMAGE_NAME
        if security_rules is None:
            security_rules = DEFAULT_SECURITY_RULES

        # Attempt to create VM with retries
        for creation_attempt in range(max_creation_retries):
            try:
                vm_id = self.get_or_create_vm(
                    vm_name=vm_name,
                    flavor_name=flavor_name,
                    keypair_name=keypair_name,
                    image_name=image_name,
                    enable_public_ip=enable_public_ip,
                    labels=labels,
                    security_rules=security_rules,
                )

                # Wait for VM to become active
                if self.wait_for_vm_active(vm_id, max_status_retries, retry_interval):
                    return vm_id
                else:
                    lg.error(
                        f"VM {vm_id} failed to become active, retrying creation..."
                    )

                # Clean up failed VM before retrying
                if creation_attempt < max_creation_retries - 1:
                    try:
                        self.delete_vm(vm_id)
                    except Exception as e:
                        lg.error(f"Failed to delete VM {vm_id} during cleanup: {e}")
            except Exception as e:
                lg.error(f"VM creation attempt {creation_attempt + 1} failed: {e}")

        lg.error(f"Failed to create VM after {max_creation_retries} attempts")
        return None

    def wait_for_ips_available(
        self,
        vm_id_or_ids: Union[str, int, List[str]],
        ip_key: str,
        vm_details_dict: Optional[dict] = None,
        max_retries: int = 10,
        retry_interval: int = 20,
    ):
        if isinstance(vm_id_or_ids, str) or isinstance(vm_id_or_ids, int):
            vm_ids = [vm_id_or_ids]
        elif isinstance(vm_id_or_ids, list):
            vm_ids = vm_id_or_ids
        else:
            raise ValueError("'vm_id_or_ids' must be a string or a list of strings")

        # Get VM details if not passed
        if vm_details_dict is None:
            vm_details_dict = self.get_vm_details_dict(vm_ids)

        # Check if all ips are available
        for _ in range(max_retries):
            # all should be filled and not 127.0.0.1 (loopback IP)
            all_ips_valid = all(
                [
                    vm_details[ip_key] != "127.0.0.1" and vm_details[ip_key] is not None
                    for vm_details in vm_details_dict.values()
                ]
            )
            if all_ips_valid:
                lg.debug("All VM IP addresses are available")
                break
            lg.debug("Waiting for all VMs to get IP addresses...")
            time.sleep(retry_interval)
            vm_details_dict = self.get_vm_details_dict(vm_ids)

        return vm_details_dict

    def get_or_create_multiple_vms(
        self,
        vm_names: List[str],
        flavor_names: List[str],
        keypair_name: str = None,
        image_name: str = None,
        enable_public_ip: bool = True,
        labels: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Create multiple VMs concurrently using threading.

        Args:
            vm_names (List[str]): List of VM names to create
            flavor_names (List[str]): List of flavor names for the VMs
            keypair_name (str): Name of the keypair to use
            image_name (str): Name of the VM image to use
            enable_public_ip (bool): Flag to enable public IP for the VMs
            labels (Optional[List[str]]): Labels to attach to the VMs

        Returns:
            List[str]: List of successfully created VM IDs
        """
        if keypair_name is None:
            keypair_name = DEFAULT_KEYPAIR_NAME
        if image_name is None:
            image_name = DEFAULT_IMAGE_NAME

        lg.debug(f"Creating {len(flavor_names)} VMs concurrently using threading")

        # Ensure keypair exists
        keypair_name, key_path = self._get_or_create_keypair(keypair_name)

        # Define default security rules for SSH access
        security_rules = [
            CreateSecurityRulePayload(
                direction="ingress",
                ethertype="IPv4",
                protocol="tcp",
                remote_ip_prefix="0.0.0.0/0",
                port_range_min=22,
                port_range_max=22,
            ),
            CreateSecurityRulePayload(
                direction="ingress",
                ethertype="IPv6",
                protocol="tcp",
                remote_ip_prefix="::/0",
                port_range_min=22,
                port_range_max=22,
            ),
        ]

        successful_vm_ids = []

        max_workers = len(vm_names)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all VM creation tasks
            future_to_flavor = {
                executor.submit(
                    self.get_or_create_and_wait_for_vm,
                    vm_name=vm_name,
                    flavor_name=flavor,
                    keypair_name=keypair_name,
                    image_name=image_name,
                    enable_public_ip=enable_public_ip,
                    labels=labels,
                    security_rules=security_rules,
                ): (vm_name, flavor)
                for vm_name, flavor in zip(vm_names, flavor_names)
            }

            # Collect results as they complete
            for future in as_completed(future_to_flavor):
                flavor = future_to_flavor[future]
                pending_vms = len(future_to_flavor) - len(successful_vm_ids) - 1
                try:
                    vm_id = future.result()
                    if vm_id:
                        successful_vm_ids.append(vm_id)
                        lg.debug(
                            f"Successfully created VM with flavor {flavor} [{pending_vms}/{len(future_to_flavor)} pending]"
                        )
                    else:
                        lg.error(
                            f"Failed to create VM with flavor {flavor} [{pending_vms}/{len(future_to_flavor)} pending]"
                        )
                except Exception as e:
                    lg.error(
                        f"Exception creating VM with flavor {flavor}: {e} [{pending_vms}/{len(future_to_flavor)} pending]"
                    )

        lg.debug(
            f"Successfully created {len(successful_vm_ids)} out of {len(flavor_names)} VMs"
        )
        return successful_vm_ids

    def delete_vm(self, vm_id: str):
        """
        Delete a virtual machine.

        Args:
            vm_id (str): ID of the VM to delete
        """
        lg.debug(f"Deleting VM with ID: {vm_id}")
        with self.api_client as api_client:
            api_instance = hyperstack.VirtualMachineApi(api_client)
            try:
                api_instance.delete_virtual_machine(vm_id)
                lg.debug(f"Successfully deleted VM {vm_id}")
            except ApiException as e:
                lg.error(f"Failed to delete VM {vm_id}: {e}")
                raise

    def delete_multiple_vms(self, vm_ids: List[str]) -> List[str]:
        """
        Delete multiple VMs concurrently using threading.

        Args:
            vm_ids (List[str]): List of VM IDs to delete

        Returns:
            List[str]: List of successfully deleted VM IDs
        """
        lg.debug(f"Deleting {len(vm_ids)} VMs concurrently")
        successful_deletions = []

        max_workers = len(vm_ids)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all deletion tasks
            future_to_vm = {
                executor.submit(self.delete_vm, vm_id): vm_id for vm_id in vm_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_vm):
                vm_id = future_to_vm[future]
                try:
                    future.result()  # This will raise any exceptions that occurred
                    successful_deletions.append(vm_id)
                    lg.debug(f"Successfully deleted VM {vm_id}")
                except Exception as e:
                    lg.error(f"Failed to delete VM {vm_id}: {e}")

        lg.debug(
            f"Successfully deleted {len(successful_deletions)} out of {len(vm_ids)} VMs"
        )
        return successful_deletions

    def hibernate_vm(self, vm_id: str):
        """
        Hibernate a virtual machine.

        Args:
            vm_id (str): ID of the VM to hibernate
        """
        lg.debug(f"Hibernating VM with ID: {vm_id}")
        with self.api_client as api_client:
            api_instance = hyperstack.VirtualMachineApi(api_client)
            try:
                api_instance.hibernate_virtual_machine(vm_id)
                lg.debug(f"Successfully hibernated VM {vm_id}")
            except ApiException as e:
                lg.error(f"Failed to hibernate VM {vm_id}: {e}")
                raise
