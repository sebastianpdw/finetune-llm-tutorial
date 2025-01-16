import logging as lg
import os
import subprocess
import time
from io import StringIO
from typing import Any, Dict, List, Optional

import paramiko
from fabric import Connection
from loguru import logger as lg

from deployment.hyperstack import HyperstackManager


class SSHManager:
    """
    Manages SSH connections and operations to remote instances.
    """

    @staticmethod
    def verify_connection(
        vm_ip: str,
        private_key_str: str,
        max_retries: int = 60,
        sleep_time: int = 10,
    ) -> None:
        """
        Verify SSH connection to instance with retry mechanism.

        Args:
            vm_ip (str): IP address of the remote instance
            private_key_str (str): SSH private key string
            max_retries (int): Maximum number of retry attempts
            sleep_time (int): Time to wait between retries in seconds

        Raises:
            Exception: If unable to establish SSH connection after max retries
        """
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        for attempt in range(max_retries):
            try:
                # Convert private key string to RSA key object
                pkey = paramiko.Ed25519Key.from_private_key(StringIO(private_key_str))
                lg.debug(f"Attempting SSH connection to {vm_ip} ...")

                # Attempt connection
                ssh_client.connect(
                    vm_ip,
                    username="ubuntu",
                    pkey=pkey,
                    timeout=10,
                )

                # Verify connection with test command
                stdin, stdout, stderr = ssh_client.exec_command(
                    "echo 'Verifying connection on remote instance...'"
                )
                lg.debug(f"Connection verified: {stdout.read().decode('utf-8')}")
                ssh_client.close()
                return

            except Exception as e:
                lg.error(f"Connection attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    lg.debug(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        raise Exception("Failed to establish SSH connection after multiple retries")


class VMManager:
    """
    Manages operations on one VM instance

    Attributes:
        vm_details (Dict[str, Any]): Dictionary containing VM details
        private_key_path (str): SSH private key path
        use_fixed_ips (bool): Flag to use fixed/private IP for SSH connections
        user (str): SSH username
        max_retries (int): Maximum number of retry attempts
        sleep_time (int): Time to wait between retries in seconds
        logs_dir (str): Directory for storing operation logs
    """

    def __init__(
        self,
        vm_details: Dict[str, Any],
        private_key_path: str,
        use_fixed_ips: bool = False,
        user: str = "ubuntu",
        base_logs_dir: str = "logs/ssh",
    ):
        """Initialize VMsManager with required parameters."""
        self.vm_details = vm_details
        self.use_fixed_ips = use_fixed_ips
        self.private_key_path = private_key_path
        self.user = user

        self.ip_key = "fixed_ip" if use_fixed_ips else "floating_ip"

        # Create logs directory with timestamp
        self.logs_dir = os.path.join(base_logs_dir, f"{int(time.time())}")

    def copy_from_remote(
        self,
        remote_path: str,
        local_path: Optional[str] = None,
        max_retries: int = 3,
        sleep_time: int = 5,
    ) -> bool:
        """
        Copy files from a remote host to local machine using rsync over SSH.

        This method creates a temporary SSH key file and uses rsync to securely copy
        files from the remote host. It handles directory creation and cleanup of
        temporary files.

        Args:
            vm_ip (str): Remote host IP address
            remote_path (str): Path to remote file or directory to copy
            local_path (Optional[str]): Destination path on local machine.
                                      If None, uses remote_path.

        Returns:
            bool: True if copy was successful, False otherwise

        Note:
            - Uses rsync over SSH for efficient file transfer
            - Creates temporary key file for SSH authentication
            - Automatically creates local directories as needed
            - Implements retry mechanism for reliability
        """
        if local_path is None:
            local_path = remote_path
        lg.debug(
            f"Copying files from {self.vm_details[self.ip_key]}:{remote_path} to {local_path}"
        )
        vm_ip = self.vm_details[self.ip_key]

        success = False
        retries = 0
        while retries < max_retries:
            try:
                # Ensure local directory exists
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Configure extended timeout for warnings
                os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "60000"

                # Construct rsync command with SSH options
                rsync_command = [
                    "rsync",
                    "-az",  # archive mode + compression
                    "-e",
                    f"ssh -i {self.private_key_path} -o StrictHostKeyChecking=no",
                    f"ubuntu@{vm_ip}:{remote_path}",
                    local_path,
                ]

                # Execute rsync command
                result = subprocess.run(
                    rsync_command,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise exception on non-zero exit
                )

                if result.returncode == 0:
                    lg.debug(f"Successfully copied files from {vm_ip}:{remote_path}")
                    lg.debug(f"Files copied to: {local_path}")
                    if result.stdout:
                        lg.debug(f"rsync output: {result.stdout}")
                    success = True
                    break
                else:
                    lg.error(f"rsync failed with exit code {result.returncode}")
                    lg.error(f"Error output: {result.stderr}")
                    retries += 1

            except subprocess.SubprocessError as e:
                lg.error(f"Subprocess error on attempt {retries + 1}: {str(e)}")
                retries += 1
            except Exception as e:
                lg.error(f"Unexpected error during file copy: {str(e)}")
                break

            if retries < max_retries:
                lg.debug(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return success

    def copy_to_remote(
        self,
        local_filepath: str,
        remote_filepath: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        max_retries: int = 3,
        sleep_time: int = 5,
    ) -> bool:
        """
        Copy a local file or directory to a remote VM using rsync over SSH.

        This method creates a temporary SSH key file and uses rsync to securely copy
        files to the remote host. It handles both single files and directories.

        Args:
            local_filepath (str): Path to the file/directory on local machine
            remote_filepath (Optional[str]): Destination path on remote VM.
                                        If None, uses local_filepath
            exclude (Optional[List[str]]): List of patterns to exclude from transfer.
                                        Example: ['*.pyc', '__pycache__']
            max_retries (int): Maximum number of retry attempts
            sleep_time (int): Time to wait between retries in seconds

        Returns:
            bool: True if copy was successful, False otherwise

        Note:
            - Uses rsync over SSH for efficient file transfer
            - Creates temporary key file for SSH authentication
            - Automatically creates remote directories as needed
            - Implements retry mechanism for reliability
        """
        vm_ip = self.vm_details[self.ip_key]
        if remote_filepath is None:
            lg.warning(f"No remote path provided. Using local path: {local_filepath}")
            remote_filepath = local_filepath

        success = False
        retries = 0
        while retries < max_retries:
            try:
                # Configure extended timeout for warnings
                os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "60000"

                # Construct rsync command with SSH options
                rsync_command = [
                    "rsync",
                    "-az",  # archive mode + compression
                    "-e",
                    f"ssh -i {self.private_key_path} -o StrictHostKeyChecking=no",
                ]

                # Add exclude patterns if provided
                if exclude:
                    for pattern in exclude:
                        rsync_command.extend(["--exclude", pattern])

                # Add source and destination paths
                rsync_command.extend(
                    [
                        local_filepath,
                        f"ubuntu@{vm_ip}:{remote_filepath}",
                    ]
                )

                # Execute rsync command
                result = subprocess.run(
                    rsync_command,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise exception on non-zero exit
                )

                if result.returncode == 0:
                    lg.debug(
                        f"Successfully copied {local_filepath} to {vm_ip}:{remote_filepath}"
                    )
                    if result.stdout:
                        lg.debug(f"rsync output: {result.stdout}")
                    success = True
                    break
                else:
                    lg.error(f"rsync failed with exit code {result.returncode}")
                    lg.error(f"Error output: {result.stderr}")
                    retries += 1

            except subprocess.SubprocessError as e:
                lg.error(f"Subprocess error on attempt {retries + 1}: {str(e)}")
                retries += 1
            except Exception as e:
                lg.error(f"Unexpected error during file copy: {str(e)}")
                break

            if retries < max_retries:
                lg.debug(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return success

    def create_dir(
        self, remote_dir: str, max_retries: int = 3, sleep_time: int = 5
    ) -> bool:
        """
        Create a directory on the remote VM.

        Args:
            remote_dir (str): Path to the directory to create
            max_retries (int): Maximum number of retry attempts
            sleep_time (int): Time to wait between retries in seconds

        Returns:
            bool: True if directory was created successfully, False otherwise
        """
        vm_ip = self.vm_details[self.ip_key]

        success = False
        retries = 0
        while retries < max_retries:
            try:
                # Setup SSH connection
                pkey = self._get_private_key()
                connection = Connection(
                    host=vm_ip, user=self.user, connect_kwargs={"pkey": pkey}
                )

                with connection as conn:
                    lg.debug(f"Creating directory on {vm_ip}: {remote_dir}")
                    conn.run(f"mkdir -p {remote_dir}")

                lg.debug(f"Directory created on {vm_ip}: {remote_dir}")
                success = True
                break

            except Exception as e:
                lg.error(f"Error creating directory on {vm_ip}: {str(e)}")
                retries += 1

            if retries < max_retries:
                lg.debug(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return success

    def _get_private_key(self) -> paramiko.Ed25519Key:
        """
        Get the private key for SSH connection.

        Returns:
            paramiko.Ed25519Key: Private key object for SSH connection
        """

        with open(self.private_key_path, "r") as key_file:
            private_key_str = key_file.read()

        return paramiko.Ed25519Key.from_private_key(StringIO(private_key_str))

    def _get_vm_id(self, hyperstack_manager: HyperstackManager) -> str:
        """
        Get the ID of the VM from the HyperstackManager instance.

        Args:
            hyperstack_manager (HyperstackManager): HyperstackManager instance

        Returns:
            str: ID of the VM
        """
        all_vms = hyperstack_manager.list_vms()
        vm_ids = [
            vm["id"]
            for vm in all_vms
            if vm[self.ip_key] == self.vm_details[self.ip_key]
        ]
        if len(vm_ids) == 0:
            raise ValueError(
                f"VM ID not found in HyperstackManager instance for VM {self.vm_details['name']}"
            )
        elif len(vm_ids) > 1:
            lg.warning(
                f"Multiple VMs found with IP {self.vm_details[self.ip_key]}. Using first VM ID."
            )
        vm_id = vm_ids[0]
        return vm_id

    def run_command(
        self,
        cmd: str,
        log_filepath: Optional[str] = None,
        copy_from_remote_path: Optional[str] = None,
        copy_to_local_path: Optional[str] = None,
        delete_vm_after_completion: bool = False,
        hyperstack_manager: Optional[HyperstackManager] = None,
        max_retries: int = 10,
        sleep_time: int = 30,
    ) -> Optional[Any]:
        """
        Execute a command on a remote VM and optionally copy results back.

        Args:
            vm_ip (str): IP address of the remote VM
            cmd (str): Command to execute on the remote VM
            log_filepath (Optional[str]): Path to log file for command output
            copy_from_remote_path (Optional[str]): Path on remote VM to copy from
            copy_to_local_path (Optional[str]): Local path to copy files to
            delete_vm_after_completion (bool): Flag to delete VM after command execution
            hyperstack_manager (Optional[HyperstackManager]): HyperstackManager instance (required for deletion)
            max_retries (int): Maximum number of retry attempts
            sleep_time (int): Time to wait between retries in seconds

        Returns:
            Optional[Any]: Result of the command execution if successful, None otherwise

        Note:
            Command output is logged to the specified log file.
            If remote and local paths are provided, files will be copied after command execution.
        """
        if log_filepath is None:
            os.makedirs(self.logs_dir, exist_ok=True)
            log_filepath = os.path.join(self.logs_dir, f"{int(time.time())}.log")
        if delete_vm_after_completion and not hyperstack_manager:
            raise ValueError(
                "HyperstackManager instance required for deleting VM after completion"
            )
        if delete_vm_after_completion and not copy_from_remote_path:
            lg.warning(
                "No 'copy_from_remote_path' provided but 'delete_vm_after_completion' is set to True. "
                "VM will be deleted without copying files before deletion."
            )

        # Setup SSH connection
        vm_ip = self.vm_details[self.ip_key]
        pkey = self._get_private_key()
        connection = Connection(
            host=vm_ip, user=self.user, connect_kwargs={"pkey": pkey}
        )

        result = None
        retries = 0
        while retries < max_retries:
            try:
                with connection as conn:
                    lg.debug(f"Executing command on {vm_ip}")
                    lg.debug(f"Log file: {log_filepath}")

                    # Execute command and log output
                    with open(log_filepath, "a+") as log_file:
                        result = conn.run(cmd, out_stream=log_file, err_stream=log_file)

                    if result.exited == 0:
                        lg.debug("Command executed successfully")
                    else:
                        lg.error(f"Command failed with exit code {result.exited}")
                    break

            except paramiko.SSHException as e:
                retries += 1
                error_msg = f"SSH error on attempt {retries}: {e}\n"
                lg.error(error_msg)
                time.sleep(sleep_time)

            except Exception as e:
                error_msg = f"Unexpected error: {e}\n"
                lg.error(error_msg)
                break

        # Copy files from remote if paths are provided
        copy_succeeded = True
        if copy_from_remote_path and copy_to_local_path:
            lg.debug(
                f"Copying files from {vm_ip}:{copy_from_remote_path} to {copy_to_local_path}"
            )
            copy_succeeded = self.copy_from_remote(
                copy_from_remote_path, local_path=copy_to_local_path
            )

        # Delete VM after completion if specified
        if delete_vm_after_completion and copy_succeeded:
            vm_id = self._get_vm_id(hyperstack_manager)
            lg.debug(f"Deleting VM {vm_id} after completion")
            hyperstack_manager.delete_vm(vm_id)
        elif delete_vm_after_completion and not copy_succeeded:
            vm_id = self._get_vm_id(hyperstack_manager)
            lg.warning(
                f"VM {vm_id} will be hibernated instead of deleted because file copy failed"
            )
            hyperstack_manager.hibernate_vm(vm_id)

        return result
