import textwrap

from loguru import logger as lg


def clean_cmd(cmd):
    # Remove leading newlines
    cmd = cmd.lstrip("\n")

    # Remove indententation created by Python layout
    cmd = textwrap.dedent(cmd)

    return cmd


def add_environment_vars_cmd(environment_vars_dict, cmd=None):
    if not cmd:
        cmd = ""
    cmd += "echo 'Setting environment variables...'\n"
    for key, value in environment_vars_dict.items():
        cmd += f"export {key}={value}\n"

    return clean_cmd(cmd)


def get_setup_vm_cmd():
    cmd = """
    echo "Setting up the VM..."

    # Set function
    retry_command() {
    local retries=10
    local wait_time=10
    local count=0

    until "$@"; do
        exit_code=$?
        count=$((count + 1))
        if [ $count -lt $retries ]; then
        echo "Command failed. Attempt $count/$retries. Retrying in $wait_time seconds..."
        sleep $wait_time
        else
        echo "Command failed after $retries attempts. Exiting with code $exit_code."
        return $exit_code
        fi
    done
    }

    # Install libraries (with retry because sometimes there is an unattended upgrade going on)
    echo "Installing libraries"
    retry_command sudo apt-get update
    retry_command sudo apt-get install -y python3-pip

    # Make ubuntu the owner of /ephemeral
    echo "Changing ownership of /ephemeral to ubuntu..."
    sudo chown -R ubuntu:ubuntu /ephemeral

    # Install uv 
    echo "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    # Create the virtual environment
    echo "Creating virtual environment..."
    uv venv
    source .venv/bin/activate

    # Install dependencies
    echo "Installing dependencies..."
    uv pip install -r src/requirements.txt

    # Create Huggingface cache directory
    echo "Creating Huggingface cache directory..."
    mkdir -p /ephemeral/huggingface/ 2>/dev/null
    export HF_HOME=/ephemeral/huggingface/

    # Create directories
    echo "Creating directoroes..."
    mkdir -p data/datasets 2>/dev/null

    echo "SETUP COMPLETED"
    """

    return clean_cmd(cmd)


def get_run_finetuning_command(
    config_path,
    environment_vars_dict=None,
):
    cmd = ""
    if environment_vars_dict:
        cmd = add_environment_vars_cmd(environment_vars_dict)

    cmd += f"""
    echo "Running finetuning..."

    # Closing any chatbot docker containers
    docker kill ubuntu-nginx-1
    docker kill ubuntu-streamlit_app-1
    docker kill ubuntu-llm-model-1

    # Wait 5 seconds for the containers to stop
    sleep 5

    export PYTHONPATH=$PYTHONPATH:/home/ubuntu/src
    .venv/bin/python3 src/finetuning/finetuning_peft.py --config_path {config_path}"""

    return clean_cmd(cmd)


def get_inference_command(config_path):
    cmd = f"""
    echo "Running inference..."
    export PYTHONPATH=$PYTHONPATH:/home/ubuntu/src
    .venv/bin/python3 src/inference/inference.py --config_path {config_path}
    """

    return clean_cmd(cmd)


def generate_env_var_update_command(var_name, var_value):
    cmd = f"""
    # Define the desired {var_name} value
    NEW_{var_name}={var_value}

    # Path to the shell config file (adjust if needed)
    BASHRC_PATH="$HOME/.bashrc"

    # Check if {var_name} is already defined in the file
    if grep -q "^export {var_name}=" "$BASHRC_PATH"; then
        # Overwrite the existing line
        sed -i "s/^export {var_name}=.*/export {var_name}=$NEW_{var_name}/" "$BASHRC_PATH"
    else
        # Append the new {var_name} to the end of the file
        echo "export {var_name}=$NEW_{var_name}" >> "$BASHRC_PATH"
    fi

    # Apply changes for the current session
    source "$BASHRC_PATH"
    """

    return clean_cmd(cmd)


def get_deploy_chatbot_app_command(
    model_to_deploy=None,
    base_model=None,
    environment_vars_dict=None,
):
    cmd = "echo 'Deploying chatbot app...'"
    if environment_vars_dict:
        cmd += add_environment_vars_cmd(environment_vars_dict)

    cmd += clean_cmd(f"""
    # Set some variables
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
    export MODEL_TO_DEPLOY={model_to_deploy}
    export BASE_MODEL={base_model}

    # Print the variables
    echo "Number of GPUs: $NUM_GPUS"
    echo "Model to deploy: $MODEL_TO_DEPLOY"
    echo "Base model: $BASE_MODEL"

    echo "Running docker build..."
    docker compose -f docker-compose.yaml build
    echo "Running docker compose up..."
    docker compose -f docker-compose.yaml up -d
    echo "Chatbot app deployed successfully"
    """)

    return clean_cmd(cmd)
