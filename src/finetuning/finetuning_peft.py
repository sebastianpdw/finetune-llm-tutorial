# Used references:
# - https://huggingface.co/docs/transformers/en/preprocessing
# - https://huggingface.co/docs/transformers/en/training
# - https://huggingface.co/blog/peft
# - https://huggingface.co/docs/transformers/en/tasks/language_modeling
# - https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_with_additional_tokens.ipynb
# - https://github.com/tloen/alpaca-lora/blob/main/finetune.py

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from peft import get_peft_model, LoraConfig, TaskType, PromptTuningConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from loguru import logger as lg

from typing import Tuple, Any

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET_PATH = "./data/datasets/train.csv"

DEFAULT_PEFT_TYPE = "lora"
DEFAULT_PEFT_HYPER_PARAMS = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.2,
}

DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 20

DEFAULT_RUNS_BASE_DIR = "./results/runs"


def check_hyperparams(
    required_params: List[str],
    supported_params: List[str],
    hyperparams: Optional[Dict[str, any]],
) -> None:
    """Check hyperparameters for required and unsupported parameters.

    Args:
        required_params (List[str]): List of required hyperparameters.
        supported_params (List[str]): List of supported hyperparameters.
        hyperparams (Optional[Dict[str, any]]): Dictionary of hyperparameters to check.

    Raises:
        ValueError: If any required hyperparameters are missing.
    """
    if hyperparams is None:
        return

    unsupported_hyperparams = [
        param for param in hyperparams if param not in supported_params
    ]
    if unsupported_hyperparams:
        lg.warning(
            f"Unsupported hyperparameters for peft: {unsupported_hyperparams}. Ignoring ..."
        )

    missing_params = [param for param in required_params if param not in hyperparams]
    if missing_params:
        raise ValueError(f"Missing required hyperparameters for peft: {missing_params}")


class PeftTrainer:
    SUPPORTED_PEFT = ["lora", "prompt_tuning", "prefix_tuning"]

    def __init__(
        self,
        model_name_or_path,
        dataset_csv_path,
        output_dir,
        peft_type="lora",
        peft_hyperparams=None,
    ):
        lg.debug(f"Initializing PEFT trainer with model: {model_name_or_path}")
        self.model_name_or_path = model_name_or_path
        self.dataset_csv_path = dataset_csv_path
        self.peft_type = peft_type
        self.output_dir = output_dir

        self.trainer = None
        if peft_hyperparams is None:
            lg.debug(
                "PEFT hyperparameters not set. "
                f"Using default hyperparameters: {DEFAULT_PEFT_HYPER_PARAMS}"
            )
            peft_hyperparams = DEFAULT_PEFT_HYPER_PARAMS
        self.peft_configs = self._load_peft_configs(peft_type, peft_hyperparams)
        lg.debug(f"PEFT inference_configs: {self.peft_configs}")

        os.makedirs(self.output_dir, exist_ok=True)
        lg.debug(
            f"Initialized PEFT trainer, all relevant files will be written to: {self.output_dir}"
        )

    def _load_model_and_tokenizer(
        self, load_in_8bit: bool = False, load_in_4bit: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer with optional 8-bit or 4-bit precision.

        Args:
            load_in_8bit (bool): Whether to load the model in 8-bit precision.
            load_in_4bit (bool): Whether to load the model in 4-bit precision.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
        """
        if not self.peft_configs:
            raise ValueError(
                "PEFT inference_configs not set. Call _load_peft_configs first."
            )
        lg.debug(
            f"Loading model and tokenizer with 8bit: {load_in_8bit}, 4bit: {load_in_4bit}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        model = get_peft_model(model, self.peft_configs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # Add padding token if not present
        if tokenizer.pad_token_id is None:
            lg.warning("Adding padding token to tokenizer (same as eos_token) ...")
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _load_peft_configs(self, peft: str, peft_hyperparams: Dict[str, Any]) -> Any:
        """Load PEFT configurations based on the specified PEFT type and hyperparameters.

        Args:
            peft (str): The type of PEFT model.
            peft_hyperparams (Dict[str, Any]): The hyperparameters for the PEFT model.

        Returns:
            Any: The PEFT configuration object.
        """
        lg.debug("Loading PEFT inference_configs ...")
        if peft not in self.SUPPORTED_PEFT:
            raise ValueError(
                f"PEFT model {peft} not supported. "
                f"Supported models: {self.SUPPORTED_PEFT}"
            )
        elif peft == "lora":
            required_hyperparams = []
            supported_hyperparams = ["r", "lora_alpha", "lora_dropout"]
            check_hyperparams(
                required_hyperparams, supported_hyperparams, peft_hyperparams
            )

            peft_configs = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=peft_hyperparams["r"],
                lora_alpha=peft_hyperparams["lora_alpha"],
                lora_dropout=peft_hyperparams["lora_dropout"],
            )
        elif peft == "prompt_tuning":
            lg.warning(
                "This functionality is experimental and has not been tested thoroughly."
            )
            required_hyperparams = ["prompt_tuning_init_text"]
            supported_hyperparams = [
                "num_virtual_tokens",
                "token_dim",
                "num_transformer_submodules",
                "num_attention_heads",
                "num_layers",
                "prompt_tuning_init_text",
            ]
            check_hyperparams(
                required_hyperparams, supported_hyperparams, peft_hyperparams
            )

            peft_configs = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                tokenizer_name_or_path=self.model_name_or_path,
                num_virtual_tokens=peft_hyperparams.get("num_virtual_tokens", 20),
                token_dim=peft_hyperparams.get("token_dim", 768),
                num_transformer_submodules=peft_hyperparams.get(
                    "num_transformer_submodules", 1
                ),
                num_attention_heads=peft_hyperparams.get("num_attention_heads", 12),
                num_layers=peft_hyperparams.get("num_layers", 12),
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text=peft_hyperparams["prompt_tuning_init_text"],
            )
        else:
            raise NotImplementedError(f"PEFT model {peft} not implemented")

        lg.debug(f"PEFT inference_configs loaded: {peft_configs}")
        return peft_configs

    def prepare_dataset(
        self,
        dataset: datasets.DatasetDict,
        tokenizer: AutoTokenizer,
        test_size: float = 0.2,
    ) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Prepare the dataset by formatting prompts and splitting into train and test sets.

        Args:
            dataset (datasets.DatasetDict): The dataset to prepare.
            tokenizer (AutoTokenizer): The tokenizer to use for formatting.
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
            Tuple[datasets.Dataset, datasets.Dataset]: The train and test datasets.
        """

        def formatting_prompts_func(examples):
            # from: https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=LjY75GoYUCB8
            inputs = examples["user"]
            outputs = examples["response"]

            texts = []
            for input, output in zip(inputs, outputs):
                chat = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": output},
                ]
                text = tokenizer.apply_chat_template(chat, tokenize=False)
                texts.append(text)
            return {
                "text": texts,
            }

        # Apply formatting function
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
        )

        # Split dataset
        train_test_split = dataset["train"].train_test_split(test_size=test_size)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        return train_dataset, test_dataset

    def load_and_tokenize_dataset(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
    ) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Load and tokenize the dataset.

        Args:
            model (AutoModelForCausalLM): The model to use for tokenization.
            tokenizer (AutoTokenizer): The tokenizer to use.

        Returns:
            Tuple[datasets.Dataset, datasets.Dataset]: The train and test datasets.
        """
        lg.debug("Loading and tokenizing datasets ...")
        trainable_params, all_param = model.get_nb_trainable_parameters()
        lg.debug(
            f"Trainable params: {trainable_params:,d}\n"
            f"all params: {all_param:,d}\n"
            f"trainable%: {100 * trainable_params / all_param}"
        )

        # Load datasets
        lg.debug("Loading dataset...")
        dataset = datasets.load_dataset("csv", data_files=self.dataset_csv_path)

        # Prepare dataset
        train_dataset, test_dataset = self.prepare_dataset(dataset, tokenizer)

        lg.debug(f"Train dataset: {train_dataset}")
        lg.debug(f"Test dataset: {test_dataset}")
        return train_dataset, test_dataset

    def setup_trainer(
        self,
        model: AutoModelForCausalLM,
        train_dataset: datasets.Dataset,
        test_dataset: datasets.Dataset,
        num_epochs: int = 20,
        early_stopping: bool = True,
        per_device_train_batch_size: int = 8,
    ) -> None:
        """Set up the trainer with the specified parameters.

        Args:
            model (AutoModelForCausalLM): The model to train.
            train_dataset (datasets.Dataset): The training dataset.
            test_dataset (datasets.Dataset): The evaluation dataset.
            num_epochs (int): The number of training epochs.
            early_stopping (bool): Whether to use early stopping.
            per_device_train_batch_size (int): The batch size per device for training.
        """
        lg.debug("Setting up trainer ...")
        output_dir = os.path.join(self.output_dir, "training_output")
        os.makedirs(output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            dataloader_drop_last=True,
            num_train_epochs=num_epochs,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            per_device_train_batch_size=per_device_train_batch_size,
            report_to=[],
        )

        if early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.0
            )
        else:
            early_stopping_callback = None

        self.trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[early_stopping_callback],
        )

    def train_and_evaluate(self) -> None:
        """Train and evaluate the model."""
        lg.debug("Training and evaluating model...")
        self.trainer.train()
        self.plot_train_eval_loss()

    def plot_train_eval_loss(self) -> None:
        """Plot the training and evaluation loss."""
        log_history = self.trainer.state.log_history
        training_loss = [log["loss"] for log in log_history if "loss" in log]
        eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]

        sns.set(style="darkgrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=range(len(training_loss)), y=training_loss, label="train_loss")
        sns.lineplot(x=range(len(eval_loss)), y=eval_loss, label="eval_loss")
        plt.xlabel("Epoch")
        output_path = os.path.join(self.output_dir, "plots", "train_vs_eval_loss.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.show()
        lg.debug(f"Succesfully saved train vs eval loss plot to {output_path}")

    def save_model(self, model: AutoModelForCausalLM) -> None:
        """Save the trained model.

        Args:
            model (AutoModelForCausalLM): The model to save.
        """
        lg.debug("Saving model ...")
        model_dir = os.path.join(self.output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_name = self.model_name_or_path.split("/")[-1]
        output_path = os.path.join(model_dir, model_name)
        model.save_pretrained(output_path)
        lg.debug(f"Model saved to {output_path}")

    def run(
        self,
        num_epochs: int = 20,
        per_device_train_batch_size: int = 8,
        load_model_in_8bit: bool = False,
        load_model_in_4bit: bool = False,
    ) -> None:
        """Run the training and evaluation process.

        Args:
            num_epochs (int): The number of training epochs.
            per_device_train_batch_size (int): The batch size per device for training.
            load_model_in_8bit (bool): Whether to load the model in 8-bit precision.
            load_model_in_4bit (bool): Whether to load the model in 4-bit precision.
        """
        lg.debug(
            f"Starting training with {num_epochs} epochs, and batch size: {per_device_train_batch_size}"
        )
        model, tokenizer = self._load_model_and_tokenizer(
            load_in_8bit=load_model_in_8bit, load_in_4bit=load_model_in_4bit
        )
        train_dataset, test_dataset = self.load_and_tokenize_dataset(model, tokenizer)
        self.setup_trainer(
            model,
            train_dataset,
            test_dataset,
            num_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
        )
        self.train_and_evaluate()
        self.save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetuning script for PEFT models (LoRA, PromptTuning, PrefixTuning)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config_000.json",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Extract finetuning config
    finetuning_config = config.get("finetuning", {})
    if not finetuning_config.get("enabled", False):
        lg.info("Finetuning is disabled in config. Exiting...")
        sys.exit(0)

    # Set up logging
    curr_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        DEFAULT_RUNS_BASE_DIR, f"{config['run_name']}_{curr_time_str}"
    )
    lg.add(os.path.join(output_dir, "logs", "{time}.log"))
    lg.info(f"Starting finetuning with config: {finetuning_config}")

    trainer = PeftTrainer(
        model_name_or_path=finetuning_config["model_name_or_path"],
        dataset_csv_path=finetuning_config["dataset_csv_path"],
        output_dir=output_dir,
        peft_type=finetuning_config.get("peft", DEFAULT_PEFT_TYPE),
        peft_hyperparams=finetuning_config.get("peft_hyperparams", {}),
    )

    trainer.run(
        num_epochs=finetuning_config.get("num_train_epochs", DEFAULT_NUM_EPOCHS),
        per_device_train_batch_size=finetuning_config.get(
            "batch_size", DEFAULT_BATCH_SIZE
        ),
        load_model_in_4bit=finetuning_config.get("load_model_in_4bit", False),
        load_model_in_8bit=finetuning_config.get("load_model_in_8bit", False),
    )
