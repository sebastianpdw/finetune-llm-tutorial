import argparse
import os
import time
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from loguru import logger as lg

from utils.io import read_json, get_latest_run_dir, get_model_path_from_run_dir


class FinetunedLLM:
    DEFAULT_TEMPERATURE = 1
    DEFAULT_MAX_LENGTH = 500

    SUPPORTED_MODES = ["csv", "text"]

    def __init__(self, model_path_or_name: str):
        """
        Initializes the finetuned Large Language Model.

        Args:
            model_path_or_name (str): Path to the directory containing your fine-tuned model.
        """
        self.model_path_or_name = model_path_or_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model_base_name = self.model.config._name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_base_name, add_bos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
        )

    def _get_assistant_reply(
        self,
        user_message: str,
        temperature: float,
        max_length: int,
    ) -> str:
        """
        Generates a response from the assistant for a given user message.

        Args:
            user_message (str): The user's message.
            temperature (float): Sampling temperature.
            max_length (int): Maximum length of the generated response.

        Returns:
            str: The assistant's generated response.
        """
        chat = [{"role": "user", "content": user_message}]
        output = self.pipe(
            chat,
            temperature=temperature,
            max_length=max_length,
        )[0]
        assistant_message = output["generated_text"][-1]["content"]
        return assistant_message

    def generate_from_csv(
        self,
        csv_path: str,
        input_column: str,
        output_column: str,
        temperature: Optional[float] = None,
        max_length: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> List[str]:
        """
        Generates responses to the messages in the given CSV file.

        Args:
            csv_path (str): The path to the CSV file containing the messages to generate responses to.
            input_column (str): The name of the column containing the messages to generate responses to.
            temperature (float, optional): The temperature to use when sampling. Defaults to 0.7.
            max_length (int, optional): The maximum length of the generated response. Defaults to 500.
            output_path (str, optional): The path to save the generated responses to. Defaults to None.

        Returns:
            list: A list of the generated responses.
        """
        temperature = temperature or self.DEFAULT_TEMPERATURE
        max_length = max_length or self.DEFAULT_MAX_LENGTH

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found for inference: {csv_path}")

        lg.debug(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Add tqdm progress bar to track message processing
        # todo: improve this with batching
        tqdm.pandas(desc="Processing messages")
        df[output_column] = df[input_column].progress_apply(
            lambda x: self._get_assistant_reply(
                x,
                temperature,
                max_length,
            )
        )

        df = df[[input_column, output_column]]

        # Save to CSV if output_csv_path is provided
        if output_path:
            if not output_path.lower().endswith(".csv"):
                raise ValueError(
                    f"Output path must be a CSV file. Provided output path: {output_path}"
                )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            lg.debug(f"Saving generated responses to: {output_path}")
            df.to_csv(output_path, index=False)

        return df[output_column].tolist()

    def generate_from_messages(
        self,
        message: str,
        temperature: Optional[float] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Generates a response to a single message.

        Args:
            message (str): The message to generate a response to.
            temperature (float, optional): The temperature to use when sampling. Defaults to 0.7.
            max_length (int, optional): The maximum length of the generated response. Defaults to 500.

        Returns:
            str: The generated response.
        """
        temperature = temperature or self.DEFAULT_TEMPERATURE
        max_length = max_length or self.DEFAULT_MAX_LENGTH

        return self._get_assistant_reply(
            message,
            temperature,
            max_length,
        )

    def generate(
        self,
        run_dir: str,
        inference_configs: dict,
    ) -> List[str]:
        """
        Generates responses based on the provided inference configurations.

        Args:
            run_dir (str): The directory of the current run.
            inference_configs (dict): The inference configurations.

        Returns:
            list: A list of generated responses.
        """
        mode = inference_configs.get("mode")
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode: {mode}. Supported modes are: {self.SUPPORTED_MODES}"
            )
        lg.debug(f"Generating responses in {mode} mode.")
        lg.debug(f"Inference configs: {inference_configs}")

        model_hyperparams_configs = inference_configs.get("model_hyperparams", {})
        temperature = model_hyperparams_configs.get("temperature")
        max_length = model_hyperparams_configs.get("max_length")

        mode_configs = inference_configs["mode_configs"]

        if mode == "csv":
            if "csv" not in mode_configs:
                raise ValueError("CSV mode requires a 'csv' key in the mode_configs.")
            csv_mode_configs = mode_configs["csv"]
            input_path = csv_mode_configs["input_path"]
            input_column = csv_mode_configs["input_column"]
            output_column = csv_mode_configs["output_column"]
            output_path = os.path.join(run_dir, "output", "inference.csv")
            return self.generate_from_csv(
                input_path,
                input_column,
                output_column,
                temperature=temperature,
                max_length=max_length,
                output_path=output_path,
            )
        elif mode == "text":
            if "text" not in mode_configs:
                raise ValueError("Text mode requires a 'text' key in the mode_configs.")
            text_mode_configs = mode_configs["text"]
            text_list = text_mode_configs.get("text")
            if isinstance(text_list, str):
                text_list = [text_list]

            return [
                self.generate_from_messages(
                    text,
                    temperature=temperature,
                    max_length=max_length,
                )
                for text in text_list
            ]
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for PEFT models (LoRA, PromptTuning, PrefixTuning)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config_000.json",
        help="Path to the  config file.",
    )
    args = parser.parse_args()

    # Read config
    config_path = args.config_path
    config = read_json(config_path)
    run_name = config["run_name"]

    # Initialize logger
    run_dir = get_latest_run_dir(run_base_name=run_name)
    curr_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    lg.add(f"{run_dir}/logs/{curr_time_str}.log")

    lg.info("Starting inference...")
    model_path = get_model_path_from_run_dir(run_dir)
    finetuned_llm = FinetunedLLM(model_path)
    inference_configs = config["inference"]
    finetuned_llm.generate(run_dir, inference_configs)
    lg.info("Inference completed.")
