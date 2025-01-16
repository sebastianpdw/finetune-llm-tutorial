### Configuration Keys and Values

| Name                        | Type    | Required | Example Value                      | Description                                                                           |
| --------------------------- | ------- | -------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| `run_name`                  | string  | Yes      | "meta-llama-3-1-8b-instruct"       | The name of the run. Used to create a directory to store all relevant files.          |
| `hyperstack_deployment`     | dict    | Yes      |                                    | Configuration for deploying a VM using Hyperstack.                                    |
| - `enabled`                 | boolean | Yes      | true                               | Whether to enable Hyperstack deployment.                                              |
| - `environment_name`        | string  | Yes      | "default-CANADA-1"                 | The name of the Hyperstack environment.                                               |
| - `vm_name`                 | string  | Yes      | "finetuning-vm"                    | The name of the VM to create.                                                         |
| - `flavor_name`             | string  | Yes      | "n3-L40x1"                         | The flavor of the VM to create.                                                       |
| - `delete_after_completion` | boolean | Yes      | false                              | Whether to delete the VM after completion.                                            |
| `finetuning`                | dict    | Yes      |                                    | Configuration for fine-tuning the model.                                              |
| - `enabled`                 | boolean | Yes      | true                               | Whether to enable fine-tuning.                                                        |
| - `use_host_hf_token`       | boolean | Yes      | true                               | Whether to use the host's Hugging Face token (in environment variable).               |
| - `model_name_or_path`      | string  | Yes      | "meta-llama/Llama-3.1-8B-Instruct" | The name or path of the model to fine-tune.                                           |
| - `dataset_csv_path`        | string  | Yes      | "data/datasets/train.csv"          | The path to the dataset CSV file.                                                     |
| - `num_train_epochs`        | number  | Yes      | 10                                 | The number of training epochs.                                                        |
| - `peft`                    | string  | Yes      | "lora"                             | The type of PEFT to use (e.g., "lora").                                               |
| - `peft_hyperparams`        | dict    | Yes      |                                    | Hyperparameters for the PEFT.                                                         |
| -- `r`                      | number  | Yes      | 8                                  | The rank of the LoRA.                                                                 |
| -- `lora_alpha`             | number  | Yes      | 32                                 | The alpha value for the LoRA.                                                         |
| -- `lora_dropout`           | number  | Yes      | 0.2                                | The dropout rate for the LoRA.                                                        |
| `inference`                 | dict    | Yes      |                                    | Configuration for running inference.                                                  |
| - `enabled`                 | boolean | Yes      | true                               | Whether to enable inference.                                                          |
| - `mode`                    | string  | Yes      | "csv"                              | The mode of inference (e.g., "csv" or "text").                                        |
| - `mode_configs`            | dict    | Yes      |                                    | Configuration for the inference mode.                                                 |
| -- `csv`                    | dict    | Yes      |                                    | Configuration for CSV mode.                                                           |
| --- `input_path`            | string  | Yes      | "data/datasets/test.csv"           | The path to the input CSV file.                                                       |
| --- `input_column`          | string  | Yes      | "user"                             | The name of the input column in the CSV file.                                         |
| --- `output_column`         | string  | Yes      | "response"                         | The name of the output column in the CSV file.                                        |
| -- `text`                   | dict    | Yes      |                                    | Configuration for text mode.                                                          |
| --- `input`                 | array   | Yes      | ["Hello"]                          | A list of input texts.                                                                |
| - `model_hyperparams`       | dict    | Yes      |                                    | Hyperparameters for the model.                                                        |
| -- `temperature`            | number  | Yes      | 0.7                                | The temperature for sampling.                                                         |
| -- `max_length`             | number  | Yes      | 500                                | The maximum length of the generated sequences.                                        |
| `app_deployment`            | dict    | Yes      |                                    | Configuration for deploying the chatbot app.                                          |
| - `enabled`                 | boolean | Yes      | true                               | Whether to enable app deployment.                                                     |
| - `use_host_hf_token`       | boolean | Yes      | true                               | Whether to use the host's Hugging Face token.                                         |
| - `model_run_dir`           | string  | Yes      | "latest"                           | The directory of the model run to deploy. 'latest' will deploy latest finetuned model |
| - `base_model`              | string  | Yes      | "meta-llama/Llama-3.1-8B-Instruct" | The base model to use for deployment.                                                 |
