from typing import List, Dict, Optional, Union
from loguru import logger as lg
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel


def call_openai_chatgpt_api(
    messages: List[Dict[str, str]],
    model_name: str = "gpt-4o-mini-2024-07-18",
    response_format: Optional[Union[BaseModel, Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    client: Optional[OpenAI] = None,
) -> Choice:
    """
    Calls the OpenAI ChatGPT API with the provided messages and parameters.

    Args:
        messages (List[Dict[str, str]]): List of messages to send to the API.
        model_name (str): The model name to use for the API call.
        response_format (Optional[Union[BaseModel, Dict[str, str]]]): The format of the response.
        temperature (Optional[float]): The temperature setting for the API.
        client (Optional[OpenAI]): The OpenAI client instance.

    Returns:
        Choice: The first choice from the API response.
    """
    if client is None:
        client = OpenAI()
    if response_format not in [None, {"type": "json_object"}]:
        raise ValueError(f"response_format must be None or {{'type': 'json_object'}}")

    lg.debug(f"Calling OpenAI ChatGPT API (model: {model_name}) with: {messages}")
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )

    return completion.choices[0]


def create_question_prompt(
    user_message: str, assistant_response: str, system_prompt: str
) -> str:
    """
    Creates a question prompt to evaluate the assistant's response.

    Args:
        user_message (str): The user's message.
        assistant_response (str): The assistant's response.
        system_prompt (str): The system prompt.

    Returns:
        str: The formatted question prompt.
    """
    return f"""Does the ASSISTANT_RESPONSE to the USER_MESSAGE below follow the instructions in the SYSTEM_PROMPT? 
    Please respond with 'yes' or 'no'. 
    Respond in json format: {{'response': '[yes|no]', 'relevant_rule': [relevant_rule] 'explanation': '[explanation]'}}
    \n\nSYSTEM_PROMPT: \"\"\"{system_prompt}\"\"\"
    \n\nUSER_MESSAGE: \"\"\"{user_message}\"\"\"
    \n\nASSISTANT_RESPONSE: \"\"\"{assistant_response}\"\"\"
    """


def evaluate_response(
    user_message: str,
    assistant_response: str,
    system_prompt: str,
    temperature: float = 0.7,
    model_name: str = "gpt-3.5-turbo-1106",
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Evaluates the assistant's response against the system prompt.

    Args:
        user_message (str): The user's message.
        assistant_response (str): The assistant's response.
        system_prompt (str): The system prompt.
        temperature (float): The temperature setting for the API.
        model_name (str): The model name to use for the API call.

    Returns:
        Dict[str, Union[str, Dict[str, str]]]: The evaluation result in JSON format.
    """
    question_prompt = create_question_prompt(
        user_message, assistant_response, system_prompt
    )
    question_message = {"role": "user", "content": question_prompt}
    messages = [{"role": "system", "content": "You are a helpful AI assistant"}] + [
        question_message
    ]

    result = call_openai_chatgpt_api(
        messages,
        model_name=model_name,
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    return result
