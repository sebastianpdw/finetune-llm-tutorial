import datetime
import json
import base64
import os
from pathlib import Path
import uuid
import requests
import streamlit as st


DEFAULT_MODEL_NAME = "gym-llama"
# API_CHAT_COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_API_CHAT_COMPLETIONS_URL = "http://llm-model:8000/v1/chat/completions"

ASSISTANT_AVATAR_IMG_PATH = "media/avatar.png"


st.set_page_config(layout="wide", page_title="Lex Llama")

st.markdown(
    """
    <style>
        .stAppHeader {
            display: none;
        }
        /* Move main container up */
        .stMainBlockContainer {
            padding-top: 2rem;  /* Reduce top padding */
        }
    </style>
""",
    unsafe_allow_html=True,
)


# Modified default configs to only include the requested parameters
DEFAULT_ASSISTANT_CONFIGS = {
    "system_prompt": "",
    "temperature": 0.5,
    "max_tokens": 512,
}


def show_parameter_controls():
    """Display sliders for model parameters."""
    with st.sidebar:
        # Add reset button at the top of the sidebar
        if st.button("Reset Messages", type="primary"):
            st.session_state.messages = [
                {"role": "system", "content": "You are a gym motivator."}
            ]
            st.rerun()

        st.header("Model Parameters")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=DEFAULT_ASSISTANT_CONFIGS["temperature"],
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused and deterministic.",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=DEFAULT_ASSISTANT_CONFIGS["max_tokens"],
            step=64,
            help="The maximum number of tokens to generate in the response.",
        )

        # Update the assistant configs with new values
        st.session_state.assistant_configs = {
            "system_prompt": "",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


def stream_response(response):
    """
    Streams the response from the API.

    Args:
        response (requests.Response): The response object from the API.
    """
    if not response:
        return
    content = ""
    message_placeholder = st.empty()
    try:
        for chunk in response.iter_lines():
            if chunk:
                # Decode the chunk
                decoded_chunk = chunk.decode("utf-8")

                # Check and remove the "data: " prefix
                if decoded_chunk.startswith("data: "):
                    decoded_chunk = decoded_chunk[len("data: ") :]

                # Check for special markers like "[DONE]" and skip them
                if decoded_chunk.strip() == "[DONE]":
                    break

                # Parse JSON from the stripped chunk
                try:
                    delta = json.loads(decoded_chunk)
                    # Extract and append content
                    if (
                        part := delta.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    ):
                        content += part
                        message_placeholder.markdown(content)
                except json.JSONDecodeError as e:
                    st.error(f"JSON decoding error: {e}. Chunk: {decoded_chunk}")

        # Append the complete message to the session state
        st.session_state.messages.append({"role": "assistant", "content": content})
    except requests.exceptions.ChunkedEncodingError:
        st.error("Something went wrong. Please 'Reset messages' to continue.")


def get_base64_encoded_image(image_path):
    """Get base64 encoded image from file path."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def show_banner():
    # Get base64 encoded image
    img_base64 = get_base64_encoded_image(ASSISTANT_AVATAR_IMG_PATH)
    img_html = f"data:image/png;base64,{img_base64}"

    st.markdown(
        f"""
        <style>
            @media (max-width: 768px) {{
                .banner-container {{
                    padding: 1rem !important;
                }}
                .banner-content {{
                    flex-direction: column !important;
                    gap: 0rem !important;
                    text-align: center !important;
                }}
                .banner-avatar {{
                    width: 96px !important;
                    height: 96px !important;
                }}
                .banner-title {{
                    font-size: 2rem !important;
                }}
                .banner-status {{
                    margin-top: 1rem !important;
                    width: 100% !important;
                    justify-content: center !important;
                }}
            }}
        </style>
        <div class='banner-container' style='background: linear-gradient(to right, #0f172a, #1e3a8a); 
            padding: 1.5rem; 
            border-radius: 0.75rem; 
            margin-bottom: 2rem;
            position: relative;
            border: 2px solid transparent;
            background-clip: padding-box;'>
            <div style='position: absolute;
                       inset: 0;
                       border-radius: 0.75rem;
                       padding: 2px;
                       background: linear-gradient(to right, #ff8a00, #e900ff, #4c68d7);
                       -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
                       -webkit-mask-composite: xor;
                       mask-composite: exclude;
                       pointer-events: none;'>
            </div>
            <div style='display: flex; flex-direction: column;'>
                <div class='banner-content' style='display: flex; align-items: center; gap: 1.5rem;'>
                    <img src='{img_html}' 
                        class='banner-avatar'
                        style='width: 124px; 
                               height: 124px; 
                               border-radius: 50%; 
                               background: rgba(255,255,255,0.05); 
                               padding: 4px;
                               box-shadow: 0 0 20px rgba(0,0,0,0.2);'/>
                    <div style='flex-grow: 1;'>
                        <h1 class='banner-title' 
                            style='color: white; 
                                   font-size: 2.5rem; 
                                   font-weight: bold; 
                                   margin: 0;
                                   text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            Lex Llama
                        </h1>
                        <p style='color: rgba(255,255,255,0.8); 
                                 margin-top: 0rem;
                                 font-size: 1.1rem;'>
                            Your AI Fitness Motivation Coach
                        </p>
                    </div>
                    <div class='banner-status' 
                         style='background: rgba(255,255,255,0.05); 
                                border: 1px solid rgba(255,255,255,0.1);
                                border-radius: 9999px; 
                                padding: 0.75rem 1.25rem; 
                                display: inline-flex; 
                                align-items: center; 
                                gap: 0.5rem;
                                backdrop-filter: blur(8px);'>
                        <div style='width: 8px; 
                                   height: 8px; 
                                   background: #10b981; 
                                   border-radius: 9999px;
                                   box-shadow: 0 0 10px rgba(16,185,129,0.5);'></div>
                        <span style='color: rgba(255,255,255,0.9); 
                                   font-size: 0.95rem;
                                   font-weight: 500;'>Ready to motivate</span>
                    </div>
                </div>
                <div class='banner-footer' 
                     style='margin-top: 1rem;
                            padding-top: 1rem;
                            border-top: 1px solid rgba(255,255,255,0.1);
                            text-align: center;
                            font-size: 0.75rem;
                            color: rgba(255,255,255,0.6);
                            line-height: 1.4;'>
                    Lex Llama is an AI chatbot based on a fictional character, created for entertainment and motivation purposes only. 
                    Any resemblance to real persons or entities is purely coincidental. The creator assumes no liability for decisions 
                    made based on interactions with this chatbot. Content is generated through AI and may not be suitable as 
                    professional fitness or medical advice. No offense is intended to any person or organization.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_footer():
    st.markdown(
        """
            """,
        unsafe_allow_html=True,
    )


def log_messages(messages, session_id):
    """
    Log messages to a JSON file.
    """
    log_dir = Path("messages")
    log_dir.mkdir(exist_ok=True)
    curr_date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = log_dir / session_id / f"{curr_date_time_str}.json"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    last_two_messages = messages[-2:]

    with open(log_file_path, "w") as f:
        json.dump(last_two_messages, f, indent=2)


def main(api_url=None, model_name=None):
    """
    The main function to initialize and run the Streamlit app.
    """
    os.makedirs("messages", exist_ok=True)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        print("-----------------------------------")
        print(
            f"Starting Lex Llama chatbot with session ID: {st.session_state['session_id']}"
        )
    session_id = st.session_state.session_id

    if api_url is None:
        api_url = DEFAULT_API_CHAT_COMPLETIONS_URL
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    # Initialize assistant configs if not present
    if "assistant_configs" not in st.session_state:
        print("Initializing assistant configs")
        st.session_state.assistant_configs = DEFAULT_ASSISTANT_CONFIGS.copy()

    # Disabled for now, but can be enabled to show parameter controls
    show_parameter_controls()

    show_banner()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a gym motivator."}
        ]

    avatar_dict = {"assistant": ASSISTANT_AVATAR_IMG_PATH, "user": "🦥"}

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"], avatar=avatar_dict[message["role"]]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=avatar_dict["user"]):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=avatar_dict["assistant"]):
            messages = [
                {
                    "role": "system",
                    "content": st.session_state.assistant_configs["system_prompt"],
                }
            ] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            data = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "temperature": st.session_state.assistant_configs["temperature"],
                "max_tokens": st.session_state.assistant_configs["max_tokens"],
            }

            # Make API call
            try:
                response = requests.post(
                    api_url,
                    json=data,
                    stream=data["stream"],
                )
                status_code = response.status_code

                if status_code != 200 or not response:
                    response_json = response.text
                    st.error(f"Error from API: {response_json}")

            except (requests.exceptions.ConnectionError, AttributeError):
                message = "Hi, I am currently in the gym. Please try again later."
                st.markdown(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message}
                )
                response = None

            if data["stream"]:
                stream_response(response)
            else:
                response_json = response.json()
                content = response_json["choices"][0]["message"]["content"]
                st.markdown(content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )

    if len(st.session_state.messages) > 1:
        log_messages(st.session_state.messages, session_id)


if __name__ == "__main__":
    main(os.environ.get("API_CHAT_COMPLETIONS_URL"), os.environ.get("MODEL_NAME"))
