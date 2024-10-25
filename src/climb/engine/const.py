# For all options, see: site-packages/openai/types/chat_model.py
ALLOWED_MODELS = [
    # GPT-4o:
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    # GPT-4-turbo:
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    # GPT-3.5-turbo:
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
]

# For the values, see: https://platform.openai.com/docs/models/
MODEL_MAX_MESSAGE_TOKENS = {
    "gpt-4o-2024-08-06": 16_384,
    "gpt-4o-2024-05-13": 4_096,
    # ---
    "gpt-4-1106-preview": 4_096,
    "gpt-4-0125-preview": 4_096,
    # ---
    "gpt-3.5-turbo-1106": 4_096,
    "gpt-3.5-turbo-0125": 4_096,
}
MODEL_CONTEXT_SIZE = {
    "gpt-4o-2024-08-06": 4_096,
    "gpt-4o-2024-05-13": 4_096,
    # ---
    "gpt-4-1106-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    # ---
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-3.5-turbo-0125": 16_385,
}
