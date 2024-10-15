ALLOWED_MODELS = [
    # GPT-4-turbo:
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    # GPT-3.5-turbo:
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
]
# All options:
# # GPT-4-turbo:
# "gpt-4-0125-preview",
# "gpt-4-turbo-preview",
# "gpt-4-1106-preview",
# # + Vision:
# "gpt-4-vision-preview",
# # GPT-4 (non-turbo):
# "gpt-4",
# "gpt-4-0314",
# "gpt-4-0613",
# # GPT-4 (32k):
# "gpt-4-32k",
# "gpt-4-32k-0314",
# "gpt-4-32k-0613",
# # GPT-3.5-turbo:
# "gpt-3.5-turbo",
# "gpt-3.5-turbo-16k",
# "gpt-3.5-turbo-0301",
# "gpt-3.5-turbo-0613",
# "gpt-3.5-turbo-1106",
# "gpt-3.5-turbo-0125",
# # GPT-3.5-turbo (16k):
# "gpt-3.5-turbo-16k-0613",

MODEL_MAX_MESSAGE_TOKENS = {
    "gpt-4-1106-preview": 4096,
    "gpt-4-0125-preview": 4096,
    "gpt-3.5-turbo-1106": 4096,
    "gpt-3.5-turbo-0125": 4096,
}

MODEL_CONTEXT_SIZE = {
    "gpt-4-1106-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-3.5-turbo-0125": 16_385,
}
