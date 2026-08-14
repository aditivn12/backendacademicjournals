def call_llm(model, prompt):

    if model == "cheap":
        return {
            "text": "This is a response from the cheap model.",
            "input_tokens": 100,
            "output_tokens": 50
        }

    else:
        return {
            "text": "This is a response from the expensive model.",
            "input_tokens": 100,
            "output_tokens": 150
        }