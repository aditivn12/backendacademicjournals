def choose_model(prompt):
    prompt = prompt.lower()

    simple_words = [
        "summarize",
        "rewrite",
        "classify",
        "extract"
    ]

    for word in simple_words:
        if word in prompt:
            return "cheap"

    return "expensive"