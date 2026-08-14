PRICES = {
    "cheap": {
        "input": 0.10,
        "output": 0.40
    },

    "expensive": {
        "input": 2.00,
        "output": 8.00
    }
}


def calculate_cost(model, input_tokens, output_tokens):

    input_price = PRICES[model]["input"]
    output_price = PRICES[model]["output"]

    input_cost = input_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price

    return input_cost + output_cost