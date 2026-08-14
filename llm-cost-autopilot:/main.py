from router import choose_model
from llm import call_llm
from cost_tracker import calculate_cost


def cost_autopilot(prompt):


    model = choose_model(prompt)

    print(f"Selected model: {model}")


    response = call_llm(model, prompt)


    cost = calculate_cost(
        model,
        response["input_tokens"],
        response["output_tokens"]
    )


    print(f"Input tokens: {response['input_tokens']}")
    print(f"Output tokens: {response['output_tokens']}")
    print(f"Cost: ${cost:.6f}")


    return response["text"]


prompt = input("Enter your prompt: ")

answer = cost_autopilot(prompt)

print("\nAnswer:")
print(answer)