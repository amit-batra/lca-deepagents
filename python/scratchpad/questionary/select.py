import questionary

model = questionary.select(
    "Choose your local LLM:",
    choices=[
        "Qwen3-Coder",
        "Gemma",
        "Llama",
    ],
).ask()

if model is None:
    print("Operation cancelled.")
else:
    confirmed = questionary.confirm(
        f"Use {model}?"
    ).ask()

    if confirmed:
        print(f"Selected model: {model}")
