import questionary

models = questionary.checkbox(
    "Choose all LLMs that you want to use:",
    choices=[
        "Qwen3-Coder",
        "Gemma",
        "Llama",
        "Mistral",
        "Falcon",
        "MPT",
    ],
).ask()

if models is None:
    print("Operation cancelled.")
else:
    confirmed = questionary.confirm(
        f"Use {models}?"
    ).ask()

    if confirmed:
        print(f"Selected models: {models}")
