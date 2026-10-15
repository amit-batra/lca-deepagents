import questionary

port = questionary.text(
    "Enter server port:",
    default="8000",
    validate=lambda value: (
        value.isdigit()
        and 1 <= int(value) <= 65535
    ),
).ask()