import time

from colorama import Fore, Style


def build_prompt_structure(prompt: str, role: str) -> dict:
    """
    Builds a structured prompt that includes the role and content.

    Args:
        prompt (str): The actual content of the prompt.
        role (str): The role of the speaker (e.g., user, assistant).

    Returns:
        dict: A dictionary representing the structured prompt.
    """
    return {"role": role, "content": prompt}


def fancy_step_tracker(step: int, total_steps: int) -> None:
    """
    Displays a fancy step tracker for each iteration of the generation-reflection loop.

    Args:
        step (int): The current step in the loop.
        total_steps (int): The total number of steps in the loop.
    """
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"STEP {step + 1}/{total_steps}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)
