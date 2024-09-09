from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.reflection_pattern.utils import (
    build_prompt_structure,
    fancy_step_tracker,
)

load_dotenv()


class ReflectionAgent:
    """
    A class that implements a simple Reflection Agent, that generates responses
    and reflects on them using a Groq model
    """

    def __init__(self, model: str = "llama3-70b-8192"):
        self.client = Groq()
        self.model = model

    def __str__(self):
        return "Reflection Agent"

    def generate(self, generation_history: list, verbose: int = 0):
        """
        Generates a response based on the provided generation history.

        Args:
            generation_history (list): List of messages forming the conversation history
            verbose (int, optional): Verbosity level

        """
        output = (
            self.client.chat.completions.create(
                messages=generation_history, model=self.model
            )
            .choices[0]
            .message.content
        )
        if verbose > 0:
            print(Fore.BLUE + "\n\nGENERATION\n\n " + str(output))

        return output

    def reflect(self, reflection_history: list, verbose: int = 0):
        """
        Reflects on the generation by generating a critique or feedback.

        Args:
            reflection_history (list): List of messages forming the reflection history.
            verbose (int, optional): Verbosity level.
        """
        output = (
            self.client.chat.completions.create(
                messages=reflection_history, model=self.model
            )
            .choices[0]
            .message.content
        )

        if verbose > 0:
            print(Fore.GREEN + "\n\nREFLECTION\n\n" + str(output))

        return output

    def run(
        self,
        generation_system_prompt: str,
        reflection_system_prompt: str,
        user_prompt: str,
        n_steps: int = 3,
        verbose: int = 0,
    ) -> str:
        """_summary_

        Args:
            generation_system_prompt (str): _description_
            reflection_system_promp (str): _description_
            user_prompt (str): _description_
        """
        generation_history = [
            build_prompt_structure(prompt=generation_system_prompt, role="system"),
            build_prompt_structure(prompt=user_prompt, role="user"),
        ]
        reflection_history = [
            build_prompt_structure(prompt=reflection_system_prompt, role="system")
        ]

        for step in range(n_steps):

            fancy_step_tracker(step, n_steps)

            # Generate the output based on generation history
            generation = self.generate(generation_history, verbose=verbose)

            # Update histories
            generation_history.append(
                build_prompt_structure(prompt=generation, role="assistant")
            )
            reflection_history.append(
                build_prompt_structure(prompt=generation, role="user")
            )

            # Reflect and critique the generation
            critique = self.reflect(reflection_history, verbose=verbose)
            reflection_history.append(
                build_prompt_structure(prompt=critique, role="assistant")
            )
            generation_history.append(
                build_prompt_structure(prompt=critique, role="user")
            )

        return generation
