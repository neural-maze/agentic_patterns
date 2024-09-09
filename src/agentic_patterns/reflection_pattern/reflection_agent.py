"""
This module provides the implementation of a ReflectionAgent class, 
which generates responses based on a provided prompt and reflects 
on the generated content to improve or critique the responses. 
"""

import time

from colorama import Fore, Style
from dotenv import load_dotenv
from groq import Groq


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

    @staticmethod
    def build_prompt_structure(prompt: str, role: str):
        """
        Builds a structured prompt that includes the role and content.

        Args:
            prompt (str): The actual content of the prompt.
            role (str): The role of the speaker (e.g., user, assistant).

        Returns:
            dict: A dictionary representing the structured prompt.
        """
        return {"role": role, "content": prompt}

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

    def fancy_step_tracker(self, step: int, total_steps: int):
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

    def run(
        self,
        generation_system_prompt: str,
        reflection_system_prompt: str,
        prompt: str,
        n_steps: int = 3,
        verbose: int = 0,
    ):
        """_summary_

        Args:
            generation_system_prompt (str): _description_
            reflection_system_promp (str): _description_
            prompt (str): _description_
        """
        generation_history = [{"role": "system", "content": generation_system_prompt}]
        generation_history.append({"role": "user", "content": prompt})

        reflection_history = [{"role": "system", "content": reflection_system_prompt}]

        for step in range(n_steps):

            self.fancy_step_tracker(step, n_steps)
            
            # Generate the output based on generation history
            generation = self.generate(generation_history, verbose=verbose)

            # Update histories
            generation_history.append(
                self.build_prompt_structure(generation, "assistant")
            )
            reflection_history.append(self.build_prompt_structure(generation, "user"))

            # Reflect and critique the generation
            critique = self.reflect(reflection_history, verbose=verbose)
            reflection_history.append(
                self.build_prompt_structure(critique, "assistant")
            )
            generation_history.append(self.build_prompt_structure(critique, "user"))

        return generation
