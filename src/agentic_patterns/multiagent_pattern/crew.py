from collections import deque

from colorama import Fore
from graphviz import Digraph  # type: ignore

from agentic_patterns.utils.logging import fancy_print


class Crew:
    """
    A class representing a crew of agents working together.

    This class manages a group of agents, their dependencies, and provides methods
    for running the agents in a topologically sorted order.

    Attributes:
        current_crew (Crew): Class-level variable to track the active Crew context.
        agents (list): A list of agents in the crew.
    """

    current_crew = None

    def __init__(self):
        self.agents = []

    def __enter__(self):
        """
        Enters the context manager, setting this crew as the current active context.

        Returns:
            Crew: The current Crew instance.
        """
        Crew.current_crew = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context manager, clearing the active context.

        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The traceback, if an exception was raised.
        """
        Crew.current_crew = None

    def add_agent(self, agent):
        """
        Adds an agent to the crew.

        Args:
            agent: The agent to be added to the crew.
        """
        self.agents.append(agent)

    @staticmethod
    def register_agent(agent):
        """
        Registers an agent with the current active crew context.

        Args:
            agent: The agent to be registered.
        """
        if Crew.current_crew is not None:
            Crew.current_crew.add_agent(agent)

    def topological_sort(self):
        """
        Performs a topological sort of the agents based on their dependencies.

        Returns:
            list: A list of agents sorted in topological order.

        Raises:
            ValueError: If there's a circular dependency among the agents.
        """
        in_degree = {agent: len(agent.dependencies) for agent in self.agents}
        queue = deque([agent for agent in self.agents if in_degree[agent] == 0])

        sorted_agents = []

        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)

            for dependent in current_agent.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_agents) != len(self.agents):
            raise ValueError(
                "Circular dependencies detected among agents, preventing a valid topological sort"
            )

        return sorted_agents

    def plot(self):
        """
        Plots the Directed Acyclic Graph (DAG) of agents in the crew using Graphviz.

        Returns:
            Digraph: A Graphviz Digraph object representing the agent dependencies.
        """
        dot = Digraph(format="png")  # Set format to PNG for inline display

        # Add nodes and edges for each agent in the crew
        for agent in self.agents:
            dot.node(agent.name)
            for dependency in agent.dependencies:
                dot.edge(dependency.name, agent.name)
        return dot

    def run(self):
        """
        Runs all agents in the crew in topologically sorted order.

        This method executes each agent's run method and prints the results.
        """
        sorted_agents = self.topological_sort()
        for agent in sorted_agents:
            fancy_print(f"RUNNING AGENT: {agent}")
            print(Fore.RED + f"{agent.run()}")
