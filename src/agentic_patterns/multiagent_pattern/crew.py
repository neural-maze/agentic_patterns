from collections import deque

from colorama import Fore
from graphviz import Digraph

from agentic_patterns.utils.logging import fancy_print


class Crew:
    # Class-level variable to track the active Crew context
    current_crew = None

    def __init__(self):
        self.agents = []

    def __enter__(self):
        # Set this crew as the current active context
        Crew.current_crew = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear the active context when exiting the block
        Crew.current_crew = None

    def add_agent(self, agent):
        self.agents.append(agent)

    @staticmethod
    def register_agent(agent):
        if Crew.current_crew is not None:
            Crew.current_crew.add_agent(agent)

    def topological_sort(self):
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
                "There's a circle of dependencies on the agents! Topological sort is not possible"
            )

        return sorted_agents

    def plot(self):
        """
        Plots the Directed Acyclic Graph (DAG) of agents in the given crew using Graphviz.

        Parameters:
        crew (Crew): The crew instance containing the registered agents.

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
        sorted_agents = self.topological_sort()
        for agent in sorted_agents:
            fancy_print(f"RUNNING AGENT: {agent}")
            print(Fore.RED + f"{agent.run()}")
