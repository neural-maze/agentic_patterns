import time


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


def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the current chat history.
        msg (str): The message to append.
        role (str): The role type (e.g. 'user', 'assistant', 'system')
    """
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        """Add a message to the queue.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        super().__init__(messages, total_length)

    def append(self, msg: str):
        """Add a message to the queue. The first messaage will always stay fixed.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)
