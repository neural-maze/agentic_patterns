import re
from dataclasses import dataclass


@dataclass
class TagContentResult:
    """
    A data class to represent the result of extracting tag content.

    Attributes:
        content (List[str]): A list of strings containing the content found between the specified tags.
        found (bool): A flag indicating whether any content was found for the given tag.
    """

    content: list[str]
    found: bool


def extract_tag_content(text: str, tag: str) -> TagContentResult:
    """
    Extracts all content enclosed by specified tags (e.g., <thought>, <response>, etc.).

    Parameters:
        text (str): The input string containing multiple potential tags.
        tag (str): The name of the tag to search for (e.g., 'thought', 'response').

    Returns:
        dict: A dictionary with the following keys:
            - 'content' (list): A list of strings containing the content found between the specified tags.
            - 'found' (bool): A flag indicating whether any content was found for the given tag.
    """
    # Build the regex pattern dynamically to find multiple occurrences of the tag
    tag_pattern = rf"<{tag}>(.*?)</{tag}>"

    # Use findall to capture all content between the specified tag
    matched_contents = re.findall(tag_pattern, text, re.DOTALL)

    # Return the dataclass instance with the result
    return TagContentResult(
        content=[content.strip() for content in matched_contents],
        found=bool(matched_contents),
    )
