import re

from llm_open_ie.llm.gpt import GPTOpenIE
from llm_open_ie.logger import LOGGER

SYSTEM = '''\
The information in a text is expressed by mentions of concepts such as assertions, facts, individuals, objects,\
 events, tasks, activities, and more. We call these concepts "entities".
You identify all the "entities" in the text, and for each one, you provide both: 
 - a brief description of that entity (max 1 sentence);
 - a list of types, i.e. a term (or compound term) to define a category or hyperonym for that entity.
The output you provide is the list of entities, descriptions, and types, formatted with an initial hyphen this way:
`- <entity>|||<description>|||[<type 1>, <type 2>, ...]`
The user text language may be anything, but your output should be {output_language}!
'''


def extract_entities(gpt: GPTOpenIE, text: str, output_language: str) -> list[dict]:
    answer = gpt.chat_completion(
        system=SYSTEM.format(output_language=output_language),
        user=text)
    return _answer_parser(answer)


def _answer_parser(answer: str) -> list[dict]:
    answer_lines = answer.strip().split('\n')
    parsed_answer_lines = [_parse_answer_line(line) for line in answer_lines if _is_valid_answer_line_pattern(line)]

    n_dropped_lines = len(answer_lines) - len(parsed_answer_lines)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(answer_lines)} wrongly spelled entities information.')

    return parsed_answer_lines


def _is_valid_answer_line_pattern(answer_line: str) -> bool:
    # Regular expression to describe the full pattern of a line in the answer
    is_valid = bool(re.fullmatch(
        r'- ?([^|]+)'                # Entity label
        r'([|]{3})'                  # Separator
        r'([^|]+)'                   # Entity description
        r'([|]{3})'                  # Separator
        r'( ?\[([^,]+, ?)*[^,]+])',  # Entity types (comma separator)
        answer_line.strip()))
    if not is_valid:
        LOGGER.warning(f'Wrongly spelled entity information: "{answer_line}".')

    return is_valid


def _parse_answer_line(answer_line: str) -> dict:
    # Skip the starting '- ' and then get the 3 fields separated by '|||'
    label, description, types_str = [a.strip() for a in re.sub(r'^- ?', '', answer_line).split('|||')]
    # Skip '[' and ']' and get the types separated by commas (or commas and space)
    types = [t.strip() for t in types_str.strip()[1:-1].replace(', ', ',').split(',')]

    return {
        'label': label.strip(),
        'description': description.strip(),
        'types': types
    }
