from difflib import SequenceMatcher
import re

from llm_open_ie.llm.gpt import GPTOpenIE
from llm_open_ie.logger import LOGGER

SYSTEM = '''\
You identify mentions of entities in the text: the user provides you a list of known entities and a sentence in which\
 those entities may be mentioned. For each entity you tell if they are actually mentioned in the sentence by saying\
 "yes" or "no".
You write the user list again in the same order with your additional answer formatted this way:
`<entity ID>) <entity>|||<yes/no>`
'''

USER = '''\
User entities:
```
{entities_numbered_list}
```

Sentence:
```
{context_sentence}
```
'''

ENTITY_COUNT_START = 1
LABEL_SIM_RATIO_TH = 0.90


def recognize_mentions(gpt: GPTOpenIE, sentence: str, entities: list[dict]) -> list[int]:
    entities_list_str = '\n'.join([f'{i}) {e["label"]}' for i, e in enumerate(entities, ENTITY_COUNT_START)])

    answer = gpt.chat_completion(
        system=SYSTEM,
        user=USER.format(entities_numbered_list=entities_list_str, context_sentence=sentence)
    )
    return _answer_parser(answer, entities)


def _answer_parser(answer: str, entities: list[dict]) -> list[int]:
    answer_lines = answer.strip().split('\n')
    parsed_answer_lines = [_parse_answer_line(line) for line in answer_lines if _is_valid_answer_line_pattern(line)]

    n_dropped_lines = len(answer_lines) - len(parsed_answer_lines)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(answer_lines)} wrongly spelled mentions information.')

    consistent_answers = [answer for answer in parsed_answer_lines if _check_mention_consistency(answer, entities)]
    n_dropped_lines = len(parsed_answer_lines) - len(consistent_answers)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(parsed_answer_lines)} inconsistent mentions information.')

    return [entity_mention['id'] for entity_mention in consistent_answers if entity_mention['is_mentioned']]


def _is_valid_answer_line_pattern(answer_line: str) -> bool:
    # Regular expression to describe the full pattern of a line in the answer
    is_valid = bool(re.fullmatch(
        r'\d+\) ?'                   # Entity ID
        r'[^|]+'                     # Entiy label
        r'([|]{3}) ?'                # Separator
        r'([Yy][Ee][Ss]|[Nn][Oo])',  # Answer
        answer_line.strip()))
    if not is_valid:
        LOGGER.warning(f'Wrongly spelled mention information: "{answer_line}".')

    return is_valid


def _parse_answer_line(answer_line: str) -> dict:
    # Replace the parenthesis ') ' that separates the ID number from the label with '|||'; then split the 3 fields
    #  using the only separator '|||'
    e_id, e_label, e_is_mentioned = re.sub(r'(?<=\d)\) ?', '|||', answer_line).split('|||')

    e_id = int(e_id) - ENTITY_COUNT_START
    e_label = e_label.strip()
    e_is_mentioned_str = e_is_mentioned.lower().strip()
    if e_is_mentioned_str == 'no':
        e_is_mentioned = False
    elif e_is_mentioned_str == 'yes':
        e_is_mentioned = True
    else:
        assert False  # if `_is_valid_answer_line_pattern` has been correctly checked, this branch should not be reached

    return {
        'id': e_id,
        'label': e_label,
        'is_mentioned': e_is_mentioned
    }


def _check_mention_consistency(mention_dict: dict, entities: list[dict]) -> bool:
    e_index = mention_dict['id']
    # There is no such entity
    if not (0 <= e_index < len(entities)):
        return False
    # Forgive little syntactic differences: do not request perfect string matches
    seq_match = SequenceMatcher(None, mention_dict['label'].lower(), entities[e_index]['label'].lower())
    return seq_match.ratio() >= LABEL_SIM_RATIO_TH
