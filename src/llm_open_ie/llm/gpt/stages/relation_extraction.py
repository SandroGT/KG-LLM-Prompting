from difflib import SequenceMatcher
import re

from llm_open_ie.llm.gpt import GPTOpenIE
from llm_open_ie.logger import LOGGER

SYSTEM = '''\
You find relations between entities in the text: the user provides you a list of known entities and a sentence in\
 which those entities may be related. For each entity you check the relation with the other entities and if it exists\
 you express the relation using a "predicate" that is expressive yet straightforward (max 5 words). Think about a\
 predicate that could be used in an ontology for a knowledge graph.
Your output is the list of relations formatted as RDF triplets, that you format with an initial hyphen this way:
`- <entity> (<entity index>)|||<predicate>|||<entity> (<entity index>)`
The user text language may be anything, but your output should be {output_language}!
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


def extract_relations(gpt: GPTOpenIE, sentence: str, sentence_entities_ids: list[int],
                      entities: list[dict], output_language: str) -> list[dict]:
    sentence_entities = [(i + ENTITY_COUNT_START, entities[i],) for i in sentence_entities_ids]
    entities_list_str = '\n'.join([f'{i}) {e["label"]}' for i, e in sentence_entities])

    answer = gpt.chat_completion(
        system=SYSTEM.format(output_language=output_language),
        user=USER.format(entities_numbered_list=entities_list_str, context_sentence=sentence)
    )
    return _answer_parser(answer, entities)


def _answer_parser(answer: str, entities: list[dict]) -> list[dict]:
    answer_lines = answer.strip().split('\n')
    parsed_answer_lines = [_parse_answer_line(line) for line in answer_lines if _is_valid_answer_line_pattern(line)]

    n_dropped_lines = len(answer_lines) - len(parsed_answer_lines)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(answer_lines)} wrongly spelled relation information.')

    consistent_answers = [answer for answer in parsed_answer_lines if _check_mention_consistency(answer, entities)]
    n_dropped_lines = len(parsed_answer_lines) - len(consistent_answers)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(parsed_answer_lines)} inconsistent relation information.')

    return consistent_answers


def _is_valid_answer_line_pattern(answer_line: str) -> bool:
    # Regular expression to describe the full pattern of a line in the answer
    is_valid = bool(re.fullmatch(
        r'- ?([^|]+)\(\d+\) ?'  # Entity label and ID
        r'([|]{3})'              # Separator
        r'([^|]+)'               # Predicate label
        r'([|]{3})'              # Separator
        r'([^|]+)\(\d+\)',      # Object label and ID
        answer_line.strip()))
    if not is_valid:
        LOGGER.warning(f'Wrongly spelled relation information: "{answer_line}".')

    return is_valid


def _parse_answer_line(answer_line: str) -> dict:
    # Skip the starting hyphen '- ' and split by '|||' to get the triplets components
    subject, predicate, object = re.sub(r'- ?', '', answer_line.strip()).split('|||')

    subject_label, subject_id = _parse_entity_string(subject)
    predicate_label = predicate.strip()
    object_label, object_id = _parse_entity_string(object)

    return {
        'subj_label': subject_label,
        'subj_id': subject_id,
        'pred_label': predicate_label,
        'obj_label': object_label,
        'obj_id': object_id
    }


def _parse_entity_string(e_string: str) -> tuple[str, int]:
    e_string = e_string.strip()

    # Skip the ending closing parenthesis
    assert e_string.endswith(')')  # if `_is_valid_answer_line_pattern` made its job, this should be true
    e_string = e_string[:-1]

    # Split the entity label from its ID using the opening parenthesis that encloses the ID
    e_label, e_id_str = re.sub(r' ?\(', '|||', e_string).split('|||')

    # Return the cleaned label and the entity ID
    return e_label.strip(), int(e_id_str) - ENTITY_COUNT_START


def _check_mention_consistency(relation_dict: dict, entities: list[dict]) -> bool:
    for e_str in ['subj', 'obj']:
        e_index = relation_dict[f'{e_str}_id']
        # There is no such entity
        if not (0 <= e_index < len(entities)):
            return False
        # Forgive little syntactic differences: do not request perfect string matches
        seq_match = SequenceMatcher(None, relation_dict[f'{e_str}_label'].lower(), entities[e_index]['label'].lower())
        if seq_match.ratio() < LABEL_SIM_RATIO_TH:
            return False
    return True
