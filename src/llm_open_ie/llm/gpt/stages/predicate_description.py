from difflib import get_close_matches
import re

from llm_open_ie.llm.gpt import GPTOpenIE
from llm_open_ie.logger import LOGGER

SYSTEM = '''\
You provide an extended description of the "predicates" in a set of RDF triplets, which means you give a summary of\
 the type of relation, characteristics, behaviors, or associations that the "predicate" is expressing between the\
 "subject" and "object". The description must be general and reusable, so it must make no explicit references to the\
 "subject" and "object".
The user provides you a sentence from which the triplets are extracted and a list of triplets formatted this way:
`- <subject>|||<predicate>|||<object>`
For each unique predicate, you provide the so said description, formatting your output this way:
`- <predicate>|||<predicate description><new line>`
The user text language may be anything, but your output should be {output_language}!
'''

USER = '''\
Sentence:
```
{context_sentence}
```

User triplets:
```
{triplets_list}
```
'''

LABEL_SIM_RATIO_TH = 0.90


def describe_predicates(gpt: GPTOpenIE, sentence: str, triplets: list[dict], output_language: str) -> None:
    triplets_str = '\n'.join(
        [f'- {t_dict["subj_label"]}|||{t_dict["pred_label"]}|||{t_dict["obj_label"]}'for t_dict in triplets]
    )

    answer = gpt.chat_completion(
        system=SYSTEM.format(output_language=output_language),
        user=USER.format(context_sentence=sentence, triplets_list=triplets_str)
    )
    _answer_parser(answer, triplets)


def _answer_parser(answer: str, triplets: list[dict]):
    answer_lines = answer.strip().split('\n')
    parsed_answer_lines = [_parse_answer_line(line) for line in answer_lines if _is_valid_answer_line_pattern(line)]

    n_dropped_lines = len(answer_lines) - len(parsed_answer_lines)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(answer_lines)} wrongly spelled predicate information.')

    consistent_answers = [answer for answer in parsed_answer_lines if _check_mention_consistency(answer, triplets)]
    n_dropped_lines = len(parsed_answer_lines) - len(consistent_answers)
    if n_dropped_lines > 0:
        LOGGER.warning(f'Dropped {n_dropped_lines}/{len(parsed_answer_lines)} inconsistent predicate information.')

    predicates_dict = {d['label'].lower(): d['description'] for d in consistent_answers}
    predicates_keys = list(predicates_dict.keys())
    for triplet_dict in triplets:
        pred_key = get_close_matches(triplet_dict['pred_label'], predicates_keys, n=1)
        if pred_key:
            pred_key = pred_key[0]
            triplet_dict['pred_description'] = predicates_dict[pred_key]
        else:
            triplet_dict['pred_description'] = None


def _is_valid_answer_line_pattern(answer_line: str) -> bool:

    # TODO CONTINUE HERE
    # Regular expression to describe the full pattern of a line in the answer
    is_valid = bool(re.fullmatch(
        r'- ?([^|]+)'  # Predicate label
        r'([|]{3})'    # Separator
        r'([^|]+)',    # Predicate description
        answer_line.strip()))
    if not is_valid:
        LOGGER.warning(f'Wrongly spelled predicate information: "{answer_line}".')

    return is_valid


def _parse_answer_line(answer_line: str) -> dict:
    # Skip the starting hyphen '- ' and split by '|||' to get the predicate information
    label, description = re.sub(r'- ?', '', answer_line.strip()).split('|||')

    return {
        'label': label.strip(),
        'description': description.strip()
    }


def _check_mention_consistency(predicate_dict: dict, triplets: list[dict]):
    all_predicates = list({t['pred_label'].lower() for t in triplets})
    # Forgive little syntactic differences: do not request perfect string matches
    match = get_close_matches(predicate_dict['label'].lower(), all_predicates, n=1, cutoff=LABEL_SIM_RATIO_TH)
    return bool(match)
