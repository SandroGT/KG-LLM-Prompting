from llm_open_ie.llm import LLMOpenIE
from llm_open_ie.logger import LOGGER


def oie_pipeline(text: str, llm_oie: LLMOpenIE, output_language='English') -> tuple[list[dict], list[dict]]:
    output_language = output_language[0].upper() + output_language[1:].lower()

    # 0) Entity Extraction: find all the entities in the text, with a label, description and list of types for each one
    text_entities = llm_oie.entity_extraction(text, output_language)
    LOGGER.info(f'Found {len(text_entities)} entities in text.')
    for e_dict in text_entities:
        LOGGER.debug(f'{e_dict["label"]} ({e_dict["types"]})\n\t{e_dict["description"]}')

    # Iterative triple extraction: repeat for each one of the found entities
    text_triplets = []
    for e_i, e_dict in enumerate(text_entities):
        LOGGER.info(f'Checking entity {e_i+1}/{len(text_entities)}: "{e_dict["label"]}"')

        # 1) Phrase Selection: get a phrase focusing on the actual entity
        e_sentence = llm_oie.phrase_selection(text, e_i, text_entities)
        LOGGER.debug(f'Searching for mentions of other entities in "{e_dict["label"]}" sentence: "{e_sentence}".')

        # 2) Mention Recognition: find which other entities are mentioned in the artificial sentence
        sentence_mentioned_entities_ids = llm_oie.mention_recognition(e_sentence, text_entities)
        mentions_str = ', '.join([f'"{text_entities[i]["label"]}"' for i in sentence_mentioned_entities_ids])
        if len(sentence_mentioned_entities_ids) < 2:
            LOGGER.debug(f'Not enough mentions to find relations: you need at least 2 entities to search for triplets.'
                         f' Skipping!')
            continue
        LOGGER.debug(f'Found mentions of {mentions_str}.')

        # 3) Relation Extraction: find the relations between the entities mentioned in the artificial sentence
        sentence_relations = llm_oie.relation_extraction(
            e_sentence, sentence_mentioned_entities_ids, text_entities, output_language)
        triplets_str = '\n'.join([
            f'{t_dict["subj_label"]} | {t_dict["pred_label"]} | {t_dict["obj_label"]}' for t_dict in sentence_relations
        ])
        if len(sentence_relations) == 0:
            LOGGER.debug(f'No triplets found. Skipping!')
            continue
        LOGGER.debug(f'Found triplets:\n{triplets_str}')

        # 4) Predicate Description: get a better description of the predicates used in the extracted relations and add
        #  it directly in the relations dictionary
        llm_oie.predicate_description(e_sentence, sentence_relations, output_language)

        # Add triplets to the global output
        for r in sentence_relations:
            text_triplets.append(r)

    return text_entities, text_triplets
