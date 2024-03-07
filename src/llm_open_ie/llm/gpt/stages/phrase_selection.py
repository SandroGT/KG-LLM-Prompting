

def select_phrase(self, text: str, entity_id: int, entities: list[dict]) -> str:
    e_dict = entities[entity_id]
    return f'{e_dict["label"]}: {_lower_first(e_dict["description"])}'


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:]
