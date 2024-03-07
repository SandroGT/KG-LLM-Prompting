from getpass import getpass
import os
import time

from openai import OpenAI, OpenAIError, APIError, RateLimitError
import tiktoken

from llm_open_ie.llm import LLMOpenIE
from llm_open_ie.logger import LOGGER


class GPTOpenIE(LLMOpenIE):
    __last_request_timestamp: float
    __rate_limit_sleep: float
    __client: OpenAI
    __encoder: tiktoken.Encoding

    def __init__(self, api_key_input: str = 'environ', rate_limit_sleep: float = 10.0):
        api_key_input_values = ['environ', 'keyboard']
        if api_key_input == api_key_input_values[0]:
            openai_api_key = os.environ.get('OPENAI_API_KEY', None)
            if not openai_api_key:
                raise ValueError(
                    '`api_key_input` is set to `environ`, but the environmental variable `OPENAI_API_KEY` is not set.')
        elif api_key_input == api_key_input_values[1]:
            openai_api_key = getpass(prompt='Insert your OpenAI API key: ')
        else:
            options_str = ', '.join([f'`{v}`' for v in api_key_input_values])
            raise ValueError(f'`{api_key_input}` is not a valid option: choose one between {options_str}')

        self.__last_request_timestamp = 0.0
        self.__rate_limit_sleep = rate_limit_sleep
        self.__client = OpenAI(api_key=openai_api_key)
        self.__encoder = tiktoken.get_encoding('cl100k_base')

    def chat_completion(self, user: str, system: str = None,
                        model: str = 'gpt-3.5-turbo-0301', temperature: float = 0, top_p: float = 0):
        # More info here: https://platform.openai.com/docs/api-reference/chat/create
        messages = [{'role': 'user', 'content': user}]
        if system is not None:
            messages.insert(0, {'role': 'system', 'content': system})

        elapsed = time.time() - self.__last_request_timestamp
        if elapsed < self.__rate_limit_sleep:
            time.sleep(self.__rate_limit_sleep - elapsed)

        response = None
        need_completion = True
        while need_completion:
            try:
                response = self.__client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p
                )
                need_completion = False
            except (OpenAIError, APIError, RateLimitError) as e:
                LOGGER.warning(
                    f'OpenAI `{type(e).__name__}` faced.'
                    f' Trying request again in {self.__rate_limit_sleep: .2f} seconds.')
                time.sleep(self.__rate_limit_sleep)

        answer = response.choices[0].message.content

        return answer

    def get_num_tokens(self, text):
        return len(self.__encoder.encode(text))

    def entity_extraction(self, text: str, output_language: str) -> list[dict]:
        from llm_open_ie.llm.gpt.stages.entity_extraction import extract_entities
        return extract_entities(self, text, output_language)

    def phrase_selection(self, text: str, entity_id: int, entities: list[dict]) -> str:
        from llm_open_ie.llm.gpt.stages.phrase_selection import select_phrase
        return select_phrase(self, text, entity_id, entities)

    def mention_recognition(self, sentence: str, entities: list[dict]) -> list[int]:
        from llm_open_ie.llm.gpt.stages.mention_recognition import recognize_mentions
        return recognize_mentions(self, sentence, entities)

    def relation_extraction(self, sentence: str, sentence_entities_ids: list[int],
                            entities: list[dict], output_language: str) -> list[dict]:
        from llm_open_ie.llm.gpt.stages.relation_extraction import extract_relations
        return extract_relations(self, sentence, sentence_entities_ids, entities, output_language)

    def predicate_description(self, sentence: str, triplets: list[dict], output_language: str) -> None:
        from llm_open_ie.llm.gpt.stages.predicate_description import describe_predicates
        describe_predicates(self, sentence, triplets, output_language)
