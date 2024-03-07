from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def chat_completion(self, user: str | list[str], system: str = None,
                        temperature: float = 0, top_p: float = 0, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_num_tokens(self, text):
        raise NotImplementedError


class LLMOpenIE(LLM):
    @abstractmethod
    def entity_extraction(self, text: str, output_language: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def phrase_selection(self, text: str, entity_id: int, entities: list[dict]) -> str:
        raise NotImplementedError

    @abstractmethod
    def mention_recognition(self, sentence: str, entities: list[dict]) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def relation_extraction(self, sentence: str, sentence_entities_ids: list[int],
                            entities: list[dict], output_language: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def predicate_description(self, sentence: str, triplets: list[dict], output_language: str) -> None:
        raise NotImplementedError
