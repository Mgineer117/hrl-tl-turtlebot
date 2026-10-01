from pydantic import BaseModel, computed_field, field_serializer
from rl_pipeline.core import YAMLReaderMixin


class AvailableSpecs(BaseModel, YAMLReaderMixin):
    predicates: list[str]
    specifications: list[str]

    @computed_field
    @property
    def num_specifications(self) -> int:
        return len(self.specifications)

    # Sort predicates and specifications for consistent ordering
    @field_serializer("predicates")
    def sort_predicates(self, predicates: list[str]) -> list[str]:
        return sorted(predicates)

    @field_serializer("specifications")
    def sort_specifications(self, specifications: list[str]) -> list[str]:
        return sorted(specifications)
