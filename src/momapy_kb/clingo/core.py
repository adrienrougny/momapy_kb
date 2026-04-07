import typing

import fieldz_kb.clingo.session


class Session:

    def __init__(self) -> None:
        self._session = fieldz_kb.clingo.session.Session()

    def __enter__(self) -> "Session":
        self._session.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        return self._session.__exit__(exc_type, exc_val, exc_tb)

    def make_facts_from_object(
        self,
        obj: object,
        id_to_object: dict | None = None,
        integration_mode: typing.Literal["hash", "id"] = "id",
        exclude_from_integration: tuple[type, ...] | None = None,
    ) -> list:
        return self._session.make_facts_from_object(
            obj,
            id_to_object=id_to_object,
            integration_mode=integration_mode,
            exclude_from_integration=exclude_from_integration,
        )

    def get_or_make_predicate_classes_from_type(
        self,
        type_: type,
        module: str | None = None,
        make_predicate_classes_recursively: bool = True,
    ) -> list:
        return self._session.get_or_make_predicate_classes_from_type(
            type_,
            module=module,
            make_predicate_classes_recursively=make_predicate_classes_recursively,
        )

    def make_ontology_rules_from_type(self, type_: type) -> list[str]:
        return self._session.make_ontology_rules_from_type(type_)
