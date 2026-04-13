"""Clingo session for momapy_kb.

Provides a Session class for converting momapy objects to clingo
predicates and facts for answer set programming.
"""

import typing

import fieldz_kb.clingo.session


class Session:
    """Session for converting momapy objects to clingo predicates and facts.

    Example:
        >>> import momapy_kb.clingo.core
        >>> with momapy_kb.clingo.core.Session() as session:
        ...     facts = session.make_facts_from_object(my_model)
    """

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
        """Convert a Python object to clingo facts.

        Args:
            obj: The object to convert.
            id_to_object: Optional cache mapping fact IDs to objects.
            integration_mode: How to handle duplicate objects ("hash" or "id").
            exclude_from_integration: Types to exclude from integration logic.

        Returns:
            A list of clorm facts.
        """
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
        """Get or create predicate classes for a given Python type.

        Args:
            type_: The Python type.
            module: Module name for resolving forward references.
            make_predicate_classes_recursively: Whether to create predicates for nested types.

        Returns:
            A list of predicate classes.
        """
        return self._session.get_or_make_predicate_classes_from_type(
            type_,
            module=module,
            make_predicate_classes_recursively=make_predicate_classes_recursively,
        )

    def make_ontology_rules_from_type(self, type_: type) -> list[str]:
        """Generate ontology rules expressing type inheritance as ASP rules.

        Args:
            type_: The Python type to generate rules for.

        Returns:
            A sorted list of ASP rule strings.
        """
        return self._session.make_ontology_rules_from_type(type_)
