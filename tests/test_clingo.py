"""Tests for momapy_kb clingo session.

These tests verify predicate class generation and fact creation.
No external services are required.
"""

import dataclasses
import typing

import pytest
import clorm

import momapy_kb.clingo.core
import momapy_kb.clingo.celldesigner


@pytest.fixture
def clingo_session():
    """Provide a fresh clingo session for each test."""
    return momapy_kb.clingo.core.Session()


class TestSessionLifecycle:
    """Tests for clingo session creation and context management."""

    def test_session_creation(self):
        session = momapy_kb.clingo.core.Session()
        assert session is not None

    def test_session_context_manager(self):
        with momapy_kb.clingo.core.Session() as session:
            assert session is not None


class TestPredicateClassGeneration:
    """Tests for predicate class generation from types."""

    def test_simple_class_generates_predicate(self, clingo_session):
        @dataclasses.dataclass
        class Gene:
            name: str
            chromosome: int

        predicate_classes = (
            clingo_session.get_or_make_predicate_classes_from_type(Gene)
        )
        assert len(predicate_classes) >= 1
        assert issubclass(predicate_classes[0], clorm.Predicate)

    def test_predicate_class_has_id_field(self, clingo_session):
        @dataclasses.dataclass
        class Metabolite:
            concentration: int

        predicate_classes = (
            clingo_session.get_or_make_predicate_classes_from_type(Metabolite)
        )
        main_predicate_class = predicate_classes[0]
        fact = main_predicate_class("test_id")
        assert fact.id_ == "test_id"

    def test_predicate_class_caching(self, clingo_session):
        @dataclasses.dataclass
        class Protein:
            sequence: str

        classes_1 = clingo_session.get_or_make_predicate_classes_from_type(Protein)
        classes_2 = clingo_session.get_or_make_predicate_classes_from_type(Protein)
        assert classes_1[0] is classes_2[0]

    def test_field_predicate_classes_generated(self, clingo_session):
        @dataclasses.dataclass
        class Enzyme:
            name: str
            ec_number: str

        predicate_classes = (
            clingo_session.get_or_make_predicate_classes_from_type(Enzyme)
        )
        assert len(predicate_classes) >= 3

    def test_nested_fieldz_class_field(self, clingo_session):
        @dataclasses.dataclass
        class Organism:
            name: str

        @dataclasses.dataclass
        class GeneWithOrganism:
            organism: Organism

        predicate_classes = (
            clingo_session.get_or_make_predicate_classes_from_type(GeneWithOrganism)
        )
        assert len(predicate_classes) >= 2

    def test_optional_field_type(self, clingo_session):
        @dataclasses.dataclass
        class ProteinOptional:
            name: typing.Optional[str] = None

        predicate_classes = (
            clingo_session.get_or_make_predicate_classes_from_type(ProteinOptional)
        )
        assert len(predicate_classes) >= 1

    def test_list_field_type(self, clingo_session):
        @dataclasses.dataclass
        class PathwayWithReactions:
            reactions: typing.List[str]

        predicate_classes = (
            clingo_session.get_or_make_predicate_classes_from_type(
                PathwayWithReactions
            )
        )
        assert len(predicate_classes) >= 2

    def test_unsupported_type_raises_error(self, clingo_session):
        with pytest.raises(ValueError, match="not supported"):
            clingo_session.get_or_make_predicate_classes_from_type(int)


class TestFactGeneration:
    """Tests for converting objects to facts."""

    def test_simple_object_facts(self, clingo_session):
        @dataclasses.dataclass
        class Gene:
            name: str
            chromosome: int

        obj = Gene(name="TP53", chromosome=17)
        facts = clingo_session.make_facts_from_object(obj)
        assert len(facts) > 0

    def test_entity_fact_present(self, clingo_session):
        @dataclasses.dataclass
        class Metabolite:
            concentration: int

        obj = Metabolite(concentration=42)
        facts = clingo_session.make_facts_from_object(obj)
        entity_facts = [f for f in facts if type(f).__name__ == "metabolite"]
        assert len(entity_facts) == 1
        assert entity_facts[0].id_.startswith("id_")

    def test_fact_values_are_correct(self, clingo_session):
        @dataclasses.dataclass
        class Measurement:
            value: int

        obj = Measurement(value=42)
        facts = clingo_session.make_facts_from_object(obj)
        value_facts = [f for f in facts if hasattr(f, "value")]
        assert any(f.value == 42 for f in value_facts)

    def test_string_field_value(self, clingo_session):
        @dataclasses.dataclass
        class Species:
            name: str

        obj = Species(name="Homo sapiens")
        facts = clingo_session.make_facts_from_object(obj)
        value_facts = [f for f in facts if hasattr(f, "value")]
        assert any(f.value == "Homo sapiens" for f in value_facts)

    def test_none_value_skipped(self, clingo_session):
        @dataclasses.dataclass
        class ProteinOptional:
            name: str
            alias: typing.Optional[str] = None

        obj = ProteinOptional(name="p53", alias=None)
        facts = clingo_session.make_facts_from_object(obj)
        assert len(facts) > 0
        value_facts = [f for f in facts if hasattr(f, "value")]
        assert len(value_facts) == 1
        assert value_facts[0].value == "p53"

    def test_nested_object_facts(self, clingo_session):
        @dataclasses.dataclass
        class CellLocation:
            compartment: str

        @dataclasses.dataclass
        class ProteinLocated:
            name: str
            location: CellLocation

        location = CellLocation(compartment="cytoplasm")
        protein = ProteinLocated(name="insulin", location=location)
        facts = clingo_session.make_facts_from_object(protein)

        entity_type_names = {
            type(f).__name__ for f in facts if not hasattr(f, "value")
        }
        assert "proteinLocated" in entity_type_names
        assert "cellLocation" in entity_type_names

    def test_list_of_base_types(self, clingo_session):
        @dataclasses.dataclass
        class GeneWithSynonyms:
            synonyms: typing.List[str]

        obj = GeneWithSynonyms(synonyms=["BRCA1", "FANCS", "RNF53"])
        facts = clingo_session.make_facts_from_object(obj)
        assert len(facts) > 0
        value_facts = [f for f in facts if hasattr(f, "value")]
        assert len(value_facts) == 3
        values = {f.value for f in value_facts}
        assert values == {"BRCA1", "FANCS", "RNF53"}

    def test_deterministic_ids(self, clingo_session):
        @dataclasses.dataclass
        class Enzyme:
            name: str

        obj = Enzyme(name="hexokinase")
        facts = clingo_session.make_facts_from_object(obj)
        entity_fact = [f for f in facts if type(f).__name__ == "enzyme"][0]
        assert entity_fact.id_ == "id_0"

    def test_deterministic_ids_sequential(self, clingo_session):
        @dataclasses.dataclass
        class Reaction:
            name: str

        obj1 = Reaction(name="step_1")
        obj2 = Reaction(name="step_2")
        facts1 = clingo_session.make_facts_from_object(obj1)
        facts2 = clingo_session.make_facts_from_object(obj2)

        entity1 = [f for f in facts1 if type(f).__name__ == "reaction"][0]
        entity2 = [f for f in facts2 if type(f).__name__ == "reaction"][0]
        assert entity1.id_ == "id_0"
        assert entity2.id_ == "id_1"

    def test_empty_list_field(self, clingo_session):
        @dataclasses.dataclass
        class EmptyPathway:
            reactions: typing.List[str] = dataclasses.field(default_factory=list)

        obj = EmptyPathway(reactions=[])
        facts = clingo_session.make_facts_from_object(obj)
        entity_facts = [f for f in facts if type(f).__name__ == "emptyPathway"]
        assert len(entity_facts) == 1
        value_facts = [f for f in facts if hasattr(f, "value")]
        assert len(value_facts) == 0


class TestOntologyRules:
    """Tests for ontology rule generation."""

    def test_generates_rules_for_subclass(self, clingo_session):
        @dataclasses.dataclass
        class BioEntity:
            name: str

        @dataclasses.dataclass
        class Gene(BioEntity):
            chromosome: int = 0

        clingo_session.get_or_make_predicate_classes_from_type(BioEntity)
        clingo_session.get_or_make_predicate_classes_from_type(Gene)
        rules = clingo_session.make_ontology_rules_from_type(Gene)
        assert len(rules) > 0
        assert any("bioEntity" in rule and "gene" in rule for rule in rules)

    def test_no_rules_for_base_class(self, clingo_session):
        @dataclasses.dataclass
        class Standalone:
            name: str

        clingo_session.get_or_make_predicate_classes_from_type(Standalone)
        rules = clingo_session.make_ontology_rules_from_type(Standalone)
        assert len(rules) == 0


class TestCellDesignerOntologyRules:
    """Tests for the ontology rules generated from momapy celldesigner types."""

    def test_ontology_rules_generated(self):
        assert len(momapy_kb.clingo.celldesigner.ontology_rules) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
