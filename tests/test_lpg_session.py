"""Tests for momapy_kb LPG session: save, query, collections, and round-trips."""

import dataclasses
import pathlib

import pytest

import momapy_kb.lpg.session
import momapy_kb.core


@dataclasses.dataclass
class Gene:
    name: str
    chromosome: int


@dataclasses.dataclass
class Organism:
    name: str


@dataclasses.dataclass
class GeneWithOrganism:
    name: str
    organism: Organism


@dataclasses.dataclass
class Protein:
    name: str
    sequence: str
    molecular_weight: float
    is_enzyme: bool


@dataclasses.dataclass
class CellLocation:
    compartment: str
    membrane: str


@dataclasses.dataclass
class ProteinWithLocation:
    name: str
    location: CellLocation


class TestSessionLifecycle:
    """Tests for session creation and context management."""

    def test_session_has_context(self, session):
        assert session._context is not None

    def test_session_registers_momapy_plugins(self, session):
        import momapy.drawing
        import momapy.core.mapping
        import momapy.utils

        ctx = session._context
        assert momapy.drawing.NoneValueType in ctx.type_to_node_class
        assert momapy.core.mapping.LayoutModelMapping in ctx.type_to_node_class
        assert momapy.utils.FrozenSurjectionDict in ctx.type_to_node_class


@pytest.mark.usefixtures("clear_database")
class TestSaveAndRetrieve:
    """Tests for saving objects and retrieving them."""

    def test_save_and_retrieve_simple_object(self, session):
        gene = Gene(name="TP53", chromosome=17)
        session.save_from_object(gene)

        results = session.execute_query_as_objects("MATCH (n:Gene) RETURN n")
        assert len(results) == 1

        retrieved = results[0][0]
        assert isinstance(retrieved, Gene)
        assert retrieved.name == "TP53"
        assert retrieved.chromosome == 17

    def test_save_and_retrieve_with_all_base_types(self, session):
        protein = Protein(
            name="p53", sequence="MEEPQ", molecular_weight=43.7, is_enzyme=False
        )
        session.save_from_object(protein)

        results = session.execute_query_as_objects("MATCH (n:Protein) RETURN n")
        assert len(results) == 1

        retrieved = results[0][0]
        assert retrieved.name == "p53"
        assert retrieved.sequence == "MEEPQ"
        assert retrieved.molecular_weight == 43.7
        assert retrieved.is_enzyme is False

    def test_save_and_retrieve_with_relationship(self, session):
        organism = Organism(name="Homo sapiens")
        gene = GeneWithOrganism(name="BRCA1", organism=organism)
        session.save_from_object(gene)

        results = session.execute_query(
            "MATCH (g:GeneWithOrganism)-[:HAS_ORGANISM]->(o:Organism) RETURN g, o"
        )
        assert len(results) == 1

    def test_save_and_retrieve_nested_objects(self, session):
        protein = ProteinWithLocation(
            name="insulin receptor",
            location=CellLocation(compartment="cytoplasm", membrane="plasma"),
        )
        session.save_from_object(protein)

        results = session.execute_query_as_objects(
            "MATCH (n:ProteinWithLocation) RETURN n"
        )
        assert len(results) == 1

        retrieved = results[0][0]
        assert retrieved.name == "insulin receptor"
        assert retrieved.location.compartment == "cytoplasm"
        assert retrieved.location.membrane == "plasma"

    def test_save_multiple_objects(self, session):
        genes = [
            Gene(name="TP53", chromosome=17),
            Gene(name="BRCA1", chromosome=17),
            Gene(name="EGFR", chromosome=7),
        ]
        session.save_from_objects(genes)

        results = session.execute_query("MATCH (n:Gene) RETURN n")
        assert len(results) == 3

    def test_save_with_integration_mode_hash(self, session):
        @dataclasses.dataclass(frozen=True)
        class Species:
            name: str

        species = Species(name="Homo sapiens")
        session.save_from_objects(
            [species, species], integration_mode="hash"
        )

        results = session.execute_query("MATCH (n:Species) RETURN n")
        assert len(results) == 1

    def test_save_with_integration_mode_id(self, session):
        gene = Gene(name="TP53", chromosome=17)
        session.save_from_objects([gene, gene], integration_mode="id")

        results = session.execute_query("MATCH (n:Gene) RETURN n")
        assert len(results) == 1


@pytest.mark.usefixtures("clear_database")
class TestExecuteQuery:
    """Tests for query execution."""

    def test_execute_query_returns_list_of_dicts(self, session):
        gene = Gene(name="TP53", chromosome=17)
        session.save_from_object(gene)

        results = session.execute_query("MATCH (n:Gene) RETURN n.name AS name")
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["name"] == "TP53"

    def test_execute_query_with_params(self, session):
        genes = [
            Gene(name="TP53", chromosome=17),
            Gene(name="EGFR", chromosome=7),
        ]
        session.save_from_objects(genes)

        results = session.execute_query(
            "MATCH (n:Gene) WHERE n.chromosome = $chrom RETURN n.name AS name",
            params={"chrom": 7},
        )
        assert len(results) == 1
        assert results[0]["name"] == "EGFR"

    def test_execute_query_as_objects_returns_python_objects(self, session):
        gene = Gene(name="BRCA1", chromosome=17)
        session.save_from_object(gene)

        results = session.execute_query_as_objects("MATCH (n:Gene) RETURN n")
        assert len(results) == 1
        assert isinstance(results[0][0], Gene)

    def test_execute_query_empty_result(self, session):
        results = session.execute_query("MATCH (n:NonExistent) RETURN n")
        assert len(results) == 0


@pytest.mark.usefixtures("clear_database")
class TestDeleteAll:
    """Tests for delete_all method."""

    def test_delete_all_clears_database(self, session):
        gene = Gene(name="TP53", chromosome=17)
        session.save_from_object(gene)

        results = session.execute_query("MATCH (n) RETURN n")
        assert len(results) > 0

        session.delete_all()

        results = session.execute_query("MATCH (n) RETURN n")
        assert len(results) == 0

    def test_delete_all_on_empty_database(self, session):
        session.delete_all()
        results = session.execute_query("MATCH (n) RETURN n")
        assert len(results) == 0


@pytest.mark.usefixtures("clear_database")
class TestSaveFromFile:
    """Tests for saving objects from files."""

    def test_save_from_file_celldesigner(self, session):
        test_file = pathlib.Path(__file__).parent / "celldesigner_example.xml"
        if not test_file.exists():
            pytest.skip("celldesigner_example.xml not found")
        session.save_from_file(
            str(test_file), return_type="model", integration_mode="hash"
        )

        results = session.execute_query("MATCH (n) RETURN count(n) AS count")
        assert results[0]["count"] > 0

    def test_save_from_file_sbgn(self, session):
        test_file = pathlib.Path(__file__).parent / "test.xml"
        if not test_file.exists():
            pytest.skip("test.xml not found")
        session.save_from_file(
            str(test_file), return_type="model", integration_mode="hash"
        )

        results = session.execute_query("MATCH (n) RETURN count(n) AS count")
        assert results[0]["count"] > 0


@pytest.mark.usefixtures("clear_database")
class TestCollections:
    """Tests for collection management."""

    def test_save_collections_from_entries(self, session):
        @dataclasses.dataclass(frozen=True)
        class SimpleModel:
            name: str

        entry1 = momapy_kb.core.CollectionEntry(
            id_="model1",
            model=SimpleModel(name="test_model_1"),
        )
        entry2 = momapy_kb.core.CollectionEntry(
            id_="model2",
            model=SimpleModel(name="test_model_2"),
        )
        session.save_collections_from_entries(
            [("TestCollection", [entry1, entry2])]
        )

        results = session.execute_query(
            "MATCH (n:Collection) RETURN n.name AS name"
        )
        assert len(results) == 1
        assert results[0]["name"] == "TestCollection"

    def test_save_collections_with_delete_all(self, session):
        @dataclasses.dataclass(frozen=True)
        class SimpleModel:
            name: str

        gene = Gene(name="pre_existing", chromosome=1)
        session.save_from_object(gene)

        entry = momapy_kb.core.CollectionEntry(
            id_="model1",
            model=SimpleModel(name="test"),
        )
        session.save_collections_from_entries(
            [("TestCollection", [entry])],
            delete_all=True,
        )

        results = session.execute_query("MATCH (n:Gene) RETURN n")
        assert len(results) == 0

        results = session.execute_query("MATCH (n:Collection) RETURN n")
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
