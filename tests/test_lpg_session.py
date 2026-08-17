"""Tests for momapy_kb LPG session: save, query, collections, and round-trips."""

import dataclasses
import pathlib

import pytest

import momapy.io.core

import fieldz_kb.lpg.graph

import pylpg.node

import momapy_kb.lpg.session
import momapy_kb.core

TESTS_DIR = pathlib.Path(__file__).parent
SBGN_MAPS = sorted(TESTS_DIR.glob("sbgn/maps/**/*.sbgn"))
CELLDESIGNER_MAPS = sorted(TESTS_DIR.glob("celldesigner/maps/**/*.xml"))
ALL_MAPS = SBGN_MAPS + CELLDESIGNER_MAPS
ALL_MAP_PAIRS = list(zip(ALL_MAPS[:-1], ALL_MAPS[1:]))


@dataclasses.dataclass
class Gene:
    name: str
    chromosome: int


@dataclasses.dataclass(frozen=True)
class FrozenGene:
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
        assert momapy.utils.FrozenIdentityMultiDict in ctx.type_to_node_class


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
        session.save_from_objects([species, species], integration_mode="hash")

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

    def test_execute_query_with_resolve_nodes(self, session):
        gene = Gene(name="TP53", chromosome=17)
        session.save_from_object(gene)

        results = session.execute_query(
            "MATCH (n:Gene) RETURN n", resolve_nodes=True
        )
        assert len(results) == 1
        assert isinstance(results[0]["n"], pylpg.node.Node)

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
class TestCollections:
    """Tests for collection management."""

    def test_save_collections_from_entries(self, session):
        @dataclasses.dataclass(frozen=True)
        class SimpleModel:
            name: str

        entry1 = momapy_kb.core.CollectionEntry(
            id_="model1",
            obj=SimpleModel(name="test_model_1"),
        )
        entry2 = momapy_kb.core.CollectionEntry(
            id_="model2",
            obj=SimpleModel(name="test_model_2"),
        )
        session.save_collections_from_entries([("TestCollection", [entry1, entry2])])

        results = session.execute_query("MATCH (n:Collection) RETURN n.name AS name")
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
            obj=SimpleModel(name="test"),
        )
        session.save_collections_from_entries(
            [("TestCollection", [entry])],
            delete_all=True,
        )

        results = session.execute_query("MATCH (n:Gene) RETURN n")
        assert len(results) == 0

        results = session.execute_query("MATCH (n:Collection) RETURN n")
        assert len(results) == 1


@pytest.mark.usefixtures("clear_database")
class TestSaveFromFile:
    """Tests for saving maps from files."""

    @pytest.mark.parametrize(
        "map_file",
        SBGN_MAPS,
        ids=[p.stem for p in SBGN_MAPS],
    )
    def test_save_from_file_sbgn(self, session, map_file):
        session.save_from_file(
            str(map_file), return_type="model", integration_mode="hash"
        )
        results = session.execute_query("MATCH (n) RETURN count(n) AS count")
        assert results[0]["count"] > 0

    @pytest.mark.parametrize(
        "map_file",
        CELLDESIGNER_MAPS,
        ids=[p.stem for p in CELLDESIGNER_MAPS],
    )
    def test_save_from_file_celldesigner(self, session, map_file):
        session.save_from_file(
            str(map_file), return_type="model", integration_mode="hash"
        )
        results = session.execute_query("MATCH (n) RETURN count(n) AS count")
        assert results[0]["count"] > 0


def _assert_membership_edge_counts(session, obj, return_type):
    model_element_count = session.execute_query(
        "MATCH ()-[r:HAS_MODEL_ELEMENT]->() RETURN count(r) AS count"
    )[0]["count"]
    layout_root_count = session.execute_query(
        "MATCH (l:Layout)-[r:HAS_LAYOUT_ELEMENT]->() RETURN count(r) AS count"
    )[0]["count"]
    if return_type == "map":
        assert model_element_count == len(obj.model.descendants())
        assert layout_root_count >= len(obj.layout.descendants())
    elif return_type == "model":
        assert model_element_count == len(obj.descendants())
        assert layout_root_count == 0
    else:
        assert model_element_count == 0
        assert layout_root_count >= len(obj.descendants())


@pytest.mark.usefixtures("clear_database")
class TestMembershipEdges:
    """Tests for with_membership_edges option."""

    @pytest.mark.parametrize("return_type", ["map", "model", "layout"])
    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_save_from_object(self, session, map_file, return_type):
        obj = momapy.io.core.read(str(map_file), return_type=return_type).obj
        session.save_from_object(
            obj, integration_mode="hash", with_membership_edges=True
        )
        _assert_membership_edge_counts(session, obj, return_type)

    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_no_model_edges_when_flag_off(self, session, map_file):
        session.save_from_file(str(map_file), integration_mode="hash")
        count = session.execute_query(
            "MATCH ()-[r:HAS_MODEL_ELEMENT]->() RETURN count(r) AS count"
        )[0]["count"]
        assert count == 0

    @pytest.mark.parametrize(
        "map_file_1,map_file_2",
        ALL_MAP_PAIRS,
        ids=[f"{a.stem}+{b.stem}" for a, b in ALL_MAP_PAIRS],
    )
    def test_hash_mode_two_maps_membership_is_per_model(
        self, session, map_file_1, map_file_2
    ):
        map_1 = momapy.io.core.read(str(map_file_1), return_type="map").obj
        map_2 = momapy.io.core.read(str(map_file_2), return_type="map").obj
        session.save_from_objects(
            [map_1, map_2],
            integration_mode="hash",
            with_membership_edges=True,
        )
        total_model_edges = session.execute_query(
            "MATCH ()-[r:HAS_MODEL_ELEMENT]->() RETURN count(r) AS count"
        )[0]["count"]
        expected = len(map_1.model.descendants()) + len(map_2.model.descendants())
        assert total_model_edges == expected

    @pytest.mark.parametrize("return_type", ["map", "model", "layout"])
    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_save_collections_from_entries(self, session, map_file, return_type):
        obj = momapy.io.core.read(str(map_file), return_type=return_type).obj
        entry = momapy_kb.core.CollectionEntry(id_=map_file.stem, obj=obj)
        session.save_collections_from_entries(
            [("test", [entry])],
            integration_mode="hash",
            with_membership_edges=True,
        )
        _assert_membership_edge_counts(session, obj, return_type)

    @pytest.mark.parametrize("return_type", ["map", "model", "layout"])
    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_save_collections_from_file_paths(self, session, map_file, return_type):
        session.save_collections_from_file_paths(
            [("test", [map_file])],
            return_type=return_type,
            integration_mode="hash",
            with_membership_edges=True,
        )
        obj = momapy.io.core.read(str(map_file), return_type=return_type).obj
        _assert_membership_edge_counts(session, obj, return_type)


@pytest.mark.usefixtures("clear_database")
class TestSaveCollectionsFromFilePaths:
    """Tests for saving collections from file paths."""

    @pytest.mark.parametrize(
        "map_files,name",
        [(SBGN_MAPS, "sbgn"), (CELLDESIGNER_MAPS, "celldesigner")],
        ids=["sbgn", "celldesigner"],
    )
    def test_save_collections_from_file_paths(self, session, map_files, name):
        session.save_collections_from_file_paths(
            [(name, map_files)],
            return_type="model",
        )
        results = session.execute_query("MATCH (n:Collection) RETURN n.name AS name")
        assert len(results) == 1
        assert results[0]["name"] == name


def _node_count(session):
    return session.execute_query("MATCH (n) RETURN count(n) AS count")[0]["count"]


@pytest.mark.usefixtures("clear_database")
class TestObjectKeyToNodeCache:
    """Tests for the object_key_to_node integration cache."""

    def test_execute_query_as_objects_populates_cache(self, session):
        gene = FrozenGene(name="BRCA1", chromosome=17)
        session.save_from_object(gene, integration_mode="hash")

        cache = {}
        results = session.execute_query_as_objects(
            "MATCH (n:FrozenGene) RETURN n", object_key_to_node=cache
        )
        assert list(cache) == [results[0][0]]
        node = cache[results[0][0]]
        assert isinstance(node, fieldz_kb.lpg.graph.BaseNode)

    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_seeded_save_reuses_existing_nodes(self, session, map_file):
        obj = momapy.io.core.read(str(map_file), return_type="model").obj
        session.save_from_object(obj, integration_mode="hash")
        node_count = _node_count(session)

        cache = {}
        results = session.execute_query_as_objects(
            f"MATCH (n:{type(obj).__name__}) RETURN n", object_key_to_node=cache
        )
        session.save_from_object(
            results[0][0], integration_mode="hash", object_key_to_node=cache
        )
        assert _node_count(session) == node_count

    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_unseeded_save_duplicates_nodes(self, session, map_file):
        obj = momapy.io.core.read(str(map_file), return_type="model").obj
        session.save_from_object(obj, integration_mode="hash")
        node_count = _node_count(session)

        results = session.execute_query_as_objects(
            f"MATCH (n:{type(obj).__name__}) RETURN n"
        )
        session.save_from_object(results[0][0], integration_mode="hash")
        assert _node_count(session) > node_count

    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_shared_cache_across_saves_integrates_shared_elements(
        self, session, map_file
    ):
        obj = momapy.io.core.read(str(map_file), return_type="model").obj
        cache = {}
        session.save_from_object(
            obj, integration_mode="hash", object_key_to_node=cache
        )
        node_count = _node_count(session)

        session.save_from_object(
            obj, integration_mode="hash", object_key_to_node=cache
        )
        assert _node_count(session) == node_count

    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_shared_cache_with_membership_edges(self, session, map_file):
        obj = momapy.io.core.read(str(map_file), return_type="model").obj
        cache = {}
        session.save_from_object(
            obj, integration_mode="hash", object_key_to_node=cache
        )
        node_count = _node_count(session)

        session.save_from_object(
            obj,
            integration_mode="hash",
            with_membership_edges=True,
            object_key_to_node=cache,
        )
        count = session.execute_query(
            "MATCH ()-[r:HAS_MODEL_ELEMENT]->() RETURN count(r) AS count"
        )[0]["count"]
        assert count == len(obj.descendants())
        assert _node_count(session) == node_count

    @pytest.mark.parametrize("map_file", ALL_MAPS, ids=[p.stem for p in ALL_MAPS])
    def test_root_seeded_save_emits_no_membership_edges(self, session, map_file):
        obj = momapy.io.core.read(str(map_file), return_type="model").obj
        session.save_from_object(obj, integration_mode="hash")

        cache = {}
        results = session.execute_query_as_objects(
            f"MATCH (n:{type(obj).__name__}) RETURN n", object_key_to_node=cache
        )
        session.save_from_object(
            results[0][0],
            integration_mode="hash",
            with_membership_edges=True,
            object_key_to_node=cache,
        )
        count = session.execute_query(
            "MATCH ()-[r:HAS_MODEL_ELEMENT]->() RETURN count(r) AS count"
        )[0]["count"]
        assert count == 0

    def test_unhashable_object_raises(self, session):
        session.save_from_object({"k": 1}, exclude_from_integration=(dict,))

        with pytest.raises(ValueError, match="not hashable"):
            session.execute_query_as_objects(
                "MATCH (n:Dict) RETURN n", object_key_to_node={}
            )
