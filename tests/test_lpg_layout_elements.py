"""Tests for momapy_kb LPG layout element queries."""

import pathlib

import pytest

import momapy.core.elements
import momapy.core.layout

import momapy_kb.lpg.session

TESTS_DIR = pathlib.Path(__file__).parent
SBGN_MAPS = sorted(TESTS_DIR.glob("sbgn/maps/**/*.sbgn"))


def _save_map(session, map_file):
    session.save_from_file(
        str(map_file), return_type="map", integration_mode="hash"
    )


def _first_model_element_node(session):
    results = session.execute_query(
        "MATCH (n:ModelElement) RETURN n LIMIT 1", resolve_nodes=True
    )
    if not results:
        pytest.skip("no model element in map")
    return list(results[0].values())[0]


@pytest.mark.usefixtures("clear_database")
class TestLayoutElementNodes:
    """Tests for get_layout_element_nodes_from_model_element_node."""

    @pytest.mark.parametrize(
        "map_file",
        SBGN_MAPS,
        ids=[p.stem for p in SBGN_MAPS],
    )
    def test_returns_layout_element_nodes(self, session, map_file):
        _save_map(session, map_file)
        node = _first_model_element_node(session)
        nodes = session.get_layout_element_nodes_from_model_element_node(node)
        assert isinstance(nodes, list)


@pytest.mark.usefixtures("clear_database")
class TestMakeLayoutElements:
    """Tests for make_layout_elements_from_model_element_node."""

    @pytest.mark.parametrize(
        "map_file",
        SBGN_MAPS,
        ids=[p.stem for p in SBGN_MAPS],
    )
    def test_returns_layout_element_objects(self, session, map_file):
        _save_map(session, map_file)
        node = _first_model_element_node(session)
        layout_elements = session.make_layout_elements_from_model_element_node(node)
        assert all(
            isinstance(e, momapy.core.elements.LayoutElement) for e in layout_elements
        )


@pytest.mark.usefixtures("clear_database")
class TestCypherQueryAsLayoutElements:
    """Tests for cypher_query_as_layout_elements."""

    @pytest.mark.parametrize(
        "map_file",
        SBGN_MAPS,
        ids=[p.stem for p in SBGN_MAPS],
    )
    def test_returns_one_row_per_match(self, session, map_file):
        _save_map(session, map_file)
        query = "MATCH (n:Macromolecule) RETURN n"
        expected = len(session.execute_query(query))
        if not expected:
            pytest.skip("no macromolecule in map")
        results = session.cypher_query_as_layout_elements(query)
        assert len(results) == expected

    @pytest.mark.parametrize(
        "map_file",
        SBGN_MAPS,
        ids=[p.stem for p in SBGN_MAPS],
    )
    def test_rows_contain_layout_elements(self, session, map_file):
        _save_map(session, map_file)
        results = session.cypher_query_as_layout_elements(
            "MATCH (n:Macromolecule) RETURN n"
        )
        if not results:
            pytest.skip("no macromolecule in map")
        assert any(row for row in results)
        for row in results:
            assert all(
                isinstance(e, momapy.core.elements.LayoutElement) for e in row
            )

    def test_empty_result_returns_empty_list(self, session):
        results = session.cypher_query_as_layout_elements(
            "MATCH (n:Macromolecule) WHERE n.label = 'no_such_label' RETURN n"
        )
        assert results == []

    @pytest.mark.parametrize(
        "map_file",
        SBGN_MAPS,
        ids=[p.stem for p in SBGN_MAPS],
    )
    def test_arcs_bring_their_source_and_target(self, session, map_file):
        _save_map(session, map_file)
        results = session.cypher_query_as_layout_elements(
            "MATCH (n:GenericProcess) RETURN n"
        )
        if not results:
            pytest.skip("no generic process in map")
        for row in results:
            arcs = [
                e
                for element in row
                for e in [element] + element.descendants()
                if isinstance(e, momapy.core.layout.Arc)
            ]
            for arc in arcs:
                for attr_name in ["source", "target"]:
                    attr_value = getattr(arc, attr_name, None)
                    if isinstance(attr_value, momapy.core.elements.LayoutElement):
                        assert attr_value in row
