"""Tests for the node classes generated from momapy types.

Importing these modules scans the corresponding momapy module and creates a
node class for every map, model, layout and element class it finds, so the
import itself is what is being checked here.
"""

import momapy_kb.lpg.celldesigner
import momapy_kb.lpg.sbgn.af
import momapy_kb.lpg.sbgn.pd


class TestCellDesignerNodeClasses:

    def test_map_node_class(self):
        assert hasattr(momapy_kb.lpg.celldesigner, "CellDesignerMap")

    def test_model_element_node_class(self):
        assert hasattr(momapy_kb.lpg.celldesigner, "GenericProtein")


class TestSBGNAFNodeClasses:

    def test_map_node_class(self):
        assert hasattr(momapy_kb.lpg.sbgn.af, "SBGNAFMap")

    def test_model_element_node_class(self):
        assert hasattr(momapy_kb.lpg.sbgn.af, "BiologicalActivity")


class TestSBGNPDNodeClasses:

    def test_map_node_class(self):
        assert hasattr(momapy_kb.lpg.sbgn.pd, "SBGNPDMap")

    def test_model_element_node_class(self):
        assert hasattr(momapy_kb.lpg.sbgn.pd, "Macromolecule")
