"""Tests for momapy_kb LPG utility functions."""

import dataclasses
import io

import pytest
import pylpg.relationship

import fieldz_kb.lpg.core
import fieldz_kb.lpg.graph

import momapy_kb.lpg.utils


@pytest.fixture
def sample_node_class():
    """Create a sample node class with properties and relationships."""

    class HasFriend(pylpg.relationship.Relationship):
        __type__ = "HAS_FRIEND"

    class Person(fieldz_kb.lpg.graph.BaseNode):
        name: str | None = None
        age: int | None = None
        friends = pylpg.relationship.RelationshipTo(HasFriend)

    return Person


class TestGetNodeClass:
    """Tests for get_node_class function."""

    def test_returns_relationship_class(self):
        class MyRel(pylpg.relationship.Relationship):
            __type__ = "MY_REL"

        descriptor = pylpg.relationship.RelationshipTo(MyRel)
        assert momapy_kb.lpg.utils.get_node_class(descriptor) is MyRel


class TestGetRelationshipType:
    """Tests for get_relationship_type function."""

    def test_returns_type_string(self):
        class Knows(pylpg.relationship.Relationship):
            __type__ = "KNOWS"

        descriptor = pylpg.relationship.RelationshipTo(Knows)
        assert momapy_kb.lpg.utils.get_relationship_type(descriptor) == "KNOWS"


class TestGetProperties:
    """Tests for get_properties function."""

    def test_includes_primitive_properties(self, sample_node_class):
        properties = momapy_kb.lpg.utils.get_properties(sample_node_class)
        primitive_props = [p for p in properties if p["kind"] == "property"]
        names = {p["name"] for p in primitive_props}
        assert "name" in names
        assert "age" in names

    def test_includes_relationship_properties(self, sample_node_class):
        properties = momapy_kb.lpg.utils.get_properties(sample_node_class)
        rel_props = [p for p in properties if p["kind"] == "relationship"]
        assert len(rel_props) == 1
        assert rel_props[0]["name"] == "friends"
        assert rel_props[0]["relationship_type"] == "HAS_FRIEND"


class TestGetInheritedLabels:
    """Tests for get_inherited_labels function."""

    def test_simple_class(self):
        class Gene(fieldz_kb.lpg.graph.BaseNode):
            pass

        labels = momapy_kb.lpg.utils.get_inherited_labels(Gene)
        assert "Gene" in labels

    def test_inherited_class(self):
        class BioEntity(fieldz_kb.lpg.graph.BaseNode):
            pass

        class Gene(BioEntity):
            pass

        labels = momapy_kb.lpg.utils.get_inherited_labels(Gene)
        assert "Gene" in labels
        assert "BioEntity" in labels

    def test_excludes_base_node(self):
        class Gene(fieldz_kb.lpg.graph.BaseNode):
            pass

        labels = momapy_kb.lpg.utils.get_inherited_labels(Gene)
        assert "BaseNode" not in labels


class TestMakeDocFromModule:
    """Tests for make_doc_from_module function."""

    def test_generates_output(self, sample_node_class, tmp_path):
        import types

        module = types.ModuleType("test_module")
        module.Person = sample_node_class

        output_file = tmp_path / "doc.html"
        momapy_kb.lpg.utils.make_doc_from_module(
            module, mode="neo4j", output_file_path=str(output_file)
        )
        assert output_file.exists()
        content = output_file.read_text()
        assert "Person" in content

    def test_invalid_mode_raises_error(self, sample_node_class):
        import types

        module = types.ModuleType("test_module")
        module.Person = sample_node_class

        with pytest.raises(ValueError, match="unrecognized mode"):
            momapy_kb.lpg.utils.make_doc_from_module(module, mode="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
