"""Tests for momapy_kb LPG type plugins."""

import pytest
import frozendict

import fieldz_kb.lpg.core
import fieldz_kb.lpg.graph
import fieldz_kb.lpg.plugins

import momapy.drawing
import momapy.core.mapping
import momapy.utils

import momapy_kb.lpg.types


@pytest.fixture
def ctx():
    """Create a fresh context with momapy plugins registered."""
    context = fieldz_kb.lpg.core.make_context()
    momapy_kb.lpg.types.register_momapy_plugins(context)
    return context


class TestRegisterMomapyPlugins:
    """Tests for the register_momapy_plugins function."""

    def test_registers_type_to_node_class_mappings(self, ctx):
        assert (
            ctx.type_to_node_class[frozendict.frozendict]
            is momapy_kb.lpg.types.FrozenDict
        )
        assert (
            ctx.type_to_node_class[momapy.drawing.NoneValueType]
            is momapy_kb.lpg.types.NoneValueType
        )
        assert (
            ctx.type_to_node_class[momapy.core.mapping.LayoutModelMapping]
            is momapy_kb.lpg.types.LayoutModelMapping
        )
        assert (
            ctx.type_to_node_class[momapy.utils.FrozenSurjectionDict]
            is momapy_kb.lpg.types.FrozenSurjectionDict
        )

    def test_registers_node_class_to_type_mappings(self, ctx):
        assert (
            ctx.node_class_to_type[momapy_kb.lpg.types.FrozenDict]
            is frozendict.frozendict
        )
        assert (
            ctx.node_class_to_type[momapy_kb.lpg.types.NoneValueType]
            is momapy.drawing.NoneValueType
        )
        assert (
            ctx.node_class_to_type[momapy_kb.lpg.types.LayoutModelMapping]
            is momapy.core.mapping.LayoutModelMapping
        )
        assert (
            ctx.node_class_to_type[momapy_kb.lpg.types.FrozenSurjectionDict]
            is momapy.utils.FrozenSurjectionDict
        )

    def test_plugins_are_registered(self, ctx):
        plugin = ctx.get_plugin_for_type(frozendict.frozendict)
        # May be FrozenDictPlugin or fieldz_kb's DictPlugin depending on
        # whether fieldz_kb still handles frozendict
        assert plugin in (
            momapy_kb.lpg.types.FrozenDictPlugin,
            fieldz_kb.lpg.plugins.DictPlugin,
        )

        plugin = ctx.get_plugin_for_type(momapy.drawing.NoneValueType)
        assert plugin is momapy_kb.lpg.types.NoneValueTypePlugin

        plugin = ctx.get_plugin_for_type(momapy.core.mapping.LayoutModelMapping)
        assert plugin is momapy_kb.lpg.types.LayoutModelMappingPlugin

        plugin = ctx.get_plugin_for_type(momapy.utils.FrozenSurjectionDict)
        assert plugin is momapy_kb.lpg.types.FrozenSurjectionDictPlugin


class TestFrozenDictPlugin:
    """Tests for the FrozenDict plugin."""

    def test_can_handle_type(self):
        assert momapy_kb.lpg.types.FrozenDictPlugin.can_handle_type(
            frozendict.frozendict
        )

    def test_cannot_handle_other_type(self):
        assert not momapy_kb.lpg.types.FrozenDictPlugin.can_handle_type(dict)

    def test_can_handle_node_class(self, ctx):
        assert momapy_kb.lpg.types.FrozenDictPlugin.can_handle_node_class(
            momapy_kb.lpg.types.FrozenDict, ctx
        )

    def test_make_nodes_from_object(self, ctx):
        d = frozendict.frozendict({"a": 1, "b": 2})
        nodes, relationships = (
            momapy_kb.lpg.types.FrozenDictPlugin.make_nodes_from_object(
                d, ctx, "id", (), {}
            )
        )
        assert len(nodes) > 0
        assert isinstance(nodes[0], momapy_kb.lpg.types.FrozenDict)
        assert len(relationships) > 0


class TestNoneValueTypePlugin:
    """Tests for the NoneValueType plugin."""

    def test_can_handle_type(self):
        assert momapy_kb.lpg.types.NoneValueTypePlugin.can_handle_type(
            momapy.drawing.NoneValueType
        )

    def test_cannot_handle_other_type(self):
        assert not momapy_kb.lpg.types.NoneValueTypePlugin.can_handle_type(int)

    def test_can_handle_node_class(self, ctx):
        assert momapy_kb.lpg.types.NoneValueTypePlugin.can_handle_node_class(
            momapy_kb.lpg.types.NoneValueType, ctx
        )

    def test_cannot_handle_other_node_class(self, ctx):
        assert not momapy_kb.lpg.types.NoneValueTypePlugin.can_handle_node_class(
            fieldz_kb.lpg.graph.BaseNode, ctx
        )

    def test_make_node_class_from_type_returns_none(self, ctx):
        result = momapy_kb.lpg.types.NoneValueTypePlugin.make_node_class_from_type(
            momapy.drawing.NoneValueType, ctx
        )
        assert result is None

    def test_make_nodes_from_object(self, ctx):
        nodes, relationships = (
            momapy_kb.lpg.types.NoneValueTypePlugin.make_nodes_from_object(
                momapy.drawing.NoneValue, ctx, "id", (), {}
            )
        )
        assert len(nodes) == 1
        assert isinstance(nodes[0], momapy_kb.lpg.types.NoneValueType)
        assert len(relationships) == 0

    def test_make_object_from_node(self, ctx):
        node = momapy_kb.lpg.types.NoneValueType()
        result = momapy_kb.lpg.types.NoneValueTypePlugin.make_object_from_node(
            node, ctx, {}
        )
        assert result is momapy.drawing.NoneValue


class TestLayoutModelMappingPlugin:
    """Tests for the LayoutModelMapping plugin."""

    def test_can_handle_type(self):
        assert momapy_kb.lpg.types.LayoutModelMappingPlugin.can_handle_type(
            momapy.core.mapping.LayoutModelMapping
        )

    def test_cannot_handle_other_type(self):
        assert not momapy_kb.lpg.types.LayoutModelMappingPlugin.can_handle_type(dict)

    def test_can_handle_node_class(self, ctx):
        assert momapy_kb.lpg.types.LayoutModelMappingPlugin.can_handle_node_class(
            momapy_kb.lpg.types.LayoutModelMapping, ctx
        )

    def test_make_nodes_from_object(self, ctx):
        mapping = momapy.core.mapping.LayoutModelMapping({"a": "b"})
        nodes, relationships = (
            momapy_kb.lpg.types.LayoutModelMappingPlugin.make_nodes_from_object(
                mapping, ctx, "id", (), {}
            )
        )
        assert len(nodes) > 0
        assert isinstance(nodes[0], momapy_kb.lpg.types.LayoutModelMapping)
        assert len(relationships) > 0


class TestFrozenSurjectionDictPlugin:
    """Tests for the FrozenSurjectionDict plugin."""

    def test_can_handle_type(self):
        assert momapy_kb.lpg.types.FrozenSurjectionDictPlugin.can_handle_type(
            momapy.utils.FrozenSurjectionDict
        )

    def test_cannot_handle_other_type(self):
        assert not momapy_kb.lpg.types.FrozenSurjectionDictPlugin.can_handle_type(
            dict
        )

    def test_can_handle_node_class(self, ctx):
        assert momapy_kb.lpg.types.FrozenSurjectionDictPlugin.can_handle_node_class(
            momapy_kb.lpg.types.FrozenSurjectionDict, ctx
        )

    def test_make_nodes_from_object(self, ctx):
        d = momapy.utils.FrozenSurjectionDict({"x": "y"})
        nodes, relationships = (
            momapy_kb.lpg.types.FrozenSurjectionDictPlugin.make_nodes_from_object(
                d, ctx, "id", (), {}
            )
        )
        assert len(nodes) > 0
        assert isinstance(nodes[0], momapy_kb.lpg.types.FrozenSurjectionDict)
        assert len(relationships) > 0


class TestNodeClassInheritance:
    """Tests for node class hierarchy."""

    def test_frozen_dict_is_mapping(self):
        assert issubclass(
            momapy_kb.lpg.types.FrozenDict, fieldz_kb.lpg.graph.Mapping
        )

    def test_none_value_type_is_base_node(self):
        assert issubclass(
            momapy_kb.lpg.types.NoneValueType, fieldz_kb.lpg.graph.BaseNode
        )

    def test_layout_model_mapping_is_frozen_dict(self):
        assert issubclass(
            momapy_kb.lpg.types.LayoutModelMapping, momapy_kb.lpg.types.FrozenDict
        )

    def test_frozen_surjection_dict_is_frozen_dict(self):
        assert issubclass(
            momapy_kb.lpg.types.FrozenSurjectionDict, momapy_kb.lpg.types.FrozenDict
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
