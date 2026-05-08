"""Custom type plugins for momapy-specific types.

Provides node classes and PylpgTypePlugin implementations for types
that are specific to momapy: FrozenDict, NoneValueType,
LayoutModelMapping, FrozenSurjectionDict, and FrozenIdentityMultiDict.
"""

import typing

import frozendict
import pylpg.relationship

import fieldz_kb.lpg.core
import fieldz_kb.lpg.graph
import fieldz_kb.lpg.plugins

import momapy.core.mapping
import momapy.core.elements
import momapy.drawing
import momapy.utils


class HasModelElement(pylpg.relationship.Relationship):
    __type__ = "HAS_MODEL_ELEMENT"


class HasLayoutElement(pylpg.relationship.Relationship):
    __type__ = "HAS_LAYOUT_ELEMENT"


class FrozenDict(fieldz_kb.lpg.graph.Mapping):
    pass


class NoneValueType(fieldz_kb.lpg.graph.BaseNode):
    pass


class LayoutModelMapping(FrozenDict):
    pass


class FrozenSurjectionDict(FrozenDict):
    pass


class FrozenIdentityMultiDict(FrozenDict):
    pass


class FrozenDictPlugin(fieldz_kb.lpg.core.PylpgTypePlugin):

    @classmethod
    def can_handle_type(cls, type_: type) -> bool:
        return type_ is frozendict.frozendict

    @classmethod
    def can_handle_node_class(
        cls, node_class: type, ctx: fieldz_kb.lpg.core.PylpgContext
    ) -> bool:
        return node_class is FrozenDict

    @classmethod
    def make_node_class_from_type(
        cls,
        type_: type,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        make_node_classes_recursively: bool = True,
        guard: set[type] | None = None,
    ) -> type[fieldz_kb.lpg.graph.BaseNode] | None:
        return None

    @classmethod
    def make_nodes_from_object(
        cls,
        obj: object,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        integration_mode: typing.Literal["hash", "id"],
        exclude_from_integration: tuple[type, ...],
        object_to_node: dict,
    ) -> tuple[
        list[fieldz_kb.lpg.graph.BaseNode], list[pylpg.relationship.Relationship]
    ]:
        node = FrozenDict()
        nodes = [node]
        relationships = []
        for key, value in obj.items():
            item_nodes, item_relationships = (
                fieldz_kb.lpg.plugins.DictPlugin._make_nodes_from_dict_item(
                    key,
                    value,
                    ctx,
                    integration_mode=integration_mode,
                    exclude_from_integration=exclude_from_integration,
                    object_to_node=object_to_node,
                )
            )
            nodes += item_nodes
            relationships += item_relationships
            item_node = item_nodes[0]
            relationships.append(
                fieldz_kb.lpg.graph.HasItem(source=node, target=item_node)
            )
        return nodes, relationships

    @classmethod
    def make_object_from_node(
        cls,
        node: fieldz_kb.lpg.graph.BaseNode,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        node_id_to_object: dict,
    ) -> object:
        d = {}
        for item_node in node.items.all():
            key_nodes = item_node.key.all()
            value_nodes = item_node.value.all()
            key = fieldz_kb.lpg.core.make_object_from_node(
                ctx, key_nodes[0], node_id_to_object
            )
            value = fieldz_kb.lpg.core.make_object_from_node(
                ctx, value_nodes[0], node_id_to_object
            )
            d[key] = value
        return frozendict.frozendict(d)


class NoneValueTypePlugin(fieldz_kb.lpg.core.PylpgTypePlugin):

    @classmethod
    def can_handle_type(cls, type_: type) -> bool:
        return type_ is momapy.drawing.NoneValueType

    @classmethod
    def can_handle_node_class(
        cls, node_class: type, ctx: fieldz_kb.lpg.core.PylpgContext
    ) -> bool:
        return node_class is NoneValueType

    @classmethod
    def make_node_class_from_type(
        cls,
        type_: type,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        make_node_classes_recursively: bool = True,
        guard: set[type] | None = None,
    ) -> type[fieldz_kb.lpg.graph.BaseNode] | None:
        return None

    @classmethod
    def make_nodes_from_object(
        cls,
        obj: object,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        integration_mode: typing.Literal["hash", "id"],
        exclude_from_integration: tuple[type, ...],
        object_to_node: dict,
    ) -> tuple[
        list[fieldz_kb.lpg.graph.BaseNode], list[pylpg.relationship.Relationship]
    ]:
        return [NoneValueType()], []

    @classmethod
    def make_object_from_node(
        cls,
        node: fieldz_kb.lpg.graph.BaseNode,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        node_id_to_object: dict,
    ) -> object:
        return momapy.drawing.NoneValue


class LayoutModelMappingPlugin(fieldz_kb.lpg.core.PylpgTypePlugin):

    @classmethod
    def can_handle_type(cls, type_: type) -> bool:
        return type_ is momapy.core.mapping.LayoutModelMapping

    @classmethod
    def can_handle_node_class(
        cls, node_class: type, ctx: fieldz_kb.lpg.core.PylpgContext
    ) -> bool:
        return node_class is LayoutModelMapping

    @classmethod
    def make_node_class_from_type(
        cls,
        type_: type,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        make_node_classes_recursively: bool = True,
        guard: set[type] | None = None,
    ) -> type[fieldz_kb.lpg.graph.BaseNode] | None:
        return None

    @classmethod
    def make_nodes_from_object(
        cls,
        obj: object,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        integration_mode: typing.Literal["hash", "id"],
        exclude_from_integration: tuple[type, ...],
        object_to_node: dict,
    ) -> tuple[
        list[fieldz_kb.lpg.graph.BaseNode], list[pylpg.relationship.Relationship]
    ]:
        node = LayoutModelMapping()
        nodes = [node]
        relationships = []
        for key, value in obj.items():
            item_nodes, item_relationships = (
                fieldz_kb.lpg.plugins.DictPlugin._make_nodes_from_dict_item(
                    key,
                    value,
                    ctx,
                    integration_mode=integration_mode,
                    exclude_from_integration=exclude_from_integration,
                    object_to_node=object_to_node,
                )
            )
            nodes += item_nodes
            relationships += item_relationships
            item_node = item_nodes[0]
            relationships.append(
                fieldz_kb.lpg.graph.HasItem(source=node, target=item_node)
            )
        return nodes, relationships

    @classmethod
    def make_object_from_node(
        cls,
        node: fieldz_kb.lpg.graph.BaseNode,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        node_id_to_object: dict,
    ) -> object:
        d = {}
        for item_node in node.items.all():
            key_nodes = item_node.key.all()
            value_nodes = item_node.value.all()
            key = fieldz_kb.lpg.core.make_object_from_node(
                ctx, key_nodes[0], node_id_to_object
            )
            value = fieldz_kb.lpg.core.make_object_from_node(
                ctx, value_nodes[0], node_id_to_object
            )
            d[key] = value
        return momapy.core.mapping.LayoutModelMapping(d)


class FrozenSurjectionDictPlugin(fieldz_kb.lpg.core.PylpgTypePlugin):

    @classmethod
    def can_handle_type(cls, type_: type) -> bool:
        return type_ is momapy.utils.FrozenSurjectionDict

    @classmethod
    def can_handle_node_class(
        cls, node_class: type, ctx: fieldz_kb.lpg.core.PylpgContext
    ) -> bool:
        return node_class is FrozenSurjectionDict

    @classmethod
    def make_node_class_from_type(
        cls,
        type_: type,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        make_node_classes_recursively: bool = True,
        guard: set[type] | None = None,
    ) -> type[fieldz_kb.lpg.graph.BaseNode] | None:
        return None

    @classmethod
    def make_nodes_from_object(
        cls,
        obj: object,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        integration_mode: typing.Literal["hash", "id"],
        exclude_from_integration: tuple[type, ...],
        object_to_node: dict,
    ) -> tuple[
        list[fieldz_kb.lpg.graph.BaseNode], list[pylpg.relationship.Relationship]
    ]:
        node = FrozenSurjectionDict()
        nodes = [node]
        relationships = []
        for key, value in obj.items():
            item_nodes, item_relationships = (
                fieldz_kb.lpg.plugins.DictPlugin._make_nodes_from_dict_item(
                    key,
                    value,
                    ctx,
                    integration_mode=integration_mode,
                    exclude_from_integration=exclude_from_integration,
                    object_to_node=object_to_node,
                )
            )
            nodes += item_nodes
            relationships += item_relationships
            item_node = item_nodes[0]
            relationships.append(
                fieldz_kb.lpg.graph.HasItem(source=node, target=item_node)
            )
        return nodes, relationships

    @classmethod
    def make_object_from_node(
        cls,
        node: fieldz_kb.lpg.graph.BaseNode,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        node_id_to_object: dict,
    ) -> object:
        d = {}
        for item_node in node.items.all():
            key_nodes = item_node.key.all()
            value_nodes = item_node.value.all()
            key = fieldz_kb.lpg.core.make_object_from_node(
                ctx, key_nodes[0], node_id_to_object
            )
            value = fieldz_kb.lpg.core.make_object_from_node(
                ctx, value_nodes[0], node_id_to_object
            )
            d[key] = value
        return momapy.utils.FrozenSurjectionDict(d)


class FrozenIdentityMultiDictPlugin(fieldz_kb.lpg.core.PylpgTypePlugin):

    @classmethod
    def can_handle_type(cls, type_: type) -> bool:
        return type_ is momapy.utils.FrozenIdentityMultiDict

    @classmethod
    def can_handle_node_class(
        cls, node_class: type, ctx: fieldz_kb.lpg.core.PylpgContext
    ) -> bool:
        return node_class is FrozenIdentityMultiDict

    @classmethod
    def make_node_class_from_type(
        cls,
        type_: type,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        make_node_classes_recursively: bool = True,
        guard: set[type] | None = None,
    ) -> type[fieldz_kb.lpg.graph.BaseNode] | None:
        return None

    @classmethod
    def make_nodes_from_object(
        cls,
        obj: object,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        integration_mode: typing.Literal["hash", "id"],
        exclude_from_integration: tuple[type, ...],
        object_to_node: dict,
    ) -> tuple[
        list[fieldz_kb.lpg.graph.BaseNode], list[pylpg.relationship.Relationship]
    ]:
        node = FrozenIdentityMultiDict()
        nodes = [node]
        relationships = []
        for key, value in obj.items():
            item_nodes, item_relationships = (
                fieldz_kb.lpg.plugins.DictPlugin._make_nodes_from_dict_item(
                    key,
                    value,
                    ctx,
                    integration_mode=integration_mode,
                    exclude_from_integration=exclude_from_integration,
                    object_to_node=object_to_node,
                )
            )
            nodes += item_nodes
            relationships += item_relationships
            item_node = item_nodes[0]
            relationships.append(
                fieldz_kb.lpg.graph.HasItem(source=node, target=item_node)
            )
        return nodes, relationships

    @classmethod
    def make_object_from_node(
        cls,
        node: fieldz_kb.lpg.graph.BaseNode,
        ctx: fieldz_kb.lpg.core.PylpgContext,
        node_id_to_object: dict,
    ) -> object:
        d = {}
        for item_node in node.items.all():
            key_nodes = item_node.key.all()
            value_nodes = item_node.value.all()
            key = fieldz_kb.lpg.core.make_object_from_node(
                ctx, key_nodes[0], node_id_to_object
            )
            value = fieldz_kb.lpg.core.make_object_from_node(
                ctx, value_nodes[0], node_id_to_object
            )
            d[key] = value
        return momapy.utils.FrozenIdentityMultiDict(d)


def register_momapy_plugins(ctx: fieldz_kb.lpg.core.PylpgContext) -> None:
    """Register all momapy-specific type plugins on a context.

    Registers plugins for FrozenDict, NoneValueType, LayoutModelMapping,
    FrozenSurjectionDict, and FrozenIdentityMultiDict, and adds the
    corresponding type-to-node-class and node-class-to-type mappings.

    Args:
        ctx: The pylpg context to register plugins on.
    """
    ctx.register(FrozenDictPlugin)
    ctx.register(NoneValueTypePlugin)
    ctx.register(LayoutModelMappingPlugin)
    ctx.register(FrozenSurjectionDictPlugin)
    ctx.register(FrozenIdentityMultiDictPlugin)

    ctx.type_to_node_class[frozendict.frozendict] = FrozenDict
    ctx.type_to_node_class[momapy.drawing.NoneValueType] = NoneValueType
    ctx.type_to_node_class[momapy.core.mapping.LayoutModelMapping] = (
        LayoutModelMapping
    )
    ctx.type_to_node_class[momapy.utils.FrozenSurjectionDict] = (
        FrozenSurjectionDict
    )
    ctx.type_to_node_class[momapy.utils.FrozenIdentityMultiDict] = (
        FrozenIdentityMultiDict
    )

    ctx.node_class_to_type[FrozenDict] = frozendict.frozendict
    ctx.node_class_to_type[NoneValueType] = momapy.drawing.NoneValueType
    ctx.node_class_to_type[LayoutModelMapping] = (
        momapy.core.mapping.LayoutModelMapping
    )
    ctx.node_class_to_type[FrozenSurjectionDict] = (
        momapy.utils.FrozenSurjectionDict
    )
    ctx.node_class_to_type[FrozenIdentityMultiDict] = (
        momapy.utils.FrozenIdentityMultiDict
    )
