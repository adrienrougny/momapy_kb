import fieldz_kb.neo4j.core

import momapy.core.mapping
import momapy.core.elements
import momapy.drawing
import momapy.io.core
import momapy.utils


class NoneValueType(fieldz_kb.neo4j.core.BaseNode):
    pass


class LayoutModelMapping(fieldz_kb.neo4j.core.FrozenDict):
    pass


class FrozenSurjectionDict(fieldz_kb.neo4j.core.FrozenDict):
    pass


fieldz_kb.neo4j.core._type_to_node_class[momapy.drawing.NoneValueType] = NoneValueType
fieldz_kb.neo4j.core._type_to_node_class[momapy.core.mapping.LayoutModelMapping] = (
    LayoutModelMapping
)
fieldz_kb.neo4j.core._type_to_node_class[momapy.utils.FrozenSurjectionDict] = (
    FrozenSurjectionDict
)


def _make_nodes_from_none_value(
    none_value, integration_mode, exclude_from_integration, object_to_node
):
    node = NoneValueType()
    return [node], []


def _make_nodes_from_layout_model_mapping_object(
    dict_object, integration_mode, exclude_from_integration, object_to_node
):
    node = LayoutModelMapping()
    nodes = [node]
    to_connect = []
    for key, value in dict_object.items():
        nodes_item, to_connect_item = fieldz_kb.neo4j.core._make_nodes_from_dict_item(
            key,
            value,
            integration_mode=integration_mode,
            exclude_from_integration=exclude_from_integration,
            object_to_node=object_to_node,
        )
        nodes += nodes_item
        to_connect += to_connect_item
        node_item = nodes_item[0]
        to_connect.append((node, "items", node_item, {}))
    return nodes, to_connect


def _make_nodes_from_frozen_surjection_dict_object(
    dict_object, integration_mode, exclude_from_integration, object_to_node
):
    node = FrozenSurjectionDict()
    nodes = [node]
    to_connect = []
    for key, value in dict_object.items():
        nodes_item, to_connect_item = fieldz_kb.neo4j.core._make_nodes_from_dict_item(
            key,
            value,
            integration_mode=integration_mode,
            exclude_from_integration=exclude_from_integration,
            object_to_node=object_to_node,
        )
        nodes += nodes_item
        to_connect += to_connect_item
        node_item = nodes_item[0]
        to_connect.append((node, "items", node_item, {}))
    return nodes, to_connect


fieldz_kb.neo4j.core.register_make_nodes_function(
    momapy.drawing.NoneValueType, _make_nodes_from_none_value
)


fieldz_kb.neo4j.core.register_make_nodes_function(
    momapy.core.mapping.LayoutModelMapping,
    _make_nodes_from_layout_model_mapping_object,
)

fieldz_kb.neo4j.core.register_make_nodes_function(
    momapy.utils.FrozenSurjectionDict,
    _make_nodes_from_frozen_surjection_dict_object,
)


def _make_none_value_from_node(node, node_element_id_to_object):
    return momapy.drawing.NoneValue


def _make_layout_model_mapping_object_from_node(node, node_element_id_to_object):
    return momapy.core.mapping.LayoutModelMapping(
        fieldz_kb.neo4j.core._make_dict_object_from_node(
            node, node_element_id_to_object=node_element_id_to_object
        )
    )


def _make_frozen_surjection_dict_object_from_node(node, node_element_id_to_object):
    return momapy.utils.FrozenSurjectionDict(
        fieldz_kb.neo4j.core._make_dict_object_from_node(
            node, node_element_id_to_object=node_element_id_to_object
        )
    )


fieldz_kb.neo4j.core.register_make_object_function(
    NoneValueType, _make_none_value_from_node
)
fieldz_kb.neo4j.core.register_make_object_function(
    LayoutModelMapping, _make_layout_model_mapping_object_from_node
)
fieldz_kb.neo4j.core.register_make_object_function(
    FrozenSurjectionDict, _make_frozen_surjection_dict_object_from_node
)
