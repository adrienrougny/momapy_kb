import typing

import fieldz_kb.neo4j.core
import momapy.core
import momapy.drawing


def connect(
    hostname,
    username,
    password,
    protocol="neo4j",
    port="7687",
    notifications_min_severity=None,
):
    fieldz_kb.neo4j.core.connect(
        hostname=hostname,
        username=username,
        password=password,
        protocol=protocol,
        port=port,
        notifications_min_severity=notifications_min_severity,
    )


def delete_all():
    fieldz_kb.neo4j.core.delete_all()


def cypher_query(query, params=None, resolve_objects=True):
    return fieldz_kb.neo4j.core.cypher_query(query, params, resolve_objects)


class NoneValueType(fieldz_kb.neo4j.core.BaseNode):
    pass


fieldz_kb.neo4j.core._type_to_node_class[momapy.drawing.NoneValueType] = NoneValueType


def _make_nodes_from_none_value(
    none_value, integration_mode, exclude_from_integration, object_to_node
):
    node = NoneValueType()
    return [node], []


fieldz_kb.neo4j.core.register_make_nodes_function(
    momapy.drawing.NoneValueType, _make_nodes_from_none_value
)


class LayoutModelMapping(fieldz_kb.neo4j.core.FrozenDict):
    pass


fieldz_kb.neo4j.core._type_to_node_class[momapy.core.LayoutModelMapping] = (
    LayoutModelMapping
)


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


fieldz_kb.neo4j.core.register_make_nodes_function(
    momapy.core.LayoutModelMapping,
    _make_nodes_from_layout_model_mapping_object,
)


def save_from_object(
    object_,
    integration_mode: typing.Literal["hash", "id"] | None = None,
    exclude_from_integration=None,
):
    node = fieldz_kb.neo4j.core.save_from_object(
        object_,
        integration_mode=integration_mode,
        exclude_from_integration=exclude_from_integration,
    )
    return node


def _make_none_value_from_node(node, node_element_id_to_object):
    return momapy.drawing.NoneValue


def _make_layout_model_mapping_object_from_node(node, node_element_id_to_object):
    return momapy.core.LayoutModelMapping(
        fieldz_kb.neo4j.core._make_dict_object_from_node(node)
    )


fieldz_kb.neo4j.core.register_make_object_function(
    NoneValueType, _make_none_value_from_node
)
fieldz_kb.neo4j.core.register_make_object_function(
    LayoutModelMapping, _make_layout_model_mapping_object_from_node
)


def make_object_from_node(node, node_element_id_to_object=None):
    return fieldz_kb.neo4j.core.make_object_from_node(
        node, node_element_id_to_object=node_element_id_to_object
    )


def get_layout_elements_from_mode_element_node(model_element_node):
    query = """
        MATCH (model_element:ModelElement)<-[:HAS_VALUE]-(item:Item)-[:HAS_KEY]->(layout_element:LayoutElement)
        WHERE elementId(model_element) = $element_id
        RETURN layout_element
    """
    results, _ = cypher_query(
        query,
        params={"element_id": model_element_node.element_id},
        resolve_objects=True,
    )
    layout_elements = [make_object_from_node(_[0]) for _ in results]
    return layout_elements
