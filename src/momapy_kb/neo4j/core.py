import typing

import fieldz_kb.neo4j.core

import momapy.core.mapping
import momapy.core.elements
import momapy.drawing
import momapy.io.core
import momapy.utils

import momapy_kb.core
import momapy_kb.utils
import momapy_kb.neo4j.types


connect = fieldz_kb.neo4j.core.connect
delete_all = fieldz_kb.neo4j.core.delete_all
cypher_query = fieldz_kb.neo4j.core.cypher_query
save_from_object = fieldz_kb.neo4j.core.save_from_object
save_from_objects = fieldz_kb.neo4j.core.save_from_objects


def cypher_query_as_objects(query, params=None, node_element_id_to_object=None):
    return fieldz_kb.neo4j.core.cypher_query_as_objects(
        query=query, params=params, node_element_id_to_object=node_element_id_to_object
    )


def save_from_file(
    file_path,
    return_type: typing.Literal["map", "model", "layout"] = "map",
    with_layout=True,
    with_model=True,
    integration_mode: typing.Literal["hash", "id"] | None = None,
):
    object_ = momapy.io.core.read(
        file_path=file_path,
        return_type=return_type,
        with_model=with_model,
        with_layout=with_layout,
    ).obj
    save_from_object(object_, integration_mode=integration_mode)


def get_layout_element_nodes_from_model_element_node(model_element_node):
    query = """
        MATCH (mapping:LayoutModelMapping)-[:HAS_ITEM]->(item:Item)-[:HAS_VALUE]->(model_element:ModelElement),
        (item)-[:HAS_KEY]->(key)
        WHERE elementId(model_element) = $element_id
        RETURN key
    """
    results, _ = cypher_query(
        query,
        params={"element_id": model_element_node.element_id},
        resolve_objects=True,
    )
    nodes = momapy_kb.utils.flatten_collection(results)
    layout_element_nodes = []
    for node in nodes:
        if isinstance(
            node,
            fieldz_kb.neo4j.core._type_to_node_class[
                momapy.core.elements.LayoutElement
            ],
        ):
            layout_element_nodes.append(node)
        else:
            layout_element_nodes += node.items.all()
    return layout_element_nodes


def make_layout_elements_from_model_element_node(model_element_node):
    layout_element_nodes = get_layout_element_nodes_from_model_element_node(
        model_element_node
    )
    layout_elements = [
        fieldz_kb.neo4j.core.make_object_from_node(_) for _ in layout_element_nodes
    ]
    return layout_elements


def cypher_query_as_layout_elements(query, params=None):
    layout_element_results = []
    results, meta = cypher_query(query, params=params, resolve_objects=True)
    for row in results:
        row = momapy_kb.utils.flatten_collection(row)
        layout_element_nodes = []
        for element in row:
            if isinstance(
                element,
                fieldz_kb.neo4j.core._type_to_node_class[
                    momapy.core.elements.LayoutElement
                ],
            ):
                layout_element_nodes.append(element)
            elif isinstance(
                element,
                fieldz_kb.neo4j.core._type_to_node_class[
                    momapy.core.elements.ModelElement
                ],
            ):
                layout_element_nodes += (
                    get_layout_element_nodes_from_model_element_node(element)
                )
        layout_elements = [make_object_from_node(_) for _ in layout_element_nodes]
        to_add = []
        for layout_element in layout_elements:
            for sub_layout_element in [layout_element] + layout_element.descendants():
                if isinstance(sub_layout_element, momapy.core.layout.Arc):
                    for attr_name in ["source", "target"]:
                        attr_value = getattr(sub_layout_element, attr_name, None)
                        if isinstance(attr_value, momapy.core.elements.LayoutElement):
                            to_add.append(attr_value)
        layout_elements += to_add
        layout_element_results.append(layout_elements)
    return layout_element_results, meta


def save_collections_from_entries(
    collection_names_and_entries,
    delete_all=False,
):
    if delete_all:
        delete_all()
    collections = []
    for collection_name, collection_entries in collection_names_and_entries:
        collection = momapy_kb.core.Collection(
            name=collection_name, entries=frozenset(collection_entries)
        )
        collections.append(collection)
    save_from_objects(
        collections,
        integration_mode="hash",
    )


def save_collections_from_file_paths(
    collection_names_and_file_paths,
    return_type: typing.Literal["map", "model", "layout"] = "map",
    delete_all=False,
):
    if delete_all:
        delete_all()
    collection_names_and_entries = []
    for (
        collection_name,
        collection_file_paths,
    ) in collection_names_and_file_paths:
        collection_entries = []
        for file_path in collection_file_paths:
            reader_result = momapy.io.core.read(file_path, return_type=return_type)
            model_id = file_path.stem
            model = reader_result.obj
            annotations = reader_result.annotations
            ids = reader_result.ids
            collection_entry = momapy_kb.core.CollectionEntry(
                id_=model_id,
                model=model,
                file_path=file_path,
                rdf_annotations=annotations,
                ids=ids,
            )
            collection_entries.append(collection_entry)
        collection_names_and_entries.append(
            (
                collection_name,
                collection_entries,
            )
        )
    save_collections_from_entries(
        collection_names_and_entries,
        delete_all=delete_all,
    )
