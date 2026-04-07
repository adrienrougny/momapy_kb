import typing

import fieldz_kb.lpg.core
import fieldz_kb.lpg.graph
import fieldz_kb.lpg.session

import momapy.core.mapping
import momapy.core.elements
import momapy.core.layout
import momapy.drawing
import momapy.io.core
import momapy.utils

import momapy_kb.core
import momapy_kb.lpg.types
import momapy_kb.utils


class Session:
    def __init__(self, backend) -> None:
        self._session = fieldz_kb.lpg.session.Session(backend)
        momapy_kb.lpg.types.register_momapy_plugins(self._session._context)

    @property
    def _context(self) -> fieldz_kb.lpg.core.PylpgContext:
        return self._session._context

    def __enter__(self) -> "Session":
        self._session.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        return self._session.__exit__(exc_type, exc_val, exc_tb)

    def save_from_object(
        self,
        object_: object,
        integration_mode: typing.Literal["hash", "id"] = "id",
        exclude_from_integration: tuple[type, ...] | None = None,
    ) -> None:
        self._session.save_from_object(
            object_,
            integration_mode=integration_mode,
            exclude_from_integration=exclude_from_integration,
        )

    def save_from_objects(
        self,
        objects: list[object],
        integration_mode: typing.Literal["hash", "id"] = "id",
        exclude_from_integration: tuple[type, ...] | None = None,
    ) -> None:
        self._session.save_from_objects(
            objects,
            integration_mode=integration_mode,
            exclude_from_integration=exclude_from_integration,
        )

    def execute_query(
        self,
        query: str,
        params: dict | None = None,
    ) -> list[dict]:
        return self._session.execute_query(query, params=params)

    def execute_query_as_objects(
        self,
        query: str,
        params: dict | None = None,
        node_id_to_object: dict | None = None,
    ) -> list[list[object]]:
        return self._session.execute_query_as_objects(
            query, params=params, node_id_to_object=node_id_to_object
        )

    def delete_all(self) -> None:
        self._session.delete_all()

    def save_from_file(
        self,
        file_path: str,
        return_type: typing.Literal["map", "model", "layout"] = "map",
        with_layout: bool = True,
        with_model: bool = True,
        integration_mode: typing.Literal["hash", "id"] | None = None,
    ) -> None:
        object_ = momapy.io.core.read(
            file_path=file_path,
            return_type=return_type,
            with_model=with_model,
            with_layout=with_layout,
        ).obj
        self.save_from_object(object_, integration_mode=integration_mode)

    def get_layout_element_nodes_from_model_element_node(
        self, model_element_node: fieldz_kb.lpg.graph.BaseNode
    ) -> list[fieldz_kb.lpg.graph.BaseNode]:
        query = """
            MATCH (mapping:LayoutModelMapping)-[:HAS_ITEM]->(item:Item)-[:HAS_VALUE]->(model_element:ModelElement),
            (item)-[:HAS_KEY]->(key)
            WHERE elementId(model_element) = $element_id
            RETURN key
        """
        results = self._session._pylpg_session.execute_query(
            query,
            parameters={"element_id": model_element_node._database_id},
            resolve_nodes=True,
        )
        nodes = momapy_kb.utils.flatten_collection(
            [list(row.values()) for row in results]
        )
        layout_element_node_class = self._context.type_to_node_class.get(
            momapy.core.elements.LayoutElement
        )
        layout_element_nodes = []
        for node in nodes:
            if isinstance(node, fieldz_kb.lpg.graph.BaseNode):
                if layout_element_node_class is not None and isinstance(
                    node, layout_element_node_class
                ):
                    layout_element_nodes.append(node)
                else:
                    layout_element_nodes += node.items.all()
        return layout_element_nodes

    def make_layout_elements_from_model_element_node(
        self, model_element_node: fieldz_kb.lpg.graph.BaseNode
    ) -> list[momapy.core.elements.LayoutElement]:
        layout_element_nodes = self.get_layout_element_nodes_from_model_element_node(
            model_element_node
        )
        layout_elements = [
            fieldz_kb.lpg.core.make_object_from_node(self._context, node)
            for node in layout_element_nodes
        ]
        return layout_elements

    def cypher_query_as_layout_elements(
        self, query: str, params: dict | None = None
    ) -> list[list[momapy.core.elements.LayoutElement]]:
        layout_element_results = []
        results = self._session._pylpg_session.execute_query(
            query, parameters=params, resolve_nodes=True
        )
        layout_element_node_class = self._context.type_to_node_class.get(
            momapy.core.elements.LayoutElement
        )
        model_element_node_class = self._context.type_to_node_class.get(
            momapy.core.elements.ModelElement
        )
        for row in results:
            row_values = momapy_kb.utils.flatten_collection(list(row.values()))
            layout_element_nodes = []
            for element in row_values:
                if not isinstance(element, fieldz_kb.lpg.graph.BaseNode):
                    continue
                if layout_element_node_class is not None and isinstance(
                    element, layout_element_node_class
                ):
                    layout_element_nodes.append(element)
                elif model_element_node_class is not None and isinstance(
                    element, model_element_node_class
                ):
                    layout_element_nodes += (
                        self.get_layout_element_nodes_from_model_element_node(element)
                    )
            layout_elements = [
                fieldz_kb.lpg.core.make_object_from_node(self._context, node)
                for node in layout_element_nodes
            ]
            to_add = []
            for layout_element in layout_elements:
                for sub_layout_element in [
                    layout_element
                ] + layout_element.descendants():
                    if isinstance(sub_layout_element, momapy.core.layout.Arc):
                        for attr_name in ["source", "target"]:
                            attr_value = getattr(sub_layout_element, attr_name, None)
                            if isinstance(
                                attr_value,
                                momapy.core.elements.LayoutElement,
                            ):
                                to_add.append(attr_value)
            layout_elements += to_add
            layout_element_results.append(layout_elements)
        return layout_element_results

    def save_collections_from_entries(
        self,
        collection_names_and_entries: list[
            tuple[str, list[momapy_kb.core.CollectionEntry]]
        ],
        delete_all: bool = False,
    ) -> None:
        if delete_all:
            self.delete_all()
        collections = []
        for (
            collection_name,
            collection_entries,
        ) in collection_names_and_entries:
            collection = momapy_kb.core.Collection(
                name=collection_name, entries=frozenset(collection_entries)
            )
            collections.append(collection)
        self.save_from_objects(
            collections,
            integration_mode="hash",
        )

    def save_collections_from_file_paths(
        self,
        collection_names_and_file_paths: list[tuple[str, list]],
        return_type: typing.Literal["map", "model", "layout"] = "map",
        delete_all: bool = False,
    ) -> None:
        if delete_all:
            self.delete_all()
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
                    file_path=str(file_path),
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
        self.save_collections_from_entries(
            collection_names_and_entries,
            delete_all=delete_all,
        )
