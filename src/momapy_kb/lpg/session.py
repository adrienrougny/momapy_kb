"""LPG session for momapy_kb.

Provides a Session class that wraps a fieldz_kb LPG session and adds
momapy-specific functionality: file loading, collection management,
and layout element queries.
"""

import typing

import pylpg.relationship

import fieldz_kb.lpg.core
import fieldz_kb.lpg.graph
import fieldz_kb.lpg.session

import momapy.core.mapping
import momapy.core.elements
import momapy.core.layout
import momapy.core.map
import momapy.core.model
import momapy.drawing
import momapy.io.core
import momapy.utils

import momapy_kb.core
import momapy_kb.lpg.types
import momapy_kb.utils


class Session:
    """Session for saving and retrieving momapy objects via a graph database.

    Wraps a fieldz_kb LPG session and registers momapy-specific type plugins.
    Use as a context manager.

    Args:
        backend: A pylpg backend instance (Neo4jBackend, FalkorDBBackend, etc.)

    Example:
        >>> import momapy_kb.lpg.session
        >>> import momapy_kb.lpg.backends.neo4j
        >>> backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(hostname="localhost")
        >>> with momapy_kb.lpg.session.Session(backend) as session:
        ...     session.save_from_file("model.xml", integration_mode="hash")
    """

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
        with_membership_edges: bool = False,
    ) -> None:
        """Save a single object to the database.

        Args:
            object_: The object to save.
            integration_mode: How to handle duplicate objects ("hash" or "id").
            exclude_from_integration: Types to exclude from integration logic.
            with_membership_edges: If True, emit HAS_MODEL_ELEMENT and
                HAS_LAYOUT_ELEMENT edges from each Model/Layout root to every
                contained element.
        """
        self.save_from_objects(
            [object_],
            integration_mode=integration_mode,
            exclude_from_integration=exclude_from_integration,
            with_membership_edges=with_membership_edges,
        )

    def save_from_objects(
        self,
        objects: list[object],
        integration_mode: typing.Literal["hash", "id"] = "id",
        exclude_from_integration: tuple[type, ...] | None = None,
        with_membership_edges: bool = False,
    ) -> None:
        """Save multiple objects to the database.

        Args:
            objects: The objects to save.
            integration_mode: How to handle duplicate objects ("hash" or "id").
            exclude_from_integration: Types to exclude from integration logic.
            with_membership_edges: If True, emit HAS_MODEL_ELEMENT and
                HAS_LAYOUT_ELEMENT edges from each Model/Layout root to every
                contained element.
        """
        if exclude_from_integration is None:
            exclude_from_integration = tuple()
        object_to_node = {}
        saved_node_ids = set()
        all_nodes = []
        all_relationships = []
        for object_ in objects:
            nodes, relationships = fieldz_kb.lpg.core.make_nodes_from_object(
                self._context,
                object_,
                integration_mode,
                exclude_from_integration,
                object_to_node,
            )
            for node in nodes:
                if id(node) not in saved_node_ids:
                    if not isinstance(node, fieldz_kb.lpg.graph.BaseNode):
                        raise ValueError(
                            f"node type {type(node)} must be a subclass of BaseNode"
                        )
                    all_nodes.append(node)
                    saved_node_ids.add(id(node))
            all_relationships += relationships
        if with_membership_edges:
            all_relationships += self._make_membership_relationships(
                objects, object_to_node, integration_mode
            )
        self._session._pylpg_session.save(all_nodes)
        self._session._pylpg_session.save(all_relationships)

    def _resolve_node(
        self,
        element: object,
        object_to_node: dict,
        integration_mode: typing.Literal["hash", "id"],
    ) -> fieldz_kb.lpg.graph.BaseNode | None:
        key = element if integration_mode == "hash" else id(element)
        return object_to_node.get(key)

    def _iter_model_and_layout_roots(
        self, objects: list[object]
    ) -> typing.Iterator[tuple[str, object]]:
        for object_ in objects:
            if isinstance(object_, momapy.core.map.Map):
                if object_.model is not None:
                    yield ("model", object_.model)
                if object_.layout is not None:
                    yield ("layout", object_.layout)
            elif isinstance(object_, momapy.core.model.Model):
                yield ("model", object_)
            elif isinstance(object_, momapy.core.layout.Layout):
                yield ("layout", object_)
            elif isinstance(object_, momapy_kb.core.Collection):
                for entry in object_.entries:
                    yield from self._iter_model_and_layout_roots([entry.obj])
            elif isinstance(object_, (list, tuple, frozenset, set)):
                yield from self._iter_model_and_layout_roots(list(object_))

    def _make_membership_relationships(
        self,
        objects: list[object],
        object_to_node: dict,
        integration_mode: typing.Literal["hash", "id"],
    ) -> list[pylpg.relationship.Relationship]:
        relationships = []
        for kind, root in self._iter_model_and_layout_roots(objects):
            root_node = self._resolve_node(root, object_to_node, integration_mode)
            if root_node is None:
                continue
            for element in root.descendants():
                element_node = self._resolve_node(
                    element, object_to_node, integration_mode
                )
                if element_node is None:
                    continue
                if kind == "model":
                    relationships.append(
                        momapy_kb.lpg.types.HasModelElement(
                            source=root_node, target=element_node
                        )
                    )
                else:
                    relationships.append(
                        momapy_kb.lpg.types.HasLayoutElement(
                            source=root_node, target=element_node
                        )
                    )
        return relationships

    def execute_query(
        self,
        query: str,
        params: dict | None = None,
        resolve_nodes: bool = False,
    ) -> list[dict]:
        """Execute a Cypher query against the database.

        Args:
            query: The Cypher query string.
            params: Optional query parameters.
            resolve_nodes: When True, any driver node objects in the results
                are hydrated into `pylpg.node.Node` instances.

        Returns:
            Query results as a list of dicts.
        """
        return self._session.execute_query(
            query, params=params, resolve_nodes=resolve_nodes
        )

    def execute_query_as_objects(
        self,
        query: str,
        params: dict | None = None,
        node_id_to_object: dict | None = None,
    ) -> list[list[object]]:
        """Execute a Cypher query and convert results to Python objects.

        Args:
            query: The Cypher query string.
            params: Optional query parameters.
            node_id_to_object: Optional cache mapping node database IDs to objects.

        Returns:
            A list of rows, where each row is a list of Python objects.
        """
        return self._session.execute_query_as_objects(
            query, params=params, node_id_to_object=node_id_to_object
        )

    def delete_all(self) -> None:
        """Delete all nodes and relationships from the database."""
        self._session.delete_all()

    def save_from_file(
        self,
        file_path: str,
        return_type: typing.Literal["map", "model", "layout"] = "map",
        with_layout: bool = True,
        with_model: bool = True,
        integration_mode: typing.Literal["hash", "id"] | None = None,
        with_membership_edges: bool = False,
    ) -> None:
        """Load a map from a file and save it to the database.

        Supports CellDesigner, SBGN, and SBML file formats.

        Args:
            file_path: Path to the input file.
            return_type: What to extract from the file ("map", "model", or "layout").
            with_layout: Whether to include layout data.
            with_model: Whether to include model data.
            integration_mode: How to handle duplicate objects ("hash" or "id").
            with_membership_edges: If True, emit HAS_MODEL_ELEMENT and
                HAS_LAYOUT_ELEMENT edges from each Model/Layout root to every
                contained element.
        """
        object_ = momapy.io.core.read(
            file_path=file_path,
            return_type=return_type,
            with_model=with_model,
            with_layout=with_layout,
        ).obj
        self.save_from_object(
            object_,
            integration_mode=integration_mode,
            with_membership_edges=with_membership_edges,
        )

    def get_layout_element_nodes_from_model_element_node(
        self, model_element_node: fieldz_kb.lpg.graph.BaseNode
    ) -> list[fieldz_kb.lpg.graph.BaseNode]:
        """Get layout element nodes linked to a model element node.

        Queries the LayoutModelMapping in the database to find layout
        elements corresponding to the given model element.

        Args:
            model_element_node: A saved model element node.

        Returns:
            A list of layout element nodes.
        """
        database_id_func_name = (
            self._session._pylpg_session._backend._database_id_func_name
        )
        query = f"""
            MATCH (mapping:LayoutModelMapping)-[:HAS_ITEM]->(item:Item)-[:HAS_VALUE]->(model_element:ModelElement),
            (item)-[:HAS_KEY]->(key)
            WHERE {database_id_func_name}(model_element) = $element_id
            RETURN key
        """
        results = self.execute_query(
            query,
            params={"element_id": model_element_node._database_id},
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
        """Get layout elements as Python objects for a model element node.

        Args:
            model_element_node: A saved model element node.

        Returns:
            A list of momapy LayoutElement objects.
        """
        layout_element_nodes = self.get_layout_element_nodes_from_model_element_node(
            model_element_node
        )
        layout_elements = [
            fieldz_kb.lpg.core.make_object_from_node(self._context, node)
            for node in layout_element_nodes
        ]
        return layout_elements

    def execute_query_as_layout_elements(
        self, query: str, params: dict | None = None
    ) -> list[list[momapy.core.elements.LayoutElement]]:
        """Execute a Cypher query and return results as layout elements.

        For each row in the query results, model element nodes are resolved
        to their corresponding layout elements via the LayoutModelMapping.

        Args:
            query: The Cypher query string.
            params: Optional query parameters.

        Returns:
            A list of rows, where each row is a list of LayoutElement objects.
        """
        layout_element_results = []
        results = self.execute_query(query, params=params, resolve_nodes=True)
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
        integration_mode: typing.Literal["id", "hash"] = "id",
        delete_all: bool = False,
        with_membership_edges: bool = False,
    ) -> None:
        """Save collections from pre-built CollectionEntry objects.

        Args:
            collection_names_and_entries: List of (name, entries) tuples.
            delete_all: If True, clear the database before saving.
            with_membership_edges: If True, emit HAS_MODEL_ELEMENT and
                HAS_LAYOUT_ELEMENT edges from each Model/Layout root to every
                contained element.
        """
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
            integration_mode=integration_mode,
            with_membership_edges=with_membership_edges,
        )

    def save_collections_from_file_paths(
        self,
        collection_names_and_file_paths: list[tuple[str, list]],
        return_type: typing.Literal["map", "model", "layout"] = "map",
        integration_mode: typing.Literal["id", "hash"] = "id",
        delete_all: bool = False,
        with_membership_edges: bool = False,
    ) -> None:
        """Load maps from files and save them as named collections.

        Args:
            collection_names_and_file_paths: List of (name, file_paths) tuples.
            return_type: What to extract from files ("map", "model", or "layout").
            delete_all: If True, clear the database before saving.
            with_membership_edges: If True, emit HAS_MODEL_ELEMENT and
                HAS_LAYOUT_ELEMENT edges from each Model/Layout root to every
                contained element.
        """
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
                entry_id = file_path.stem
                collection_entry = momapy_kb.core.CollectionEntry(
                    id_=entry_id,
                    obj=reader_result.obj,
                    file_path=str(file_path),
                    element_to_annotations=reader_result.element_to_annotations,
                    id_to_element=reader_result.id_to_element,
                    source_id_to_model_element=reader_result.source_id_to_model_element,
                    source_id_to_layout_element=reader_result.source_id_to_layout_element,
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
            integration_mode=integration_mode,
            with_membership_edges=with_membership_edges,
        )
