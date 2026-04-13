"""Core data models for momapy_kb.

Provides CollectionEntry and Collection dataclasses for organizing
maps into named collections.
"""

import dataclasses

import frozendict

import momapy.core.model
import momapy.core.map
import momapy.core.layout
import momapy.core.elements
import momapy.sbml.core


@dataclasses.dataclass(frozen=True)
class CollectionEntry:
    """A single entry in a collection.

    Attributes:
        id_: Unique identifier for this entry.
        model: The momapy map, model, or layout object.
        rdf_annotations: Optional RDF annotations per map element.
        file_path: Optional path to the source file.
        ids: Optional ID mappings per map element.
        notes: Optional notes per map element.
    """

    id_: str
    model: momapy.core.map.Map | momapy.core.model.Model | momapy.core.layout.Layout
    rdf_annotations: (
        frozendict.frozendict[
            momapy.core.elements.MapElement, frozenset[momapy.sbml.core.RDFAnnotation]
        ]
        | None
    ) = None
    file_path: str | None = None
    ids: (
        frozendict.frozendict[momapy.core.elements.MapElement, frozenset[str]] | None
    ) = None
    notes: (
        frozendict.frozendict[momapy.core.elements.MapElement, frozenset[str]] | None
    ) = None


@dataclasses.dataclass(frozen=True)
class Collection:
    """A named collection of map entries.

    Attributes:
        name: The collection name.
        entries: The entries in this collection.
    """

    name: str
    entries: frozenset[CollectionEntry] = dataclasses.field(default_factory=frozenset)
