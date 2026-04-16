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
import momapy.utils


@dataclasses.dataclass(frozen=True)
class CollectionEntry:
    """A single entry in a collection.

    Attributes:
        id_: Unique identifier for this entry.
        model: The momapy map, model, or layout object.
        element_to_annotations: Optional mapping from map elements to their
            annotations.
        file_path: Optional path to the source file.
        source_id_to_model_element: Optional mapping from source file IDs
            to model elements.
        source_id_to_layout_element: Optional mapping from source file IDs
            to layout elements.
        element_to_notes: Optional mapping from map elements to their notes.
    """

    id_: str
    model: momapy.core.map.Map | momapy.core.model.Model | momapy.core.layout.Layout
    element_to_annotations: frozendict.frozendict | None = None
    file_path: str | None = None
    source_id_to_model_element: momapy.utils.FrozenSurjectionDict | None = None
    source_id_to_layout_element: momapy.utils.FrozenSurjectionDict | None = None
    element_to_notes: frozendict.frozendict | None = None


@dataclasses.dataclass(frozen=True)
class Collection:
    """A named collection of map entries.

    Attributes:
        name: The collection name.
        entries: The entries in this collection.
    """

    name: str
    entries: frozenset[CollectionEntry] = dataclasses.field(default_factory=frozenset)
