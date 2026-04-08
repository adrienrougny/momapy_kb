import dataclasses

import frozendict

import momapy.core.model
import momapy.core.map
import momapy.core.layout
import momapy.core.elements
import momapy.sbml.core


@dataclasses.dataclass(frozen=True)
class CollectionEntry:
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
    name: str
    entries: frozenset[CollectionEntry] = dataclasses.field(default_factory=frozenset)
