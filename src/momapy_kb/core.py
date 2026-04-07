import dataclasses

import frozendict

import momapy.core.model
import momapy.core.map
import momapy.core.layout


@dataclasses.dataclass(frozen=True)
class CollectionEntry:
    id_: str
    model: momapy.core.map.Map | momapy.core.model.Model | momapy.core.layout.Layout
    rdf_annotations: frozendict.frozendict | None = None
    file_path: str | None = None
    ids: frozendict.frozendict | None = None
    notes: frozendict.frozendict | None = None


@dataclasses.dataclass(frozen=True)
class Collection:
    name: str
    entries: frozenset[CollectionEntry] = dataclasses.field(default_factory=frozenset)
