# Getting started

## LPG backends (Neo4j, FalkorDB, FalkorDBLite)

### Connecting to a database

#### Neo4j

```python
import momapy_kb.lpg.backends.neo4j

backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
    hostname="localhost",
    port=7687,
    username="neo4j",
    password="password",
)
```

#### FalkorDB

```python
import momapy_kb.lpg.backends.falkordb

backend = momapy_kb.lpg.backends.falkordb.FalkorDBBackend(
    hostname="localhost",
    port=6379,
    database="default",
)
```

#### FalkorDBLite (embedded)

No server required:

```python
import momapy_kb.lpg.backends.falkordblite

backend = momapy_kb.lpg.backends.falkordblite.FalkorDBLiteBackend(
    path="/tmp/my_graph.db",
    database="default",
)
```

### Sessions

A session manages the connection between your Python objects and the database:

```python
import momapy_kb.lpg.session

with momapy_kb.lpg.session.Session(backend) as session:
    session.save_from_object(my_object)
```

### Saving objects

#### From Python objects

```python
import dataclasses

@dataclasses.dataclass
class Gene:
    name: str
    chromosome: int

gene = Gene(name="TP53", chromosome=17)

with momapy_kb.lpg.session.Session(backend) as session:
    # Save a single object
    session.save_from_object(gene)

    # Save multiple objects
    genes = [Gene(name="BRCA1", chromosome=17), Gene(name="HK1", chromosome=10)]
    session.save_from_objects(genes)
```

#### From files

Save maps directly from CellDesigner, SBGN, or SBML files:

```python
with momapy_kb.lpg.session.Session(backend) as session:
    # Save a full map (model + layout)
    session.save_from_file("model.xml", integration_mode="hash")

    # Save only the model
    session.save_from_file("model.xml", return_type="model", integration_mode="hash")

    # Save only the layout
    session.save_from_file("model.xml", return_type="layout", integration_mode="hash")
```

### Integration modes

When saving objects, you can control how duplicates are handled:

- `"id"` (default): deduplicates by Python `id()`, so only repeated references to the
  *same* object share a node
- `"hash"`: deduplicates by hash, so objects that are merely *equal* share a node
  (requires hashable objects)

Given two `A` objects held by one parent:

| | `"id"` | `"hash"` |
| --- | --- | --- |
| the same object twice | 1 node | 1 node |
| equal but distinct objects | 2 nodes | 1 node |

```python
with momapy_kb.lpg.session.Session(backend) as session:
    # Hash-based: equal objects share the same node
    session.save_from_object(gene, integration_mode="hash")

    # Id-based (default): same Python object reuses its node
    session.save_from_object(gene, integration_mode="id")
```

#### Which mode to use

Use `"hash"` when integrating **several maps**, so elements shared between them collapse
onto one node and cross-map queries work. It requires every object to be hashable; momapy
map elements are frozen dataclasses, so they qualify, but ordinary mutable dataclasses
raise `ValueError`. Use `"id"` for a **single** object graph. Pass
`exclude_from_integration=(SomeType, ...)` to opt specific types out.

### Querying

```python
with momapy_kb.lpg.session.Session(backend) as session:
    # Raw Cypher query -- returns list[dict]
    results = session.execute_query(
        "MATCH (n:Gene) WHERE n.chromosome = $chr RETURN n.name AS name",
        params={"chr": 17},
    )
    for row in results:
        print(row["name"])

    # Query with automatic conversion to Python objects -- returns list[list[object]]
    results = session.execute_query_as_objects(
        "MATCH (n:Gene) RETURN n ORDER BY n.name"
    )
    for row in results:
        gene = row[0]
        print(gene.name, gene.chromosome)
```

### Layout element queries

When a full map (model + layout) is stored, you can query model elements and retrieve their corresponding layout elements for rendering:

```python
with momapy_kb.lpg.session.Session(backend) as session:
    session.save_from_file("map.sbgn", return_type="map", integration_mode="hash")

    # Query and get layout elements directly
    results = session.cypher_query_as_layout_elements(
        "MATCH (n:Macromolecule) RETURN n"
    )
    for layout_elements in results:
        # layout_elements can be rendered with momapy
        pass
```

### Collections

Organize maps into named collections:

```python
import pathlib
import momapy_kb.core

with momapy_kb.lpg.session.Session(backend) as session:
    # From file paths
    session.save_collections_from_file_paths(
        [
            ("COVID_DM", pathlib.Path("maps/covid/").glob("*.xml")),
            ("PD_DM", pathlib.Path("maps/pd/").glob("*.xml")),
        ],
        return_type="model",
    )

    # From pre-built entries
    entry = momapy_kb.core.CollectionEntry(
        id_="model_1",
        obj=my_model,
    )
    session.save_collections_from_entries(
        [("MyCollection", [entry])]
    )
```

### Clearing the database

```python
with momapy_kb.lpg.session.Session(backend) as session:
    session.delete_all()
```

## Clingo/ASP backend

Convert momapy objects to clingo facts for answer set programming:

```python
import momapy_kb.clingo.core

with momapy_kb.clingo.core.Session() as session:
    # Convert an object to clingo facts
    facts = session.make_facts_from_object(my_model)

    # Generate predicate classes for a type
    predicate_classes = session.get_or_make_predicate_classes_from_type(MyType)

    # Generate ontology rules (type inheritance as ASP rules)
    rules = session.make_ontology_rules_from_type(MyType)
```
