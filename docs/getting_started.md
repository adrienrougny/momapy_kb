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
    # save a single object
    session.save_from_object(gene)

    # save multiple objects
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

- `"id"` (default): deduplicates by Python `id()`; repeated references to the same object are modelled by a same node
- `"hash"`: deduplicates by hash; equal objects are modelled by a same node (requires hashable objects)

### Querying

```python
with momapy_kb.lpg.session.Session(backend) as session:
    # raw Cypher query, returns list[dict]
    results = session.execute_query(
        "MATCH (n:Gene) WHERE n.chromosome = $chr RETURN n.name AS name",
        params={"chr": 17},
    )
    for row in results:
        print(row["name"])

    # query and convert to Python objects, returns list[list[object]]
    results = session.execute_query_as_objects(
        "MATCH (n:Gene) RETURN n ORDER BY n.name"
    )
    for row in results:
        gene = row[0]
        print(gene.name, gene.chromosome)
```

### Layout element queries

When a full map (model + layout) is stored, you can query model elements and return their corresponding layout elements:

```python
with momapy_kb.lpg.session.Session(backend) as session:
    session.save_from_file("map.sbgn", return_type="map", integration_mode="hash")

    # query model elements and return layout elements
    results = session.cypher_query_as_layout_elements(
        "MATCH (n:Macromolecule) RETURN n"
    )
    for layout_elements in results:
        # for example, render with momapy
        pass
```

### Collections

Organize maps into named collections:

```python
import pathlib
import momapy_kb.core

with momapy_kb.lpg.session.Session(backend) as session:
    # from file paths, with collection names
    session.save_collections_from_file_paths(
        [
            ("MyCollection1", pathlib.Path("maps/my_collection1/").glob("*.xml")),
            ("MyCollection2", pathlib.Path("maps/my_collection2/").glob("*.xml")),
        ],
        return_type="model",
    )

    # from pre-built entries, with collection names
    entry1 = momapy_kb.core.CollectionEntry(
        id_="model1",
        obj=my_model1,
    )
    entry2 = momapy_kb.core.CollectionEntry(
        id_="model2",
        obj=my_model2,
    )
    session.save_collections_from_entries(
        [("MyCollection", [entry1, entry2])]
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
    # convert an object to clingo facts
    facts = session.make_facts_from_object(obj)

    # generate predicate classes for a type
    predicate_classes = session.get_or_make_predicate_classes_from_type(MyType)

    # generate ontology rules (type inheritance as ASP rules)
    rules = session.make_ontology_rules_from_type(MyType)
```
