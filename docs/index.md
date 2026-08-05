# momapy_kb

A library to integrate [momapy](https://github.com/adrienrougny/momapy) maps into graph databases and logic programming backends.

Built on top of [fieldz_kb](https://github.com/adrienrougny/fieldz_kb), momapy_kb provides:

- **Multiple LPG backends** -- Neo4j, FalkorDB, FalkorDBLite (embedded)
- **Clingo/ASP backend** -- convert maps to logic programming facts
- **File-based loading** -- save maps directly from CellDesigner, SBGN, and SBML files
- **Collection management** -- organize maps into named collections
- **Custom type plugins** -- handles momapy-specific types (LayoutModelMapping, NoneValue, FrozenSurjectionDict)
- **Layout element queries** -- query model elements and retrieve corresponding layout elements for rendering

## Installation

```bash
pip install momapy-kb[neo4j]        # Neo4j support
pip install momapy-kb[falkordb]     # FalkorDB support
pip install momapy-kb[falkordblite] # FalkorDBLite (embedded) support
pip install momapy-kb[clingo]       # Clingo/ASP support
pip install momapy-kb[all]          # Everything
```

## Demo

A notebook showcasing how to store, query and render maps with Neo4j is available here: [Demo](neo4j/main.ipynb).

## Quick example

```python
import momapy_kb.lpg.session
import momapy_kb.lpg.backends.neo4j

backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
    hostname="localhost",
    username="neo4j",
    password="password",
)

with momapy_kb.lpg.session.Session(backend) as session:
    # Save a map from a CellDesigner file
    session.save_from_file("model.xml", integration_mode="hash")

    # Query model elements
    results = session.execute_query_as_objects(
        "MATCH (n:Macromolecule) RETURN n"
    )

    # Get layout elements for rendering
    layout_results = session.cypher_query_as_layout_elements(
        "MATCH (n:GenericProcess) RETURN n"
    )
```
