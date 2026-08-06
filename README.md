# momapy_kb

[![License](https://img.shields.io/github/license/adrienrougny/momapy_kb)](<https://github.com/adrienrougny/momapy_kb/blob/main/COPYING>)
A library to integrate SBGN and CellDesigner maps into graph databases and logic programs.
This library relies on [momapy](https://github.com/adrienrougny/momapy) for handling maps.

## Features

- **Multiple Labelled Property Graph backends**: save maps in a Neo4j, FalkorDB or FalkorDBLite (embedded) database
- **A Clingo/ASP backend**: convert maps to logic programming facts
- **Collection management**: organize maps into named collections
- **Layout element queries**: query model elements and retrieve corresponding layout elements for rendering

## Installation

```bash
pip install momapy-kb[neo4j]        # Neo4j support
pip install momapy-kb[falkordb]     # FalkorDB support
pip install momapy-kb[falkordblite] # FalkorDBLite (embedded) support
pip install momapy-kb[clingo]       # Clingo/ASP support
pip install momapy-kb[all]          # Everything
```

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
    # save a CellDesigner map to the Neo4j database
    session.save_from_file("model.xml", integration_mode="hash")

    # query model elements, return them as momapy objects
    results = session.execute_query_as_objects(
        "MATCH (n:Protein) RETURN n"
    )

    # query model elements, return corresponding layout elements
    layout_results = session.execute_query_as_layout_elements(
        "MATCH (n:Reaction) RETURN n"
    )
```

## Documentation

Full documentation is available at [https://adrienrougny.github.io/momapy_kb/](https://adrienrougny.github.io/momapy_kb/).
