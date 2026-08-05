# momapy_kb

[![License](https://img.shields.io/github/license/adrienrougny/momapy_kb)](<https://github.com/adrienrougny/momapy_kb/blob/main/COPYING>)

A library to integrate [momapy](https://github.com/adrienrougny/momapy) maps into graph databases and logic programming backends.

## Features

- **Multiple LPG backends** -- Neo4j, FalkorDB, FalkorDBLite (embedded), via [fieldz_kb](https://github.com/adrienrougny/fieldz_kb)
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

## Demo

A Jupyter notebook walking through storing, querying, and rendering maps with Neo4j is available at `demos/neo4j/main.ipynb`. It requires a running Neo4j server:

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
```

To run the demo locally:

```bash
git clone https://github.com/adrienrougny/momapy_kb.git
cd momapy_kb
pip install .[neo4j] momapy[notebook] jupyter
jupyter notebook demos/neo4j/main.ipynb
```

The demo includes example data (`demos/neo4j/data/`) that is only available in the repository.

## Documentation

Full documentation is available at [https://adrienrougny.github.io/momapy_kb/](https://adrienrougny.github.io/momapy_kb/).
