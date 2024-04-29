import neomodel

import momapy
import momapy.sbgn.pd
import momapy.sbgn.io.sbgnml
import momapy.io

import momapy_kg.builder
import momapy_kg.sbgn
import credentials

if __name__ == "__main__":
    neomodel.config.DATABASE_URL = f"bolt://{credentials.USER_NAME}:{credentials.PASSWORD}@{credentials.HOST_NAME}:7687"
    results, meta = neomodel.db.cypher_query("MATCH (n) DETACH DELETE n")
    # momapy_neo4j.utils.pretty_print(
    #     momapy_neo4j.builder.get_or_make_node_cls(
    #         momapy.sbgn.pd.Macromolecule
    #     ),
    #     max_depth=3,
    # )
    m = momapy.io.read("Neuroinflammation.sbgn")
    momapy_kg.builder.save_node_from_object(m)
