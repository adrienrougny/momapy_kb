import dataclasses

import neomodel

import momapy
import momapy.sbgn.pd
import momapy.sbgn.io.sbgnml
import momapy.io

import momapy_neo4j.builder
import momapy_neo4j.sbgn
import momapy_neo4j.utils

if __name__ == "__main__":
    neomodel.config.DATABASE_URL = "bolt://neo4j:neofourj@10.240.6.183:7687"
    results, meta = neomodel.db.cypher_query("MATCH (n) DETACH DELETE n")
    # momapy_neo4j.utils.pretty_print(
    #     momapy_neo4j.builder.get_or_make_node_cls(
    #         momapy.sbgn.pd.Macromolecule
    #     ),
    #     max_depth=3,
    # )
    m = momapy.io.read("Neuroinflammation.sbgn")
    momapy_neo4j.builder.save_node_from_object(m)
