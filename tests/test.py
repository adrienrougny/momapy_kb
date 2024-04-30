import neomodel

import momapy.core
import momapy.sbgn.pd
import momapy.sbgn.io.sbgnml
import momapy.io

import momapy_kg.neo4j
import momapy_kg.utils
import credentials

if __name__ == "__main__":
    # neomodel.config.DATABASE_URL = f"bolt://{credentials.USER_NAME}:{credentials.PASSWORD}@{credentials.HOST_NAME}:7687"
    # results, meta = neomodel.db.cypher_query("MATCH (n) DETACH DELETE n")
    # m = momapy.io.read("Neuroinflammation.sbgn")
    # # momapy_kg.neo4j.save_node_from_object(m)
    model_node_cls = momapy_kg.neo4j.get_or_make_node_cls(
        momapy.sbgn.pd.SBGNPDModel
    )
    momapy_kg.neo4j.make_doc(
        momapy.sbgn.pd,
        mode="neomodel",
        output_file_path="test.html",
        exclude=[momapy.core.LayoutElement, momapy.builder.Builder],
    )
