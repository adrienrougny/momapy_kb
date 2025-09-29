import neo4j
import momapy_kb.neo4j.core
import fieldz_kb.neo4j.core
import momapy.celldesigner.core

import momapy_kb.neo4j.celldesigner
import credentials


if __name__ == "__main__":
    INPUT_FILE_PATH = "celldesigner_example.xml"
    # INPUT_FILE_PATH = "test.xml"
    # INPUT_FILE_PATH = "/home/rougny/collections/maps/celldesigner/pd_dm/Core_PD_map.xml"
    # MODE = "layout"
    # MODE = "model"
    MODE = "map"
    if MODE == "layout":
        LABEL = "CellDesignerLayout"
    elif MODE == "model":
        LABEL = "CellDesignerModel"
    elif MODE == "map":
        LABEL = "CellDesignerMap"
    momapy_kb.neo4j.core.connect(
        credentials.HOST_NAME,
        credentials.USER_NAME,
        credentials.PASSWORD,
        notifications_min_severity=neo4j.NotificationMinimumSeverity.OFF,
    )
    momapy_kb.neo4j.core.delete_all()
    print("SAVING OBJECT")
    momapy_kb.neo4j.celldesigner.save_from_file(
        INPUT_FILE_PATH, integration_mode="hash"
    )

    query = f"MATCH (n:{LABEL}) RETURN n"
    print("QUERYING NODE")
    results, meta = momapy_kb.neo4j.core.cypher_query(query, resolve_objects=True)
    for row in results:
        node = row[0]
        print("MAKING OBJECT FROM NODE")
        new_object = momapy_kb.neo4j.core.make_object_from_node(node)
        break
