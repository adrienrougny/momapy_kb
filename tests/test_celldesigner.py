import momapy.celldesigner.core

import momapy_kb.lpg.core
import momapy_kb.lpg.celldesigner
import momapy_kb.lpg.backends.neo4j
import credentials


if __name__ == "__main__":
    INPUT_FILE_PATH = "celldesigner_example.xml"
    MODE = "map"
    if MODE == "layout":
        LABEL = "CellDesignerLayout"
    elif MODE == "model":
        LABEL = "CellDesignerModel"
    elif MODE == "map":
        LABEL = "CellDesignerMap"
    backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
        hostname=credentials.HOST_NAME,
        username=credentials.USER_NAME,
        password=credentials.PASSWORD,
    )
    with momapy_kb.lpg.core.Session(backend) as session:
        session.delete_all()
        print("SAVING OBJECT")
        session.save_from_file(INPUT_FILE_PATH, integration_mode="hash")

        query = f"MATCH (n:{LABEL}) RETURN n"
        print("QUERYING NODE")
        results = session.execute_query_as_objects(query)
        for row in results:
            node = row[0]
            print("GOT OBJECT FROM NODE")
            print(type(node))
            break
