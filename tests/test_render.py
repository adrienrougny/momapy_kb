import momapy.rendering.skia
import momapy.rendering.core

import momapy_kb.lpg.core
import momapy_kb.lpg.celldesigner
import momapy_kb.lpg.backends.neo4j
import credentials


if __name__ == "__main__":
    OUTPUT_FILE_PATH = "test.pdf"
    INPUT_FILE_PATH = "/home/rougny/collections/maps/celldesigner/pd_dm/Glycolysis.xml"
    print("CONNECTING")
    backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
        hostname=credentials.HOST_NAME,
        username=credentials.USER_NAME,
        password=credentials.PASSWORD,
    )
    with momapy_kb.lpg.core.Session(backend) as session:
        print("DELETING ALL DATA")
        session.delete_all()
        print("SAVING OBJECT TO DB")
        session.save_from_file(INPUT_FILE_PATH, integration_mode="hash")
        print("QUERYING")
        query = "MATCH (n:GenericProtein) RETURN n"
        results = session.execute_query_as_objects(query)
        model_element_nodes = [row[0] for row in results]
        layout_elements = []
        for model_element_node in model_element_nodes:
            layout_elements += (
                session.make_layout_elements_from_model_element_node(
                    model_element_node
                )
            )
        print("RENDERING")
        momapy.rendering.core.render_layout_elements(
            layout_elements, OUTPUT_FILE_PATH, to_top_left=True, multi_pages=False
        )
