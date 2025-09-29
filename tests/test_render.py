import neo4j

import momapy.rendering.skia
import momapy.rendering.core

import momapy_kb.neo4j.core
import momapy_kb.neo4j.celldesigner
import credentials


if __name__ == "__main__":
    OUTPUT_FILE_PATH = "test.pdf"
    # INPUT_FILE_PATH = "test.xml"
    INPUT_FILE_PATH = "/home/rougny/collections/maps/celldesigner/pd_dm/Glycolysis.xml"
    print("CONNECTING")
    momapy_kb.neo4j.core.connect(
        credentials.HOST_NAME,
        credentials.USER_NAME,
        credentials.PASSWORD,
        notifications_min_severity=neo4j.NotificationMinimumSeverity.OFF,
    )
    print("DELETING ALL DATA")
    momapy_kb.neo4j.core.delete_all()
    print("SAVING OBJECT TO DB")
    momapy_kb.neo4j.celldesigner.save_from_file(
        INPUT_FILE_PATH, integration_mode="hash"
    )
    print("QUERYING")
    # query = "MATCH (n:StateTransition) RETURN n"
    query = "MATCH (n:GenericProtein) RETURN n"
    results, meta = momapy_kb.neo4j.core.cypher_query(query)
    model_element_nodes = [_[0] for _ in results]
    layout_elements = []
    for model_element_node in model_element_nodes:
        layout_elements += (
            momapy_kb.neo4j.core.get_layout_elements_from_mode_element_node(
                model_element_node
            )
        )
    print("RENDERING")
    momapy.rendering.core.render_layout_elements(
        layout_elements, OUTPUT_FILE_PATH, to_top_left=True, multi_pages=False
    )
