import momapy.core
import momapy.celldesigner.io.celldesigner
import momapy.io

import momapy_kb.neo4j.core
import credentials


if __name__ == "__main__":
    momapy_kb.neo4j.core.connect(
        credentials.HOST_NAME, credentials.USER_NAME, credentials.PASSWORD
    )
    momapy_kb.neo4j.core.delete_all()
    m = momapy.io.read(
        "/home/rougny/research/commute/commute_dm_develop/build/maps/pd/celldesigner/Neuroinflammation.xml"
    )
    momapy_kb.neo4j.core.save_node_from_object(m, object_to_node_mode="hash")
