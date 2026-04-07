import momapy.sbml.io.sbml
import momapy.io

import momapy_kb.lpg.core
import momapy_kb.lpg.backends.neo4j
import credentials


if __name__ == "__main__":
    backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
        hostname=credentials.HOST_NAME,
        username=credentials.USER_NAME,
        password=credentials.PASSWORD,
    )
    with momapy_kb.lpg.core.Session(backend) as session:
        session.delete_all()
        # session.save_from_object(m, integration_mode="hash")
