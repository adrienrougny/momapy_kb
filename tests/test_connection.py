import momapy_kb.lpg.core
import momapy_kb.lpg.backends.neo4j
import credentials

backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
    hostname=credentials.HOST_NAME,
    username=credentials.USER_NAME,
    password=credentials.PASSWORD,
)
with momapy_kb.lpg.core.Session(backend) as session:
    print("Connected successfully")
    results = session.execute_query("RETURN 1 AS n")
    print(results)
