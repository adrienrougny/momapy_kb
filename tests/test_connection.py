import momapy_kb.neo4j.core
import credentials

print(momapy_kb.neo4j.core.is_connected())
momapy_kb.neo4j.core.connect(
    credentials.HOST_NAME, credentials.USER_NAME, credentials.PASSWORD
)
print(momapy_kb.neo4j.core.is_connected())
