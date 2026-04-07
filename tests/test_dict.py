import dataclasses

import momapy_kb.lpg.core
import momapy_kb.lpg.backends.neo4j
import credentials


@dataclasses.dataclass
class Test:
    a: str
    b: dict[int, str]


if __name__ == "__main__":
    t = Test("z", {1: "x", 2: ["c", "d"]})
    backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
        hostname=credentials.HOST_NAME,
        username=credentials.USER_NAME,
        password=credentials.PASSWORD,
    )
    with momapy_kb.lpg.core.Session(backend) as session:
        session.delete_all()
        session.save_from_object(t)
