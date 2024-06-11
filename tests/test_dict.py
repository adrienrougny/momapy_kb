import dataclasses

import momapy_kb.neo4j.core

import credentials


@dataclasses.dataclass
class Test:
    a: str
    b: dict[int, str]


if __name__ == "__main__":
    t = Test("z", {1: "x", 2: ["c", "d"]})
    momapy_kb.neo4j.core.connect(
        credentials.HOST_NAME, credentials.USER_NAME, credentials.PASSWORD
    )
    momapy_kb.neo4j.core.delete_all()
    momapy_kb.neo4j.core.save_node_from_object(t)
