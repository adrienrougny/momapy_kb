import os
import dataclasses
import typing

import momapy.core
import momapy.sbgn.pd
import momapy.sbgn.io.sbgnml
import momapy.io

import momapy_kb.neo4j
import momapy_kb.utils
import credentials


@dataclasses.dataclass
class Collection:
    name: str
    models: list[momapy.core.Model] = dataclasses.field(default_factory=list)


def list_dir(path):
    files = []
    for file_name in os.listdir(path):
        if not file_name.startswith("."):
            files.append((file_name, os.path.join(path, file_name)))
    return files


if __name__ == "__main__":
    momapy_kb.neo4j.connect(
        credentials.HOST_NAME, credentials.USER_NAME, credentials.PASSWORD
    )
    momapy_kb.neo4j.delete_all()
    c = Collection("PD")
    for file_name, file_path in list_dir(
        "/home/rougny/research/commute/commute_dm_develop/build/maps/pd/sbgn/"
    ):
        if "MTOR" in file_name:
            print(file_path)
            m = momapy.io.read(file_path)
            c.models.append(m.model)
    momapy_kb.neo4j.save_node_from_object(c)
