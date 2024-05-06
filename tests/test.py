import os
import dataclasses

import momapy.core
import momapy.sbgn.pd
import momapy.sbgn.io.sbgnml
import momapy.io

import momapy_kg.neo4j
import momapy_kg.utils
import credentials


@dataclasses.dataclass
class Collection:
    name: str
    maps: list[momapy.core.Map] = dataclasses.field(default_factory=list)


def list_dir(path):
    files = []
    for file_name in os.listdir(path):
        if not file_name.startswith("."):
            files.append((file_name, os.path.join(path, file_name)))
    return files


if __name__ == "__main__":
    momapy_kg.neo4j.connect(
        credentials.HOST_NAME, credentials.USER_NAME, credentials.PASSWORD
    )
    momapy_kg.neo4j.delete_all()
    c = Collection("PD")
    for _, file_path in list_dir(
        "/home/rougny/research/commute/commute_dm_develop/build/maps/pd/sbgn/"
    ):
        print(file_path)
        m = momapy.io.read(file_path)
        c.maps.append(m)
        break
    momapy_kg.neo4j.save_node_from_object(c)
    # momapy_kg.neo4j.make_doc(
    #     momapy.sbgn.pd,
    #     mode="neomodel",
    #     output_file_path="test.html",
    #     exclude=[momapy.core.LayoutElement, momapy.builder.Builder],
    # )
