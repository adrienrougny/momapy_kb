import os

import momapy.core
import momapy.builder
import momapy.celldesigner.io.celldesigner
import momapy.io

import momapy_kb.neo4j.core
import credentials


def list_dir(path):
    files = []
    for file_name in os.listdir(path):
        if not file_name.startswith("."):
            files.append((file_name, os.path.join(path, file_name)))
            break
    return files


def read_and_save_map(file_name, file_path):
    print(file_path)
    try:
        m = momapy.io.read(file_path)
        m = momapy.builder.builder_from_object(m)
        m.id = file_name
        m = momapy.builder.object_from_builder(m)
    except Exception as e:
        print(f"error in reading: {file_path}")
    else:
        try:
            momapy_kb.neo4j.core.save_node_from_object(m)
        except Exception as e:
            print(f"error in storing {file_path}")


if __name__ == "__main__":
    momapy_kb.neo4j.core.connect(
        credentials.HOST_NAME, credentials.USER_NAME, credentials.PASSWORD
    )
    momapy_kb.neo4j.core.delete_all()
    for file_name, file_path in list_dir(
        "/home/rougny/research/commute/commute_dm_develop/build/maps/covid/celldesigner/"
    ):
        read_and_save_map(file_name, file_path)
    for file_name, file_path in list_dir(
        "/home/rougny/research/commute/commute_dm_develop/build/maps/pd/celldesigner/"
    ):
        read_and_save_map(file_name, file_path)
