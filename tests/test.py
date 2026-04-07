import os

import momapy.core
import momapy.builder
import momapy.celldesigner.io.celldesigner
import momapy.io

import momapy_kb.lpg.core
import momapy_kb.lpg.backends.neo4j
import credentials


def list_dir(path):
    files = []
    for file_name in os.listdir(path):
        if not file_name.startswith("."):
            files.append((file_name, os.path.join(path, file_name)))
            break
    return files


def read_and_save_map(session, file_name, file_path):
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
            session.save_from_object(m)
        except Exception as e:
            print(f"error in storing {file_path}")


if __name__ == "__main__":
    backend = momapy_kb.lpg.backends.neo4j.Neo4jBackend(
        hostname=credentials.HOST_NAME,
        username=credentials.USER_NAME,
        password=credentials.PASSWORD,
    )
    with momapy_kb.lpg.core.Session(backend) as session:
        pass
