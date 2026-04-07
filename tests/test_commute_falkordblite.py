import pathlib

import momapy_kb.lpg.core
import momapy_kb.lpg.backends.falkordblite

if __name__ == "__main__":
    covid_dm_input_dir_path = pathlib.Path(
        "/home/rougny/research/commute/commute_dm/data/maps/covid_dm/celldesigner"
    )
    pd_dm_input_dir_path = pathlib.Path(
        "/home/rougny/research/commute/commute_dm/data/maps/pd_dm/celldesigner"
    )
    backend = momapy_kb.lpg.backends.falkordblite.FalkorDBLiteBackend()
    with momapy_kb.lpg.core.Session(backend) as session:
        session.delete_all()
        session.save_collections_from_file_paths(
            [
                ("COVID_DM", covid_dm_input_dir_path.glob("*.xml")),
                ("PD_DM", pd_dm_input_dir_path.glob("*.xml")),
            ],
            return_type="model",
        )
