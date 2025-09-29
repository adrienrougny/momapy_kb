import sys
import typing

import fieldz_kb.neo4j

import momapy.core
import momapy.celldesigner.core
import momapy.celldesigner.io.celldesigner

import momapy_kb.neo4j.core

module = momapy.celldesigner.core
for attr_name in dir(module):
    if not attr_name.startswith("_"):
        attr_value = getattr(module, attr_name)
        if isinstance(attr_value, type) and issubclass(
            attr_value,
            (
                momapy.core.ModelElement,
                momapy.core.LayoutElement,
                momapy.core.Map,
                momapy.core.Model,
                momapy.core.Layout,
            ),
        ):
            node_class = fieldz_kb.neo4j.core.get_or_make_node_class_from_type(
                attr_value, make_node_classes_recursively=True
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)


def save_from_file(
    file_path,
    return_type: typing.Literal["map", "model", "layout"] = "map",
    with_layout=True,
    with_model=True,
    integration_mode: typing.Literal["hash", "id"] | None = None,
):
    object_ = momapy.io.read(
        file_path=file_path,
        return_type=return_type,
        with_model=with_model,
        with_layout=with_layout,
    ).obj
    momapy_kb.neo4j.core.save_from_object(object_, integration_mode=integration_mode)
