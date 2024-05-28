import sys

import momapy_kb.neo4j.core

import momapy.core
import momapy.celldesigner.core

for attr_name in dir(momapy.sbgn.pd):
    if not attr_name.startswith("_"):
        attr_value = getattr(momapy.sbgn.pd, attr_name)
        if isinstance(attr_value, type) and issubclass(
            attr_value,
            (momapy.celldesigner.CellDesignerModelElement, momapy.core.Model),
        ):
            node_class = momapy_kb.neo4j.core.make_node_class_from_class(
                attr_value
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)
