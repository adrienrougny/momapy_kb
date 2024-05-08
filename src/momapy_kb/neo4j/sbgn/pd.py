import sys

import momapy_kb.neo4j.core

import momapy.core
import momapy.sbgn.core
import momapy.sbgn.pd

for attr_name in dir(momapy.sbgn.pd):
    if not attr_name.startswith("_"):
        attr_value = getattr(momapy.sbgn.pd, attr_name)
        if isinstance(attr_value, type) and issubclass(
            attr_value,
            (momapy.sbgn.core.SBGNModelElement, momapy.core.Model),
        ):
            node_class = momapy_kb.neo4j.core.make_node_class_from_class(
                attr_value
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)
