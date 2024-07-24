import sys

import momapy_kb.neo4j.core

import momapy.core
import momapy.celldesigner.core

module = momapy.celldesigner.core

for attr_name in dir(module):
    if not attr_name.startswith("_"):
        attr_value = getattr(module, attr_name)
        if isinstance(attr_value, type) and issubclass(
            attr_value,
            (
                momapy.core.ModelElement,
                momapy.core.Layout,
                momapy.core.Map,
            ),
        ):
            node_class = momapy_kb.neo4j.core.make_node_class_from_class(
                attr_value
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)
