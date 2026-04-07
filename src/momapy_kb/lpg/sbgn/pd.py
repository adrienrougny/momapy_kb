import sys

import fieldz_kb.lpg.core

import momapy.core
import momapy.sbgn.pd

ctx = fieldz_kb.lpg.core.get_default_context()
module = momapy.sbgn.pd
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
            node_class = fieldz_kb.lpg.core.get_or_make_node_class_from_type(
                ctx, attr_value, make_node_classes_recursively=True
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)
