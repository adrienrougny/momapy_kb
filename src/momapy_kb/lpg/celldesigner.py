import sys

import fieldz_kb.lpg.core

import momapy.core.map
import momapy.core.layout
import momapy.core.model
import momapy.core.elements
import momapy.celldesigner

import momapy_kb.lpg.types

ctx = fieldz_kb.lpg.core.get_default_context()
momapy_kb.lpg.types.register_momapy_plugins(ctx)
module = momapy.celldesigner
for attr_name in dir(module):
    if not attr_name.startswith("_"):
        attr_value = getattr(module, attr_name)
        if isinstance(attr_value, type) and issubclass(
            attr_value,
            (
                momapy.core.elements.ModelElement,
                momapy.core.elements.LayoutElement,
                momapy.core.map.Map,
                momapy.core.model.Model,
                momapy.core.layout.Layout,
            ),
        ):
            node_class = fieldz_kb.lpg.core.get_or_make_node_class_from_type(
                ctx, attr_value, make_node_classes_recursively=True
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)
