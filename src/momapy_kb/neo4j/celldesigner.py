import sys

import fieldz_kb.neo4j.core

import momapy.core.map
import momapy.core.layout
import momapy.core.model
import momapy.core.elements
import momapy.celldesigner.core

module = momapy.celldesigner.core
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
            node_class = fieldz_kb.neo4j.core.get_or_make_node_class_from_type(
                attr_value, make_node_classes_recursively=True
            )
            setattr(sys.modules[__name__], node_class.__name__, node_class)
