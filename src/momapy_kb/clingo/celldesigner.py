import sys

import fieldz_kb.clingo.core

import momapy.core.elements
import momapy.core.model
import momapy.celldesigner.core

_module = momapy.celldesigner.core

ontology_rules = set([])
for attr_name in dir(_module):
    if not attr_name.startswith("_"):
        attr_value = getattr(_module, attr_name)
        if isinstance(attr_value, type) and issubclass(
            attr_value,
            (
                momapy.core.elements.ModelElement,
                momapy.core.model.Model,
            ),
        ):
            predicate_class = (
                fieldz_kb.clingo.core.get_or_make_predicate_classes_from_type(
                    attr_value, make_predicate_classes_recursively=True
                )[0]
            )
            setattr(sys.modules[__name__], predicate_class.__name__, predicate_class)
            ontology_rules.update(
                fieldz_kb.clingo.core.make_ontology_rules_from_type(attr_value)
            )
ontology_rules = sorted(list(ontology_rules))
