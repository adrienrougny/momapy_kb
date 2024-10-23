import types
import importlib

import neomodel
import colorama


def evaluate_forward_ref(forward_ref):
    forward_module = forward_ref.__forward_module__
    forward_arg = forward_ref.__forward_arg__
    forward_module_name = None
    forward_cls_name = forward_arg
    if forward_module is not None:
        if isinstance(forward_module, types.ModuleType):
            forward_module_name = forward_module.__name__
        elif isinstance(forward_module, str):
            forward_module_name = forward_module
        else:
            raise ValueError(
                f"module argument of {forward_ref} must be 'str' or 'types.ModuleType'"
            )
    else:
        parts = forward_arg.rpartition(".")
        if parts[1]:
            forward_module_name = parts[0]
            forward_cls_name = parts[2]
    if forward_module_name:
        forward_module = importlib.import_module(forward_module_name)
        globals()[forward_module_name] = forward_module
        globals()[forward_cls_name] = getattr(forward_module, forward_cls_name)
    type_ = forward_ref._evaluate(
        globalns=globals(),
        localns=locals(),
        type_params=frozenset(),
        recursive_guard=set([]),
    )
    return type_


def pretty_print(
    neomodel_node_cls, max_depth=0, exclude_cls=None, _depth=0, _indent=0
):
    def _print_with_indent(s, indent):
        s_indents = "\t" * indent
        print(f"{s_indents}{s}")

    def _get_value_string(attr_value, max_len=30):
        s = str(attr_value)
        if len(s) > max_len:
            s = f"{s[:max_len]}..."
        return s

    if _depth > max_depth:
        return
    if exclude_cls is None:
        exclude_cls = []

    if neomodel_node_cls in exclude_cls:
        return

    cls_string = f"{colorama.Fore.GREEN}{neomodel_node_cls.__name__}"
    _print_with_indent(cls_string, _indent)

    for property_name in neomodel_node_cls.defined_properties():
        property_ = getattr(neomodel_node_cls, property_name)
        relationship_node_cls = None
        if isinstance(property_, neomodel.RelationshipTo):
            property_.lookup_node_class()
            relationship_node_cls = property_.definition["node_class"]
            property_value_string = (
                f"RelationshipTo("
                f"{relationship_node_cls.__name__}, "
                f"{property_.definition['relation_type']}, "
                f"{property_.manager.__name__})"
            )
        else:
            property_value_string = f"{type(property_).__name__}()"
        property_string = (
            f"{colorama.Fore.BLUE}* {property_name}"
            f"{colorama.Fore.MAGENTA} = {colorama.Fore.RED}"
            f"{property_value_string}{colorama.Style.RESET_ALL}"
        )
        _print_with_indent(property_string, _indent + 1)
        if relationship_node_cls is not None:
            pretty_print(
                relationship_node_cls,
                max_depth=max_depth,
                exclude_cls=exclude_cls,
                _depth=_depth + 1,
                _indent=_indent + 1,
            )
