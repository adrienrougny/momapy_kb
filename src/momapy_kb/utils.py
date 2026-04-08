import types
import typing
import importlib
import collections.abc
import operator
import functools

import pylpg.relationship
import pylpg.types
import colorama


def evaluate_forward_ref(forward_ref: typing.ForwardRef) -> type:
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
    try:
        type_ = forward_ref._evaluate(
            globalns=globals(),
            localns=locals(),
            type_params=frozenset(),
            recursive_guard=set([]),
        )
    except TypeError:
        type_ = forward_ref._evaluate(
            globalns=globals(),
            localns=locals(),
            recursive_guard=set([]),
        )
    return type_


def pretty_print(
    node_cls: type,
    max_depth: int = 0,
    exclude_cls: list[type] | None = None,
    _depth: int = 0,
    _indent: int = 0,
) -> None:
    def _print_with_indent(s, indent):
        s_indents = "\t" * indent
        print(f"{s_indents}{s}")

    if _depth > max_depth:
        return
    if exclude_cls is None:
        exclude_cls = []

    if node_cls in exclude_cls:
        return

    cls_string = f"{colorama.Fore.GREEN}{node_cls.__name__}"
    _print_with_indent(cls_string, _indent)

    primitive_properties = pylpg.types.get_primitive_properties(node_cls)
    for property_name, property_type in primitive_properties.items():
        property_string = (
            f"{colorama.Fore.BLUE}* {property_name}"
            f"{colorama.Fore.MAGENTA} = {colorama.Fore.RED}"
            f"{property_type}{colorama.Style.RESET_ALL}"
        )
        _print_with_indent(property_string, _indent + 1)

    relationship_descriptors = pylpg.types.get_relationship_descriptors(node_cls)
    for rel_name, descriptor in relationship_descriptors.items():
        rel_class = descriptor._relationship_class
        rel_type = getattr(rel_class, "__type__", rel_name)
        property_value_string = (
            f"RelationshipTo({rel_type})"
        )
        property_string = (
            f"{colorama.Fore.BLUE}* {rel_name}"
            f"{colorama.Fore.MAGENTA} = {colorama.Fore.RED}"
            f"{property_value_string}{colorama.Style.RESET_ALL}"
        )
        _print_with_indent(property_string, _indent + 1)


def flatten_collection(input_collection: collections.abc.Sequence) -> list:
    def _flatten_rec(a, b):
        if isinstance(b, collections.abc.Sequence) and not isinstance(
            b, (str, bytes, bytearray)
        ):
            b = flatten_collection(b)
        else:
            b = [b]
        return operator.iconcat(a, b)

    return functools.reduce(_flatten_rec, input_collection, [])
