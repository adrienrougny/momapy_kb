import colorama

import neomodel


def get_properties_from_node_cls(node_cls):
    properties = []
    for property_name in node_cls.defined_properties():
        property_args = {}
        property_ = getattr(node_cls, property_name)
        property_attrs = {
            "name": property_name,
            "args": property_args,
            "obj": property_,
        }
        if isinstance(property_, neomodel.RelationshipTo):
            property_attrs["obj"] = property_
            property_attrs["type"] = "relationship"
            property_.lookup_node_class()
            property_args["node_cls"] = property_.definition["node_class"]
            property_args["name"] = property_.definition["relation_type"]
            property_args["cardinality"] = property_.manager.__name__
        else:
            property_attrs["type"] = "property"
        properties.append(property_attrs)
    return properties


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
