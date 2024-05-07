import dataclasses
import typing
import types
import re
import enum
import pathlib
import itertools
import abc
import sys

import neomodel
import inflect
import jinja2

import momapy.drawing

import momapy_kb.utils


def connect(hostname, username, password, protocol="bolt", port="7687"):

    connection_str = f"{protocol}://{username}:{password}@{hostname}:{port}"
    neomodel.config.DATABASE_URL = connection_str


def delete_all():
    results, meta = neomodel.db.cypher_query("MATCH (n) DETACH DELETE n")


def query(query_str):
    results, meta = neomodel.db.cypher_query(query_str)
    return results, meta


def _get_properties(node_cls):
    properties = []
    for property_name in node_cls.defined_properties():
        property_ = getattr(node_cls, property_name)
        if property_ is not None:
            properties.append(property_)
    return properties


def _is_required(property_):
    if isinstance(property_, neomodel.Property):
        return property_.required
    elif isinstance(property_, neomodel.RelationshipTo):
        if (
            property_.manager == neomodel.One
            or property_.manager == neomodel.OneOrMore
        ):
            return True
    return False


def _is_many(property_):
    if isinstance(property_, neomodel.ArrayProperty):
        return True
    elif isinstance(property_, neomodel.RelationshipTo):
        if (
            property_.manager == neomodel.ZeroOrMore
            or property_.manager == neomodel.OneOrMore
        ):
            return True
    return False


def _is_ordered(property_):
    return hasattr(property_, "order")


def _make_relationship_name_from_attr_name(attr_name):
    inflect_engine = inflect.engine()
    attr_name = re.sub(
        "(.)([A-Z][a-z]+)",
        r"\1_\2",
        attr_name,
    )
    attr_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", attr_name).lower()
    plurals = attr_name.split("_")
    singulars = []
    for i, plural in enumerate(plurals):
        singular = inflect_engine.singular_noun(
            plural
        )  # returns False if already singular
        if singular and singular != plural:
            singulars.append(singular)
            break
        else:
            singulars.append(plural)
    singulars += plurals[i + 1 :]
    singulars = [singular.upper() for singular in singulars]
    rtype = f"HAS_{'_'.join(singulars)}"
    return rtype


class _OrderedRelationshipTo(neomodel.StructuredRel):
    order = neomodel.IntegerProperty(required=True)


def _get_transform_func_from_rules(rules, cls):
    transform_func = None
    for rule in rules:
        condition = rule[0]
        if isinstance(condition, type) or isinstance(condition, tuple):
            if isinstance(cls, type):
                evaluation = issubclass(cls, condition)
            else:
                continue
        else:
            evaluation = condition(cls)
        if evaluation:
            transform_func = rule[1]
    return transform_func


def _final_types_from_basetype(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    return [
        (
            type_,
            required,
            many,
            ordered,
        )
    ]


def _final_types_from_forwardref(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_ = momapy_kb.utils.evaluate_forward_ref(type_)
    return _final_types_from_type(
        type_,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


def _final_types_from_collection(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_origin = typing.get_origin(type_)
    type_args = typing.get_args(type_)
    type_arg = type_args[0]
    many = True
    if type_origin in [list, tuple]:
        ordered = True
    else:
        ordered = False
    return _final_types_from_type(
        type_arg,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=True,
    )


def _final_types_from_union(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_args = typing.get_args(type_)
    type_args_not_none = []
    for type_arg in type_args:
        if type_arg is types.NoneType:
            if not in_collection:
                required = False
        else:
            type_args_not_none.append(type_arg)
    return itertools.chain.from_iterable(
        [
            _final_types_from_type(
                type_arg,
                required=required,
                many=many,
                ordered=ordered,
                in_collection=False,
            )
            for type_arg in type_args_not_none
        ]
    )


def _final_types_from_uniontype(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_args = typing.get_args(type_)
    type_ = typing.Union[tuple(type_args)]
    return _final_types_from_type(
        type_,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


def _final_types_from_optional(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_args = typing.get_args(type_)
    type_arg = type_args[0]
    return _final_types_from_type(
        type_arg,
        required=False,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


_final_types_from_type_rules = [
    (
        (str, int, float, bool, momapy.drawing.NoneValueType, enum.Enum),
        _final_types_from_basetype,
    ),
    (
        lambda type_: isinstance(type_, typing.ForwardRef),
        _final_types_from_forwardref,
    ),
    (dataclasses.is_dataclass, _final_types_from_basetype),
    (
        lambda type_: typing.get_origin(type_)
        in [list, tuple, set, frozenset],
        _final_types_from_collection,
    ),
    (
        lambda type_: typing.get_origin(type_) is typing.Union,
        _final_types_from_union,
    ),
    (
        lambda type_: typing.get_origin(type_) is types.UnionType,
        _final_types_from_uniontype,
    ),
    (
        lambda type_: typing.get_origin(type_) is typing.Optional,
        _final_types_from_optional,
    ),
]


def _final_types_from_type(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    transform_func = _get_transform_func_from_rules(
        _final_types_from_type_rules, type_
    )
    if transform_func is None:
        raise ValueError(f"unsupported type {type_}")
    return transform_func(
        type_,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


def _node_cls_property_from_str(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.StringProperty())
    else:
        return neomodel.StringProperty(required=required)


def _node_cls_property_from_int(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.IntegerProperty())
    else:
        return neomodel.IntegerProperty(required=required)


def _node_cls_property_from_float(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.FloatProperty())
    else:
        return neomodel.FloatProperty(required=required)


def _node_cls_property_from_bool(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.BooleanProperty())
    else:
        return neomodel.BooleanProperty(required=required)


def _node_cls_property_from_enum(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    return _node_cls_property_from_str(
        type_, attr_name, required=required, many=many, ordered=ordered
    )


def _node_cls_property_from_none_value_type(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    return _node_cls_property_from_str(
        type_, attr_name, required=required, many=many, ordered=ordered
    )


def _node_cls_property_from_dataclass(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if _ongoing is None:
        _ongoing = set([])
    if type_ in _cls_to_node_cls or type_ not in _ongoing:
        node_cls = _node_cls_from_cls(type_, _ongoing=_ongoing)
        _ongoing.add(type_.__name__)
    else:
        node_cls = _node_cls_name_from_cls_name(type_.__name__)
    # if we are already in the process of making the node class, we use a
    # reference of it
    relationship_name = _make_relationship_name_from_attr_name(attr_name)
    if required:
        if many:
            cardinality = neomodel.OneOrMore
        else:
            cardinality = neomodel.One
    else:
        if many:
            cardinality = neomodel.ZeroOrMore
        else:
            cardinality = neomodel.ZeroOrOne
    if ordered:
        model = _OrderedRelationshipTo
    else:
        model = neomodel.StructuredRel
    node_cls_property = neomodel.RelationshipTo(
        node_cls, relationship_name, cardinality=cardinality, model=model
    )
    return node_cls_property


_node_cls_property_from_final_type_rules = [
    (str, _node_cls_property_from_str),
    (int, _node_cls_property_from_int),
    (float, _node_cls_property_from_float),
    (bool, _node_cls_property_from_bool),
    (enum.Enum, _node_cls_property_from_enum),
    (momapy.drawing.NoneValueType, _node_cls_property_from_none_value_type),
    (dataclasses.is_dataclass, _node_cls_property_from_dataclass),
]


def _node_cls_property_from_final_type(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if _ongoing is None:
        _ongoing = set([])
    transform_func = _get_transform_func_from_rules(
        _node_cls_property_from_final_type_rules, type_
    )
    if transform_func is None:
        raise ValueError(
            f"could not get transformation function for type {type_}"
        )
    node_cls_property = transform_func(
        type_,
        attr_name,
        required=required,
        many=many,
        ordered=ordered,
        _ongoing=_ongoing,
    )
    return node_cls_property


def _node_cls_property_name_from_attr_name(attr_name):
    if attr_name == "id":
        return "uid"
    return attr_name


def _node_cls_properties_from_type(attr_type, attr_name, _ongoing=None):
    if _ongoing is None:
        _ongoing = set([])
    node_cls_properties = []
    for final_type, required, many, ordered in _final_types_from_type(
        attr_type
    ):
        node_cls_property = _node_cls_property_from_final_type(
            final_type,
            attr_name,
            required=required,
            many=many,
            ordered=ordered,
            _ongoing=_ongoing,
        )
        node_cls_property.attr_name = attr_name
        node_cls_property.final_type = final_type
        node_cls_properties.append(node_cls_property)
    if len(node_cls_properties) > 1:
        for node_cls_property in node_cls_properties:
            node_cls_property.name = (
                f"{_node_cls_property_name_from_attr_name(node_cls_property.attr_name)}_"
                f"{node_cls_property.final_type.__name__}"
            )
    else:
        node_cls_properties[0].name = _node_cls_property_name_from_attr_name(
            attr_name
        )
    return node_cls_properties


class MomapyKBNode(neomodel.StructuredNode):
    pass


class StringNode(MomapyKBNode):
    _cls_to_build = str
    value = neomodel.StringProperty(required=True)


class IntegerNode(MomapyKBNode):
    _cls_to_build = int
    value = neomodel.IntegerProperty(required=True)


class FloatNode(MomapyKBNode):
    _cls_to_build = float
    value = neomodel.FloatProperty(required=True)


class BooleanNode(MomapyKBNode):
    _cls_to_build = bool
    value = neomodel.BooleanProperty(required=True)


_cls_to_node_cls = {
    str: StringNode,
    int: IntegerNode,
    float: FloatNode,
    bool: BooleanNode,
}


def _node_cls_name_from_cls_name(cls_name):
    return f"{cls_name}"


def _node_cls_from_basetype(cls, _ongoing=None):
    return _cls_to_node_cls[cls]


def _node_cls_from_dataclass(cls, _ongoing=None):
    if _ongoing is None:
        _ongoing = set([])
    # print(f"Transforming {cls} to node class")
    node_cls = _cls_to_node_cls.get(cls)
    if node_cls is not None:
        # print(f"{cls} already transformed to {node_cls}")
        return node_cls
    node_cls_name = _node_cls_name_from_cls_name(cls.__name__)
    node_cls_ns = {"_cls_to_build": cls}
    for field in dataclasses.fields(cls):
        node_cls_properties = _node_cls_properties_from_type(
            field.type, field.name, _ongoing=_ongoing
        )
        for node_cls_property in node_cls_properties:
            node_cls_ns[node_cls_property.name] = node_cls_property
            # we make sure potential more general parent properties are not made
            if (
                node_cls_property.name
                != _node_cls_property_name_from_attr_name(
                    node_cls_property.attr_name
                )
            ):
                node_cls_ns[node_cls_property.attr_name] = None
    cls_bases = cls.__bases__
    node_cls_bases = tuple(
        [
            _node_cls_from_cls(cls_base, _ongoing=_ongoing)
            for cls_base in cls_bases
            if cls_base
            not in (
                object,
                abc.ABC,
            )
        ]
    )
    if not node_cls_bases:
        node_cls_bases = tuple([MomapyKBNode])
    node_cls = type(node_cls_name, node_cls_bases, node_cls_ns)
    _cls_to_node_cls[cls] = node_cls
    setattr(sys.modules[__name__], node_cls.__name__, node_cls)
    # print(f"{cls} transformed to {node_cls}")
    return node_cls


_node_cls_from_cls_rules = [
    (
        (str, int, float, bool),
        _node_cls_from_basetype,
    ),
    (
        dataclasses.is_dataclass,
        _node_cls_from_dataclass,
    ),
]


def _node_cls_from_cls(cls, _ongoing=None):
    if _ongoing is None:
        _ongoing = set([])
    _ongoing.add(cls)
    transform_func = _get_transform_func_from_rules(
        _node_cls_from_cls_rules, cls
    )
    if transform_func is None:
        raise ValueError(
            f"could not get transformation function for class {cls}"
        )
    node_cls = transform_func(cls, _ongoing=_ongoing)
    return node_cls


def _node_attr_value_from_basetype_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    return obj


def _node_attr_value_from_collection_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    return [
        _node_attr_value_from_object(
            element,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        for element in obj
    ]


def _node_attr_value_from_dataclass_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    node = save_node_from_object(
        obj,
        object_to_node=object_to_node,
        object_to_node_mode=object_to_node_mode,
        object_to_node_exclude=object_to_node_exclude,
    )
    return node


def _node_attr_value_from_enum_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
) -> str:
    return f"{type(obj).__name__}.{obj.name}"


def _node_attr_value_from_none_value_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
) -> str:
    return "none"


_node_attr_value_from_object_rules = [
    (
        (str, int, float, bool, types.NoneType),
        _node_attr_value_from_basetype_object,
    ),
    ((list, tuple, set, frozenset), _node_attr_value_from_collection_object),
    (enum.Enum, _node_attr_value_from_enum_object),
    (momapy.drawing.NoneValueType, _node_attr_value_from_none_value_object),
    (dataclasses.is_dataclass, _node_attr_value_from_dataclass_object),
]


def _node_attr_value_from_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    transform_func = _get_transform_func_from_rules(
        _node_attr_value_from_object_rules, type(obj)
    )
    if transform_func is None:
        raise ValueError(
            f"could not get transformation function for object of type {type(obj)}"
        )
    attr_value = transform_func(
        obj,
        object_to_node=object_to_node,
        object_to_node_mode=object_to_node_mode,
        object_to_node_exclude=object_to_node_exclude,
    )
    return attr_value


def _save_node_from_basetype_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    node_cls = _node_cls_from_cls(type(obj))
    node = node_cls(value=obj)
    node.save()
    return node


def _save_node_from_enum_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    node_cls = _node_cls_from_cls(type(obj))
    node = node_cls(
        name=obj.name,
        value=str(obj.value),
    )
    node.save()
    return node


def _save_node_from_dataclass_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
) -> neomodel.StructuredNode:
    node_cls = _node_cls_from_cls(type(obj))
    kwargs = {}
    to_connect = []
    for node_cls_property in _get_properties(node_cls):
        node_attr_name = node_cls_property.name
        obj_attr_name = node_cls_property.attr_name
        obj_attr_value = getattr(obj, obj_attr_name)
        node_attr_value = _node_attr_value_from_object(
            obj_attr_value,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        if (
            node_attr_value is not None
        ):  # as a consequence, cannot distinguish [] from None if type is list | None
            if isinstance(node_cls_property, neomodel.RelationshipTo):
                if _is_many(node_cls_property):
                    for i, node_attr_element_value in enumerate(
                        node_attr_value
                    ):
                        if issubclass(
                            type(node_attr_element_value)._cls_to_build,
                            node_cls_property.final_type,
                        ):
                            to_connect.append(
                                (
                                    node_attr_name,
                                    node_attr_element_value,
                                    {"order": i},
                                )
                            )
                else:
                    if issubclass(
                        type(node_attr_value)._cls_to_build,
                        node_cls_property.final_type,
                    ):
                        to_connect.append(
                            (node_attr_name, node_attr_value, {})
                        )
            else:
                kwargs[node_attr_name] = node_attr_value
    node = node_cls(**kwargs)
    node.save()
    for (
        node_attr_name,
        node_attr_value,
        relationship_properties,
    ) in to_connect:
        getattr(node, node_attr_name).connect(
            node_attr_value, relationship_properties
        )
    return node


_save_node_from_object_rules = [
    ((str, int, float, bool), _save_node_from_basetype_object),
    (enum.Enum, _save_node_from_enum_object),
    (dataclasses.is_dataclass, _save_node_from_dataclass_object),
]


def save_node_from_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    if object_to_node is None:
        object_to_node = {}
    if object_to_node_exclude is None:
        object_to_node_exclude = tuple([])
    if not isinstance(obj, object_to_node_exclude):
        if object_to_node_mode == "id":
            node = object_to_node.get(id(obj))
        elif object_to_node_mode == "hash":
            node = object_to_node.get(obj)
        if node is not None:
            return node
    # print(f"Saving object of type {type(obj)} to node")
    transform_func = _get_transform_func_from_rules(
        _save_node_from_object_rules, type(obj)
    )
    if transform_func is None:
        raise ValueError(
            f"could not get transformation function for object of type {type(obj)}"
        )
    node = transform_func(
        obj,
        object_to_node=object_to_node,
        object_to_node_mode=object_to_node_mode,
        object_to_node_exclude=object_to_node_exclude,
    )
    if not isinstance(obj, object_to_node_exclude):
        if object_to_node_mode == "id":
            object_to_node[id(obj)] = node
        elif object_to_node_mode == "hash":
            object_to_node[obj] = node
    return node


def make_doc(
    module,
    mode="neo4j",
    recursive=True,
    output_file_path=None,
    exclude: list[type] | None = None,
):

    def _prepare_node_spec(
        node_cls,
        node_label_to_node_spec=None,
        recursive=True,
        exclude=None,
    ):
        if node_label_to_node_spec is None:
            node_label_to_node_spec = {}
        if exclude is None:
            exclude = []
        inherited_labels = node_cls.inherited_labels()
        label = inherited_labels[0]
        del inherited_labels[0]
        properties = []
        relationships = []
        properties_or_relationships = (
            momapy_kb.utils.get_properties_from_node_cls(node_cls)
        )
        for property_or_relationship in properties_or_relationships:
            if property_or_relationship["type"] == "property":
                properties.append(property_or_relationship)
            else:
                relationships.append(property_or_relationship)
        node_spec = {
            "label": label,
            "inherited_labels": inherited_labels,
            "properties": properties,
            "relationships": relationships,
        }
        node_label_to_node_spec[label] = node_spec
        if recursive:
            for relationship in node_spec["relationships"]:
                node_cls = relationship["args"]["node_cls"]
                if (
                    node_cls.__name__ not in node_label_to_node_spec
                    and not any(
                        [
                            issubclass(node_cls._cls_to_build, excluded_cls)
                            for excluded_cls in exclude
                        ]
                    )
                ):
                    node_label_to_node_spec = _prepare_node_spec(
                        relationship["args"]["node_cls"],
                        node_label_to_node_spec=node_label_to_node_spec,
                        recursive=recursive,
                        exclude=exclude,
                    )
        return node_label_to_node_spec

    if exclude is None:
        exclude = []
    current_module_dir = pathlib.Path(__file__).parent
    template_path = current_module_dir / "templates"
    loader = jinja2.FileSystemLoader(template_path)
    environment = jinja2.Environment(loader=loader)
    environment.loader.list_templates()
    if mode == "neo4j":
        template = environment.get_template("doc_neo4j_model.html")
    elif mode == "neomodel":
        template = environment.get_template("doc_neomodel.html")
    else:
        raise ValueError(f"unrecognized mode {mode}")
    node_label_to_node_spec = {}
    for attr_name in dir(module):
        attr_value = getattr(module, attr_name)
        if isinstance(attr_value, type) and not any(
            [issubclass(attr_value, excluded_cls) for excluded_cls in exclude]
        ):
            node_cls = _node_cls_from_cls(attr_value)
            if (
                node_cls is not None
                and node_cls.__name__ not in node_label_to_node_spec
            ):
                node_label_to_node_spec |= _prepare_node_spec(
                    node_cls, exclude=exclude
                )
    node_specs = list(node_label_to_node_spec.values())
    node_specs.sort(key=lambda node_spec: node_spec["label"])
    s = template.render(node_specs=node_specs)
    if output_file_path:
        with open(output_file_path, "w") as f:
            f.write(s)
    else:
        print(s)
