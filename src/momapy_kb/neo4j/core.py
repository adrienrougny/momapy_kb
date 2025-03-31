import dataclasses
import typing
import types
import re
import enum
import itertools
import abc
import sys
import collections.abc

import frozendict
import neomodel
import inflect

import neo4j

import momapy.core
import momapy.drawing

import momapy_kb.utils
import momapy_kb.neo4j.utils


def connect(hostname, username, password, protocol="bolt", port="7687"):
    connection_url = f"{protocol}://{username}:{password}@{hostname}:{port}"
    neomodel.db.set_connection(connection_url)
    run("RETURN 1")  # return an error if not connected


def close_connection():
    neomodel.db.close_connection()


def is_connected():
    try:
        run("RETURN 1")
    except neo4j.exceptions.ServiceUnavailable:
        return False
    return True


def delete_all():
    results, meta = neomodel.db.cypher_query("MATCH (n) DETACH DELETE n")


def run(query):
    result, meta = neomodel.db.cypher_query(query)
    return result, meta


def _make_relationship_name_from_attr_name(attr_name, many=False):
    if many:
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
    else:
        rtype = f"HAS_{attr_name.upper()}"
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
            break
    return transform_func


def _get_final_types_from_basetype(
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


def _get_final_types_from_forwardref(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_ = momapy_kb.utils.evaluate_forward_ref(type_)
    return _get_final_types_from_type(
        type_,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


def _get_final_types_from_collection(
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
    return _get_final_types_from_type(
        type_arg,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=True,
    )


def _get_final_types_from_union(
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
            _get_final_types_from_type(
                type_arg,
                required=required,
                many=many,
                ordered=ordered,
                in_collection=False,
            )
            for type_arg in type_args_not_none
        ]
    )


def _get_final_types_from_uniontype(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_args = typing.get_args(type_)
    type_ = typing.Union[tuple(type_args)]
    return _get_final_types_from_type(
        type_,
        required=required,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


def _get_final_types_from_optional(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    type_args = typing.get_args(type_)
    type_arg = type_args[0]
    return _get_final_types_from_type(
        type_arg,
        required=False,
        many=many,
        ordered=ordered,
        in_collection=in_collection,
    )


def _get_final_types_from_dict(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    return [
        (
            typing.get_origin(type_),
            required,
            many,
            ordered,
        )
    ]


_get_final_types_from_type_rules = [
    (
        lambda type_: isinstance(type_, typing.ForwardRef),
        _get_final_types_from_forwardref,
    ),
    (dataclasses.is_dataclass, _get_final_types_from_basetype),
    (
        lambda type_: typing.get_origin(type_)
        in [list, tuple, set, frozenset],
        _get_final_types_from_collection,
    ),
    (
        lambda type_: typing.get_origin(type_) is typing.Union,
        _get_final_types_from_union,
    ),
    (
        lambda type_: typing.get_origin(type_) is types.UnionType,
        _get_final_types_from_uniontype,
    ),
    (
        lambda type_: typing.get_origin(type_) is typing.Optional,
        _get_final_types_from_optional,
    ),
    (
        lambda type_: typing.get_origin(type_)
        in (
            dict,
            frozendict.frozendict,
        ),
        _get_final_types_from_dict,
    ),
    (
        (
            str,
            int,
            float,
            bool,
            list,
            tuple,
            set,
            frozenset,
            dict,
            frozendict.frozendict,
            enum.Enum,
            momapy.drawing.NoneValueType,
            momapy.core.LayoutModelMapping,
        ),
        _get_final_types_from_basetype,
    ),
]


def _get_final_types_from_type(
    type_, required=True, many=False, ordered=False, in_collection=False
):
    transform_func = _get_transform_func_from_rules(
        _get_final_types_from_type_rules, type_
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


def _make_node_class_property_from_str(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.StringProperty())
    else:
        return neomodel.StringProperty(required=required)


def _make_node_class_property_from_int(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.IntegerProperty())
    else:
        return neomodel.IntegerProperty(required=required)


def _make_node_class_property_from_float(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.FloatProperty())
    else:
        return neomodel.FloatProperty(required=required)


def _make_node_class_property_from_bool(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if many:
        return neomodel.ArrayProperty(neomodel.BooleanProperty())
    else:
        return neomodel.BooleanProperty(required=required)


def _make_node_class_property_from_enum(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    return _make_node_class_property_from_str(
        type_, attr_name, required=required, many=many, ordered=ordered
    )


def _make_node_class_property_from_none_value_type(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    return _make_node_class_property_from_str(
        type_, attr_name, required=required, many=many, ordered=ordered
    )


def _make_node_class_property_from_dataclass(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if _ongoing is None:
        _ongoing = set([])
    if type_ not in _ongoing:
        node_cls = make_node_class_from_class(type_, _ongoing=_ongoing)
    else:
        # if we are already in the process of making the node class, we use a
        # reference of it instead, to avoid infinite recursion
        node_cls = _node_cls_name_from_cls_name(type_.__name__)
    relationship_name = _make_relationship_name_from_attr_name(
        attr_name, many=many
    )
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


def _make_node_class_property_from_dict(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if _ongoing is None:
        _ongoing = set([])
    node_cls = make_node_class_from_class(type_, _ongoing=_ongoing)
    relationship_name = _make_relationship_name_from_attr_name(
        attr_name, many=many
    )
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


_make_node_class_property_from_final_type_rules = [
    (str, _make_node_class_property_from_str),
    (int, _make_node_class_property_from_int),
    (float, _make_node_class_property_from_float),
    (bool, _make_node_class_property_from_bool),
    (enum.Enum, _make_node_class_property_from_enum),
    (
        momapy.drawing.NoneValueType,
        _make_node_class_property_from_none_value_type,
    ),
    (dataclasses.is_dataclass, _make_node_class_property_from_dataclass),
    (
        (
            dict,
            frozendict.frozendict,
        ),
        _make_node_class_property_from_dict,
    ),
]


def _make_node_class_property_from_final_type(
    type_, attr_name, required=True, many=False, ordered=False, _ongoing=None
):
    if _ongoing is None:
        _ongoing = set([])
    transform_func = _get_transform_func_from_rules(
        _make_node_class_property_from_final_type_rules, type_
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


def _make_node_class_property_name_from_attr_name(attr_name):
    if attr_name == "id":
        return "uid"
    return attr_name


def _make_node_class_properties_from_type(attr_type, attr_name, _ongoing=None):
    if _ongoing is None:
        _ongoing = set([])
    node_cls_properties = []
    for final_type, required, many, ordered in _get_final_types_from_type(
        attr_type
    ):
        node_cls_property = _make_node_class_property_from_final_type(
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
                f"{_make_node_class_property_name_from_attr_name(node_cls_property.attr_name)}_"
                f"{node_cls_property.final_type.__name__}"
            )
    else:
        node_cls_properties[0].name = (
            _make_node_class_property_name_from_attr_name(attr_name)
        )
    return node_cls_properties


class MomapyKBNode(neomodel.StructuredNode):
    _cls_to_build = None


class String(MomapyKBNode):
    _cls_to_build = str
    value = neomodel.StringProperty(required=True)


class Integer(MomapyKBNode):
    _cls_to_build = int
    value = neomodel.IntegerProperty(required=True)


class Float(MomapyKBNode):
    _cls_to_build = float
    value = neomodel.FloatProperty(required=True)


class Boolean(MomapyKBNode):
    _cls_to_build = bool
    value = neomodel.BooleanProperty(required=True)


class Item(MomapyKBNode):
    _cls_to_build = None
    key = neomodel.RelationshipTo(MomapyKBNode, "HAS_KEY", neomodel.One)
    value = neomodel.RelationshipTo(MomapyKBNode, "HAS_VALUE", neomodel.One)


class Mapping(MomapyKBNode):
    _cls_to_build = dict
    items = neomodel.RelationshipTo(Item, "HAS_ITEM", neomodel.ZeroOrMore)


class Bag(MomapyKBNode):
    _cls_to_build = set
    elements = neomodel.RelationshipTo(
        MomapyKBNode, "HAS_ELEMENT", neomodel.ZeroOrMore
    )


class Sequence(MomapyKBNode):
    _cls_to_build = list
    elements = neomodel.RelationshipTo(
        MomapyKBNode,
        "HAS_ELEMENT",
        neomodel.ZeroOrMore,
        model=_OrderedRelationshipTo,
    )


_class_to_node_class = {
    str: String,
    int: Integer,
    float: Float,
    bool: Boolean,
    dict: Mapping,
    frozendict.frozendict: Mapping,
    set: Bag,
    frozenset: Bag,
    list: Sequence,
    tuple: Sequence,
}


def _node_cls_name_from_cls_name(cls_name):
    return f"{cls_name}"


def _make_node_class_from_basetype(cls, _ongoing=None):
    return _class_to_node_class[cls]


def _make_node_class_from_dataclass(cls, _ongoing=None):
    if _ongoing is None:
        _ongoing = set([])
    node_cls = _class_to_node_class.get(cls)
    if node_cls is not None:
        return node_cls
    _ongoing.add(cls)
    node_cls_name = _node_cls_name_from_cls_name(cls.__name__)
    node_cls_ns = {"_cls_to_build": cls}
    for field in dataclasses.fields(cls):
        if not field.name.startswith("_"):  # we do not want private attributes
            node_cls_properties = _make_node_class_properties_from_type(
                field.type, field.name, _ongoing=_ongoing
            )
            for node_cls_property in node_cls_properties:
                node_cls_ns[node_cls_property.name] = node_cls_property
                # we make sure potential more general parent properties are not made
                if (
                    node_cls_property.name
                    != _make_node_class_property_name_from_attr_name(
                        node_cls_property.attr_name
                    )
                ):
                    node_cls_ns[node_cls_property.attr_name] = None
    cls_bases = cls.__bases__
    node_cls_bases = tuple(
        [
            make_node_class_from_class(cls_base, _ongoing=_ongoing)
            for cls_base in cls_bases
            if cls_base not in (object, abc.ABC, collections.abc.Mapping)
        ]
    )
    if not node_cls_bases:
        node_cls_bases = tuple([MomapyKBNode])
    node_cls = type(node_cls_name, node_cls_bases, node_cls_ns)
    _class_to_node_class[cls] = node_cls
    setattr(sys.modules[__name__], node_cls.__name__, node_cls)
    _ongoing.remove(cls)
    return node_cls


make_node_class_from_class_rules = [
    (
        (
            str,
            int,
            float,
            bool,
            list,
            tuple,
            set,
            frozenset,
            dict,
            frozendict.frozendict,
        ),
        _make_node_class_from_basetype,
    ),
    (
        dataclasses.is_dataclass,
        _make_node_class_from_dataclass,
    ),
]


def make_node_class_from_class(cls, _ongoing=None):
    if _ongoing is None:
        _ongoing = set([])
    transform_func = _get_transform_func_from_rules(
        make_node_class_from_class_rules, cls
    )
    if transform_func is None:
        raise ValueError(
            f"could not get transformation function for class {cls}"
        )
    node_cls = transform_func(cls, _ongoing=_ongoing)
    return node_cls


def _make_node_attr_value_from_basetype_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    return obj


def _make_node_attr_value_from_collection_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    return [
        _make_node_attr_value_from_object(
            element,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        for element in obj
    ]


def _make_node_attr_value_from_dataclass_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    node = save_node_from_object(
        obj,
        object_to_node=object_to_node,
        object_to_node_mode=object_to_node_mode,
        object_to_node_exclude=object_to_node_exclude,
    )
    return node


def _make_node_attr_value_from_enum_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
) -> str:
    return f"{type(obj).__name__}.{obj.name}"


def _make_node_attr_value_from_none_value_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
) -> str:
    return "none"


def _make_node_attr_value_from_dict_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    node = save_node_from_object(
        obj,
        object_to_node=object_to_node,
        object_to_node_mode=object_to_node_mode,
        object_to_node_exclude=object_to_node_exclude,
    )
    return node


_make_node_attr_value_from_object_rules = [
    (
        (str, int, float, bool, types.NoneType),
        _make_node_attr_value_from_basetype_object,
    ),
    (
        (list, tuple, set, frozenset),
        _make_node_attr_value_from_collection_object,
    ),
    (enum.Enum, _make_node_attr_value_from_enum_object),
    (
        momapy.drawing.NoneValueType,
        _make_node_attr_value_from_none_value_object,
    ),
    (dataclasses.is_dataclass, _make_node_attr_value_from_dataclass_object),
    (
        (dict, frozendict.frozendict),
        _make_node_attr_value_from_dict_object,
    ),
]


def _make_node_attr_value_from_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    transform_func = _get_transform_func_from_rules(
        _make_node_attr_value_from_object_rules, type(obj)
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
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
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
        else:
            node = None
        if node is not None:
            return node
    node_cls = make_node_class_from_class(type(obj))
    node = node_cls(value=obj)
    node.save()
    if object_to_node_mode == "id":
        object_to_node[id(obj)] = node
    elif object_to_node_mode == "hash":
        object_to_node[obj] = node
    return node


def _save_node_from_enum_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
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
        else:
            node = None
        if node is not None:
            return node
    node_cls = make_node_class_from_class(type(obj))
    node = node_cls(
        name=obj.name,
        value=str(obj.value),
    )
    node.save()
    if object_to_node_mode == "id":
        object_to_node[id(obj)] = node
    elif object_to_node_mode == "hash":
        object_to_node[obj] = node
    return node


def _save_node_from_dataclass_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
) -> neomodel.StructuredNode:
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
    node_cls = make_node_class_from_class(type(obj))
    kwargs = {}
    to_connect = []
    for node_cls_property in momapy_kb.neo4j.utils.get_properties(node_cls):
        node_attr_name = node_cls_property.name
        obj_attr_name = node_cls_property.attr_name
        obj_attr_value = getattr(obj, obj_attr_name)
        if obj_attr_value is not None:
            if isinstance(node_cls_property, neomodel.RelationshipTo):
                if momapy_kb.neo4j.utils.is_many(node_cls_property):
                    for i, obj_attr_value_element in enumerate(obj_attr_value):
                        if isinstance(
                            obj_attr_value_element,
                            node_cls_property.final_type,
                        ):
                            node_attr_value_element = save_node_from_object(
                                obj_attr_value_element,
                                object_to_node=object_to_node,
                                object_to_node_mode=object_to_node_mode,
                                object_to_node_exclude=object_to_node_exclude,
                            )
                            to_connect.append(
                                (
                                    node_attr_name,
                                    node_attr_value_element,
                                    {"order": i},
                                )
                            )
                else:
                    if isinstance(
                        obj_attr_value,
                        node_cls_property.final_type,
                    ):
                        node_attr_value = save_node_from_object(
                            obj_attr_value,
                            object_to_node=object_to_node,
                            object_to_node_mode=object_to_node_mode,
                            object_to_node_exclude=object_to_node_exclude,
                        )
                        to_connect.append(
                            (
                                node_attr_name,
                                node_attr_value,
                                {},
                            )
                        )
            else:
                kwargs[node_attr_name] = _make_node_attr_value_from_object(
                    obj_attr_value,
                    object_to_node=object_to_node,
                    object_to_node_mode=object_to_node_mode,
                    object_to_node_exclude=object_to_node_exclude,
                )
    node = node_cls(**kwargs)
    node.save()
    for (
        node_attr_name,
        node_attr_value,
        relationship_properties,
    ) in to_connect:
        relationship_manager = getattr(node, node_attr_name)
        relationship_manager.connect(node_attr_value, relationship_properties)
    if object_to_node_mode == "id":
        object_to_node[id(obj)] = node
    elif object_to_node_mode == "hash":
        object_to_node[obj] = node
    return node


def _save_node_from_list_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    if object_to_node is None:
        object_to_node = {}
    if object_to_node_exclude is None:
        object_to_node_exclude = tuple([])
    if not isinstance(obj, object_to_node_exclude):
        if object_to_node_mode == "id":
            node = object_to_node.get(id(obj))
        else:
            node = None
        if node is not None:
            return node
    node_cls = make_node_class_from_class(type(obj))
    node = node_cls()
    node.save()
    for i, element in enumerate(obj):
        node_element = save_node_from_object(
            obj=element,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        node.elements.connect(node_element, {"order": i})
    if object_to_node_mode == "id":
        object_to_node[id(obj)] = node
    return node


def _save_node_from_set_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    if object_to_node is None:
        object_to_node = {}
    if object_to_node_exclude is None:
        object_to_node_exclude = tuple([])
    if not isinstance(obj, object_to_node_exclude):
        if object_to_node_mode == "id":
            node = object_to_node.get(id(obj))
        else:
            node = None
        if node is not None:
            return node
    node_cls = make_node_class_from_class(type(obj))
    node = node_cls()
    node.save()
    for element in obj:
        node_element = save_node_from_object(
            obj=element,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        node.elements.connect(node_element)
    if object_to_node_mode == "id":
        object_to_node[id(obj)] = node
    return node


def _save_node_from_dict_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    if object_to_node is None:
        object_to_node = {}
    if object_to_node_exclude is None:
        object_to_node_exclude = tuple([])
    if not isinstance(obj, object_to_node_exclude):
        if object_to_node_mode == "id":
            node = object_to_node.get(id(obj))
        else:
            node = None
        if node is not None:
            return node
    node_cls = make_node_class_from_class(type(obj))
    node = node_cls()
    node.save()
    for key, value in obj.items():
        node_item = Item()
        node_item.save()
        node_key = save_node_from_object(
            obj=key,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        node_value = save_node_from_object(
            obj=value,
            object_to_node=object_to_node,
            object_to_node_mode=object_to_node_mode,
            object_to_node_exclude=object_to_node_exclude,
        )
        node_item.key.connect(node_key)
        node_item.value.connect(node_value)
        node.items.connect(node_item)
    if object_to_node_mode == "id":
        object_to_node[id(obj)] = node
    return node


_save_node_from_object_rules = [
    ((str, int, float, bool), _save_node_from_basetype_object),
    (enum.Enum, _save_node_from_enum_object),
    (dataclasses.is_dataclass, _save_node_from_dataclass_object),
    ((list, tuple), _save_node_from_list_object),
    ((set, frozenset), _save_node_from_set_object),
    ((dict, frozendict.frozendict), _save_node_from_dict_object),
]


def save_node_from_object(
    obj,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
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
    return node


def save_nodes_from_objects(
    objs,
    object_to_node: dict[typing.Any, neomodel.StructuredNode] | None = None,
    object_to_node_mode: typing.Literal["none", "id", "hash"] = "id",
    object_to_node_exclude: tuple[type] | None = None,
):
    if object_to_node is None:
        object_to_node = {}
    if object_to_node_exclude is None:
        object_to_node_exclude = tuple([])
    nodes = []
    for obj in objs:
        node = save_node_from_object(
            obj, object_to_node, object_to_node_mode, object_to_node_exclude
        )
        nodes.append(node)
    return nodes
