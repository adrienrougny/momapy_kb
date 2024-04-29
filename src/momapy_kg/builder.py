import abc
import dataclasses
import typing
import types
import re
import enum
import collections
import sys
import importlib

import neomodel
import inflect

import momapy_neo4j.utils


class MomapyNode(neomodel.StructuredNode):
    _cls_to_build: typing.ClassVar[type]
    _metadata: typing.ClassVar[
        dict
    ]  # {property_name: {attr_name:, attr_type; attr_final_type:}}

    @abc.abstractmethod
    def build(
        self,
        inside_collections: bool = True,
        node_to_object: dict[int, typing.Any] | None = None,
    ):
        pass

    @classmethod
    @abc.abstractmethod
    def save_from_object(
        cls,
        obj,
        inside_collections: bool = True,
        omit_keys: bool = True,
        object_to_node: dict[int, "MomapyNode"] | None = None,
    ):
        pass


node_classes = {}

_base_type_to_properties = {
    int: neomodel.IntegerProperty,
    str: neomodel.StringProperty,
    float: neomodel.FloatProperty,
    bool: neomodel.BooleanProperty,
}

_collection_type_to_properties = {
    list: neomodel.ArrayProperty,
    tuple: neomodel.ArrayProperty,
    frozenset: neomodel.ArrayProperty,
    set: neomodel.ArrayProperty,
}

_attr_name_to_property_name = {"id": "uid"}
_inflect_engine = inflect.engine()


def _evaluate_forward_ref(forward_ref):
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
    type_ = forward_ref._evaluate(globals(), locals(), frozenset())
    return type_


def _make_rtype_from_attr_name(attr_name):
    attr_name = re.sub(
        "(.)([A-Z][a-z]+)",
        r"\1_\2",
        attr_name,
    )
    attr_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", attr_name).lower()
    plurals = attr_name.split("_")
    singulars = []
    for i, plural in enumerate(plurals):
        singular = _inflect_engine.singular_noun(
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


def _make_relationship(
    relationship_cls, node_cls_name, attr_name, required=True, many=False
):
    rtype = _make_rtype_from_attr_name(attr_name)
    if many:
        if required:
            cardinality = neomodel.OneOrMore
        else:
            cardinality = neomodel.ZeroOrMore
    else:
        if required:
            cardinality = neomodel.One
        else:
            cardinality = neomodel.ZeroOrOne
    relationship = relationship_cls(
        node_cls_name,
        rtype,
        cardinality=cardinality,
    )
    return relationship


def _make_property(property_cls, sub_property=None, required=True):
    property_ = property_cls(sub_property, required=required)
    return property_


def _type_to_properties(
    type_, attr_name, super_type=None, required=True, many=False
):
    """
    ForwardRef(type)->
    """
    properties_and_types = []
    if isinstance(
        type_, typing.ForwardRef
    ):  # TO DO: should find if type is already in nodes first
        type_ = _evaluate_forward_ref(type_)
        node_cls_name = f"{type_.__name__}Node"
        relationship = _make_relationship(
            neomodel.RelationshipTo,
            node_cls_name,
            attr_name,
            required=required,
            many=many,
        )
        properties_and_types = [
            (
                relationship,
                super_type,
                type_,
            )
        ]
        return properties_and_types
    # We get the origin of type_, e.g., if type_ = X[Y, Z, ...] we get X
    o_type = typing.get_origin(type_)  # returns None if not supported
    if o_type is not None:
        if isinstance(o_type, type):  # o_type is a type
            if o_type == types.UnionType:  # from t1 | t2 | ... syntax
                a_types = typing.get_args(type_)
                type_ = typing.Union[tuple(a_types)]
                properties_and_types = _type_to_properties(
                    type_, attr_name, super_type=super_type, required=True
                )
            elif (
                o_type in _collection_type_to_properties
            ):  # a supported collection
                a_types = typing.get_args(type_)
                if (
                    len(a_types) != 1
                ):  # we want type_ to have only one subtype (e.g., list[t], tuple[t])
                    raise ValueError(
                        f"transformation of type {type_} is not supported"
                    )
                sub_type = a_types[0]
                # if sub_type is str, float, int, bool
                if sub_type in _base_type_to_properties:
                    sub_property, _, _ = _type_to_properties(
                        sub_type,
                        attr_name,
                        super_type=o_type,
                        required=False,
                    )[0]
                    property_cls = _collection_type_to_properties[o_type]
                    property_ = _make_property(
                        property_cls,
                        sub_property=sub_property,
                        required=required,
                    )
                    properties_and_types = [
                        (
                            property_,
                            o_type,
                            sub_type,
                        )
                    ]
                else:
                    properties_and_types = _type_to_properties(
                        sub_type,
                        attr_name,
                        super_type=o_type,
                        many=True,
                        required=False,
                    )
        else:  # o_type is a special form from typing
            if o_type == typing.Optional:  # typing.Optional has a unique arg
                a_type = typing.get_args(type_)[0]
                type_ = typing.Union[types.NoneType, a_type]
                properties_and_types = _type_to_properties(
                    sub_type, attr_name, super_type=super_type, required=True
                )
            elif o_type == typing.Union:
                properties_and_types = []
                a_types = typing.get_args(type_)
                b_types = []
                for a_type in a_types:
                    if a_type == types.NoneType:
                        required = False
                    else:
                        b_types.append(a_type)
                for b_type in b_types:
                    sub_properties_and_types = _type_to_properties(
                        b_type,
                        attr_name,
                        super_type=super_type,
                        required=False,
                        many=many,
                    )
                    properties_and_types += sub_properties_and_types
    else:  # no o_type
        property_cls = _base_type_to_properties.get(type_)
        if property_cls is not None:  # base type
            property_ = _make_property(property_cls, required=required)
            properties_and_types = [
                (
                    property_,
                    super_type,
                    type_,
                )
            ]
        elif issubclass(type_, enum.Enum):
            property_ = _make_property(
                neomodel.StringProperty, required=required
            )
            properties_and_types = [
                (
                    property_,
                    super_type,
                    type_,
                )
            ]
        else:  # other complex type
            node_cls = get_or_make_node_cls(type_)
            relationship = _make_relationship(
                neomodel.RelationshipTo,
                node_cls_name=node_cls.__name__,
                attr_name=attr_name,
                required=required,
                many=many,
            )
            properties_and_types = [
                (
                    relationship,
                    super_type,
                    type_,
                )
            ]
    return properties_and_types


def make_node_cls_from_dataclass(cls):

    def _build(
        self,
        inside_collections: bool = True,
        node_to_object: dict[int, typing.Any] | None = None,
    ):
        if node_to_object is not None:
            obj = node_to_object.get(id(self))
            if obj is not None:
                return obj
        else:
            node_to_object = {}
        args = {}
        for field in dataclasses.fields(self._cls_to_build):
            attr_value = getattr(self, field.name)
            args[field.name] = object_from_node(
                node=attr_value,
                inside_collections=inside_collections,
                node_to_object=node_to_object,
            )
        obj = self._cls_to_build(**args)
        node_to_object[id(self)] = obj
        return obj

    def _save_from_object(cls, obj, object_to_node=None):
        if object_to_node is not None:
            node = object_to_node.get(id(obj))
            if node is not None:
                return node
        else:
            object_to_node = {}
        args = {}
        to_connect = collections.defaultdict(list)
        properties = momapy_neo4j.utils.get_properties_from_node_cls(cls)
        for property_ in properties:
            property_name = property_["name"]
            if property_name in cls._metadata:
                property_metadata = cls._metadata[property_name]
                attr_name = property_metadata["attr_name"]
                attr_type = property_metadata["attr_type"]
                attr_final_type = property_metadata["attr_final_type"]
            else:
                attr_name = property_name
                attr_type = None
                attr_final_type = None
            attr_value = getattr(obj, attr_name)
            if attr_type is None or isinstance(attr_value, attr_type):
                if property_["type"] == "relationship":
                    cardinality = property_["args"]["cardinality"]
                    if cardinality.endswith("OrMore"):
                        element_values = attr_value
                    else:
                        element_values = [attr_value]
                    for element_value in element_values:
                        if attr_final_type is None or isinstance(
                            element_value, attr_final_type
                        ):
                            element_node = save_node_from_object(
                                element_value, object_to_node=object_to_node
                            )
                            to_connect[property_name].append(element_node)
                else:
                    if attr_final_type is None or isinstance(
                        attr_value, attr_final_type
                    ):
                        if isinstance(attr_value, enum.Enum):
                            attr_value = f"{type(attr_value).__name__}.{attr_value.name}"
                        args[property_name] = attr_value
        node = cls(**args)
        node.save()
        for property_name, element_nodes in to_connect.items():
            for element_node in element_nodes:
                if element_node is not None:
                    getattr(node, property_name).connect(element_node)
        object_to_node[id(obj)] = node
        return node

    node_cls_name = f"{cls.__name__}Node"
    node_cls_metadata = {}
    node_cls_ns = {
        "__module__": sys.modules[__name__],
        "_cls_to_build": cls,
        "_metadata": node_cls_metadata,
        "build": _build,
        "save_from_object": classmethod(_save_from_object),
    }
    for field in dataclasses.fields(cls):
        attr_name = field.name
        attr_type = field.type
        properties_and_types = _type_to_properties(attr_type, attr_name)
        if attr_name in _attr_name_to_property_name:
            property_name = _attr_name_to_property_name[attr_name]
        else:
            property_name = attr_name
        if len(properties_and_types) == 1:
            property_, type_, final_type = properties_and_types[0]
            node_cls_ns[property_name] = property_
            node_cls_metadata[property_name] = {
                "attr_name": attr_name,
                "attr_type": type_,
                "attr_final_type": final_type,
            }
        else:
            for property_, type_, final_type in properties_and_types:
                if type_ is not None:
                    suffix = f"{type_.__name__}_{final_type.__name__}"
                else:
                    suffix = final_type.__name__
                property_name = f"{attr_name}_{suffix}"
                node_cls_ns[property_name] = property_
                node_cls_metadata[property_name] = {
                    "attr_name": attr_name,
                    "attr_type": type_,
                    "attr_final_type": final_type,
                }
    cls_bases = cls.__bases__
    node_cls_bases = []
    for cls_base in cls_bases:
        node_cls_base = get_or_make_node_cls(cls_base)
        if node_cls_base is not None:
            node_cls_bases.append(node_cls_base)
    if not node_cls_bases:
        node_cls_bases = [MomapyNode]
    node_cls = type(node_cls_name, tuple(node_cls_bases), node_cls_ns)
    return node_cls


def get_or_make_node_cls(cls):
    node_cls = node_classes.get(cls)
    if node_cls is None:
        if dataclasses.is_dataclass(cls):
            node_cls = make_node_cls_from_dataclass(cls)
        else:
            node_cls = None
        if node_cls is not None:
            register_node_class(node_cls)
    return node_cls


def register_node_class(node_cls):
    node_classes[node_cls._cls_to_build] = node_cls
    setattr(sys.modules[__name__], node_cls.__name__, node_cls)


def object_from_node(node):
    if hasattr(node, "build"):
        return node.build()
    return node


def save_node_from_object(obj, object_to_node=None):
    if object_to_node is not None:
        node = object_to_node.get(id(obj))
        if node is not None:
            return node
    else:
        object_to_node = {}
    node_cls = get_or_make_node_cls(type(obj))
    if node_cls is None:
        raise ValueError(f"could not make a node class for object {obj}")
    node = node_cls.save_from_object(obj, object_to_node=object_to_node)
    object_to_node[id(obj)] = node
    return node
