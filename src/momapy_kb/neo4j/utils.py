import pathlib
import typing

import jinja2
import neomodel


def get_node_class(relationship):
    relationship.lookup_node_class()
    return relationship.definition["node_class"]


def get_cardinality(relationship):
    return relationship.manager


def get_relationship_type(relationship):
    return relationship.definition["relation_type"]


def get_properties(node_cls):
    properties = []
    for property_name in node_cls.defined_properties():
        property_ = getattr(node_cls, property_name)
        if property_ is not None:
            property_.name = property_name
            if property_ is not None:
                properties.append(property_)
    return properties


def is_required(property_):
    if isinstance(property_, neomodel.NormalizedProperty):
        return property_.required
    elif isinstance(property_, neomodel.RelationshipTo):
        if (
            property_.manager == neomodel.One
            or property_.manager == neomodel.OneOrMore
        ):
            return True
    return False


def is_many(property_):
    if isinstance(property_, neomodel.ArrayProperty):
        return True
    elif isinstance(property_, neomodel.RelationshipTo):
        if (
            property_.manager == neomodel.ZeroOrMore
            or property_.manager == neomodel.OneOrMore
        ):
            return True
    return False


def is_ordered(property_):
    return hasattr(property_, "order")


def make_doc_from_module(
    module,
    mode: typing.Literal["neo4j", "ogm"] = "neo4j",
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
        properties_or_relationships = get_properties(node_cls)
        for property_or_relationship in properties_or_relationships:
            if isinstance(property_or_relationship, neomodel.RelationshipTo):
                relationship = {
                    "relationship_type": get_relationship_type(
                        property_or_relationship
                    ),
                    "cardinality": get_cardinality(property_or_relationship),
                    "node_class": get_node_class(property_or_relationship),
                    "property_name": property_or_relationship.name,
                }
                relationships.append(relationship)
            else:
                property_ = {
                    "type": type(property_or_relationship),
                    "property_name": property_or_relationship.name,
                    "required": is_required(property_or_relationship),
                }
                properties.append(property_)
        node_spec = {
            "label": label,
            "inherited_labels": inherited_labels,
            "properties": properties,
            "relationships": relationships,
        }
        node_label_to_node_spec[label] = node_spec
        if recursive:
            for relationship in node_spec["relationships"]:
                node_cls = relationship["node_class"]
                if (
                    node_cls.__name__ not in node_label_to_node_spec
                    and not issubclass(node_cls, tuple(exclude))
                ):
                    node_label_to_node_spec = _prepare_node_spec(
                        node_cls,
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
        template = environment.get_template("doc_neo4j.html")
    elif mode == "ogm":
        template = environment.get_template("doc_ogm.html")
    else:
        raise ValueError(f"unrecognized mode {mode}")
    node_label_to_node_spec = {}
    for attr_name in dir(module):
        attr_value = getattr(module, attr_name)
        if (
            isinstance(attr_value, type)
            and issubclass(attr_value, neomodel.StructuredNode)
            and not issubclass(attr_value, tuple(exclude))
            and attr_value.__name__ not in node_label_to_node_spec
        ):
            node_label_to_node_spec |= _prepare_node_spec(
                attr_value, exclude=exclude
            )
    node_specs = list(node_label_to_node_spec.values())
    node_specs.sort(key=lambda node_spec: node_spec["label"])
    s = template.render(node_specs=node_specs)
    if output_file_path:
        with open(output_file_path, "w") as f:
            f.write(s)
    else:
        print(s)
