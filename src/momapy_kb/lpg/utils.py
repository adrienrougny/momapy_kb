import pathlib
import typing

import jinja2
import pylpg.relationship
import pylpg.types

import fieldz_kb.lpg.graph


def get_node_class(
    descriptor: pylpg.relationship.RelationshipDescriptor,
) -> type:
    return descriptor._relationship_class


def get_relationship_type(
    descriptor: pylpg.relationship.RelationshipDescriptor,
) -> str | None:
    rel_class = descriptor._relationship_class
    return getattr(rel_class, "__type__", None)


def get_properties(
    node_cls: type[fieldz_kb.lpg.graph.BaseNode],
) -> list[dict]:
    primitive_properties = pylpg.types.get_primitive_properties(node_cls)
    relationship_descriptors = pylpg.types.get_relationship_descriptors(node_cls)
    properties = []
    for property_name, property_type in primitive_properties.items():
        properties.append(
            {
                "kind": "property",
                "name": property_name,
                "type": property_type,
            }
        )
    for rel_name, descriptor in relationship_descriptors.items():
        properties.append(
            {
                "kind": "relationship",
                "name": rel_name,
                "descriptor": descriptor,
                "relationship_class": descriptor._relationship_class,
                "relationship_type": getattr(
                    descriptor._relationship_class, "__type__", rel_name
                ),
            }
        )
    return properties


def get_inherited_labels(
    node_cls: type[fieldz_kb.lpg.graph.BaseNode],
) -> list[str]:
    labels = []
    for cls in node_cls.__mro__:
        if cls is object or cls is fieldz_kb.lpg.graph.BaseNode:
            continue
        if issubclass(cls, fieldz_kb.lpg.graph.BaseNode):
            labels.append(cls.__name__)
    return labels


def make_doc_from_module(
    module: object,
    mode: typing.Literal["neo4j", "ogm"] = "neo4j",
    recursive: bool = True,
    output_file_path: str | None = None,
    exclude: list[type] | None = None,
) -> None:

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
        inherited_labels = get_inherited_labels(node_cls)
        label = node_cls.__name__
        if inherited_labels:
            inherited_labels = [
                l for l in inherited_labels if l != label
            ]
        primitive_properties = pylpg.types.get_primitive_properties(node_cls)
        relationship_descriptors = pylpg.types.get_relationship_descriptors(
            node_cls
        )
        properties = []
        relationships = []
        for property_name, property_type in primitive_properties.items():
            properties.append(
                {
                    "type": type(property_type)
                    if not isinstance(property_type, type)
                    else property_type,
                    "property_name": property_name,
                    "required": False,
                }
            )
        for rel_name, descriptor in relationship_descriptors.items():
            rel_class = descriptor._relationship_class
            field_info = getattr(node_cls, f"_field_info_{rel_name}", None)
            relationship = {
                "relationship_type": getattr(rel_class, "__type__", rel_name),
                "node_class": node_cls,
                "property_name": rel_name,
            }
            relationships.append(relationship)
        node_spec = {
            "label": label,
            "inherited_labels": inherited_labels,
            "properties": properties,
            "relationships": relationships,
        }
        node_label_to_node_spec[label] = node_spec
        if recursive:
            for relationship in node_spec["relationships"]:
                rel_node_cls = relationship.get("node_class")
                if rel_node_cls is not None and (
                    rel_node_cls.__name__ not in node_label_to_node_spec
                    and not issubclass(rel_node_cls, tuple(exclude))
                ):
                    node_label_to_node_spec = _prepare_node_spec(
                        rel_node_cls,
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
            and issubclass(attr_value, fieldz_kb.lpg.graph.BaseNode)
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
