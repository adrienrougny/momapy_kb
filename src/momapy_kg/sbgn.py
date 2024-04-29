import neomodel
import momapy.sbgn.pd

import momapy_kg.builder


# class AnnotationNode(
#     momapy_neo4j.builder.get_or_make_node_cls(momapy.core.ModelElement)
# ):
#     _cls_to_build = momapy.sbgn.core.Annotation
#     qualifier = neomodel.StringProperty(required=True)
#     resource = neomodel.StringProperty(required=True)
#
#     @classmethod
#     def save_from_object(cls, obj, object_to_node=None):
#         if object_to_node is not None:
#             node = object_to_node.get(id(obj))
#             if node is not None:
#                 return node
#         else:
#             object_to_node = {}
#         qualifier = f"{type(obj.qualifier).__name__}.{obj.qualifier.value}"
#         resource = obj.resource
#         node = cls(qualifier=qualifier, resource=resource)
#         node.save()
#         object_to_node[id(obj)] = node
#         return node
#
#     def build(self):
#         qualifier_ns, qualifier_name = self.qualifier.split(".")
#         qualifier_enum = getattr(momapy.sbgn.core, qualifier_ns)
#         qualifier = qualifier_enum(qualifier_name)
#         return self._cls_to_build(
#             id=self.uid, qualifier=qualifier, resource=self.resource
#         )


# momapy_neo4j.builder.register_node_class(AnnotationNode)


class NoneValueNode(momapy_neo4j.builder.MomapyNode):
    _cls_to_build = momapy.drawing.NoneValueType

    @classmethod
    def save_from_object(cls, obj, object_to_node=None):
        if object_to_node is not None:
            node = object_to_node.get(id(obj))
            if node is not None:
                return node
        else:
            object_to_node = {}
        node = cls()
        node.save()
        return node


momapy_neo4j.builder.register_node_class(NoneValueNode)
