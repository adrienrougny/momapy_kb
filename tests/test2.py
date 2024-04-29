import neomodel


class A(neomodel.StructuredNode):
    a = neomodel.RelationshipTo("A", "HAS_A")
