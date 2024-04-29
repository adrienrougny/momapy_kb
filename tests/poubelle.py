import neomodel


class N(neomodel.StructuredNode):
    s = neomodel.StringProperty()


class M(neomodel.StructuredNode):
    n = neomodel.RelationshipTo(N, "HAS_N", neomodel.OneOrMore)


for prop in M.defined_properties():
    prop = getattr(M, prop)
    print(dir(prop))
    print(prop.manager)
