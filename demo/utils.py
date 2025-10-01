import functools
import operator
import collections.abc

import IPython.display

import momapy.rendering.svg_native
import momapy.geometry
import momapy.builder
import momapy.core
import momapy.drawing
import momapy.meta.nodes
import momapy.positioning
import momapy.coloring
import momapy.styling
import momapy.sbgn.io.sbgnml
import momapy.celldesigner.io.celldesigner
import momapy.io


#
# def display(obj, markers=None, xsep=20.0, ysep=20.0, scale=1.0):
#     if markers is None:
#         markers = []
#     if isinstance(obj, str):
#         layout_element = momapy.io.read(obj, return_type="layout").obj
#     elif isinstance(obj, momapy.core.Map):
#         layout_element = obj.layout
#     elif isinstance(obj, momapy.core.LayoutElement):
#         layout_element = obj
#     else:
#         raise ValueError(f"unsupported type {type(obj)}")
#     bbox = layout_element.bbox()
#     if (
#         layout_element.group_transform is not None
#         and layout_element.group_transform != momapy.drawing.NoneValue
#     ):
#         total_transformation = functools.reduce(
#             operator.mul, layout_element.group_transform
#         )
#         bbox = momapy.positioning.fit(
#             [
#                 bbox.north_west().transformed(total_transformation),
#                 bbox.north_east().transformed(total_transformation),
#                 bbox.south_west().transformed(total_transformation),
#                 bbox.south_east().transformed(total_transformation),
#             ]
#         )
#     min_x = bbox.x - bbox.width / 2 - xsep
#     min_y = bbox.y - bbox.height / 2 - ysep
#     max_x = bbox.x + bbox.width / 2 - min_x + xsep
#     max_y = bbox.y + bbox.height / 2 - min_y + ysep
#     layout_element = momapy.builder.builder_from_object(layout_element)
#     if layout_element.group_transform is None:
#         layout_element.group_transform = momapy.core.TupleBuilder()
#     translation = momapy.geometry.Translation(-min_x, -min_y)
#     layout_element.group_transform.insert(0, translation)
#     cp_builder_cls = momapy.builder.get_or_make_builder_cls(
#         momapy.meta.nodes.CrossPoint
#     )
#     if isinstance(markers, momapy.geometry.Point):
#         markers = [markers]
#     for marker in markers:
#         position = marker
#         cp = cp_builder_cls(
#             width=12.0,
#             height=12.0,
#             stroke_width=1.5,
#             stroke=momapy.coloring.red,
#             position=position,
#         )
#         layout_element.layout_elements.append(cp)
#     width = max_x
#     height = max_y
#     layout_element = momapy.builder.object_from_builder(layout_element)
#     renderer = momapy.rendering.svg_native.SVGNativeRenderer(
#         svg=momapy.rendering.svg_native.SVGElement(
#             name="svg",
#             attributes={
#                 "xmlns": "http://www.w3.org/2000/svg",
#                 "viewBox": f"0 0 {width} {height}",
#                 "width": width * scale,
#                 "height": height * scale,
#             },
#         )
#     )
#     renderer.begin_session()
#     renderer.render_layout_element(layout_element)
#     renderer.end_session()
#     svg_string = str(renderer.svg)
#     IPython.display.display(IPython.display.SVG(data=svg_string))
#
#
def display(obj, markers=None, xsep=20.0, ysep=20.0, scale=1.0, style_sheet=None):
    if markers is None:
        markers = []
    if isinstance(style_sheet, str):
        style_sheet = momapy.styling.StyleSheet.from_file(style_sheet)
    if not isinstance(obj, collections.abc.Sequence) or isinstance(
        obj, (str, bytes, bytearray)
    ):
        obj = [obj]
    layout_elements = []
    for element in obj:
        if isinstance(element, str):
            layout_element = momapy.io.read(element, return_type="layout").obj
        elif isinstance(element, momapy.core.Map):
            layout_element = element.layout
        elif isinstance(element, momapy.core.LayoutElement):
            layout_element = element
        else:
            raise ValueError(f"unsupported type {type(element)}")
        layout_elements.append(layout_element)
    bboxes = []
    if style_sheet is not None:
        layout_elements = [
            momapy.styling.apply_style_sheet(layout_element, style_sheet)
            for layout_element in layout_elements
        ]
    for layout_element in layout_elements:
        bbox = layout_element.bbox()
        if (
            layout_element.group_transform is not None
            and layout_element.group_transform != momapy.drawing.NoneValue
        ):
            total_transformation = functools.reduce(
                operator.mul, layout_element.group_transform
            )
            bbox = momapy.positioning.fit(
                [
                    bbox.north_west().transformed(total_transformation),
                    bbox.north_east().transformed(total_transformation),
                    bbox.south_west().transformed(total_transformation),
                    bbox.south_east().transformed(total_transformation),
                ]
            )
        bboxes.append(bbox)
    bbox = momapy.positioning.fit(bboxes)
    min_x = bbox.x - bbox.width / 2 - xsep
    min_y = bbox.y - bbox.height / 2 - ysep
    max_x = bbox.x + bbox.width / 2 - min_x + xsep
    max_y = bbox.y + bbox.height / 2 - min_y + ysep
    translation = momapy.geometry.Translation(-min_x, -min_y)
    final_layout_elements = []
    for layout_element in layout_elements:
        layout_element_builder = momapy.builder.builder_from_object(layout_element)
        if layout_element.group_transform is None:
            layout_element_builder.group_transform = momapy.core.TupleBuilder()
        layout_element_builder.group_transform.insert(0, translation)
        final_layout_elements.append(
            momapy.builder.object_from_builder(layout_element_builder)
        )
    cp_builder_cls = momapy.builder.get_or_make_builder_cls(
        momapy.meta.nodes.CrossPoint
    )
    if isinstance(markers, momapy.geometry.Point):
        markers = [markers]
    for marker in markers:
        position = marker
        cp_builder = cp_builder_cls(
            width=12.0,
            height=12.0,
            stroke_width=1.5,
            stroke=momapy.coloring.red,
            position=position,
        )
        final_layout_elements.append(momapy.builder.object_from_builder(cp_builder))
    width = max_x
    height = max_y
    renderer = momapy.rendering.svg_native.SVGNativeRenderer(
        svg=momapy.rendering.svg_native.SVGElement(
            name="svg",
            attributes={
                "xmlns": "http://www.w3.org/2000/svg",
                "viewBox": f"0 0 {width} {height}",
                "width": width * scale,
                "height": height * scale,
            },
        )
    )
    renderer.begin_session()
    for layout_element in final_layout_elements:
        renderer.render_layout_element(layout_element)
    renderer.end_session()
    svg_string = str(renderer.svg)
    IPython.display.display(IPython.display.SVG(data=svg_string))
