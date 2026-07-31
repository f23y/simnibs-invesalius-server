#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a SimNIBS TMS result ``.msh`` into a VTK surface that InVesalius can render directly.
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

_SCALAR_CANDIDATES = ("magnE", "normE", "magnJ")
# Vector fields to take the magnitude of when no scalar field is present.
_VECTOR_CANDIDATES = ("E", "J")

GM_SURFACE_TAG = 1002  # SimNIBS gray-matter surface tag
_TRIANGLE_TYPE = 2     # gmsh element type for triangles


def convert(
    result_msh: str,
    out_path: str | None = None,
    surface_tag: int = GM_SURFACE_TAG,
    progress_cb=None,
):
    """Convert ``result_msh`` to a VTK surface and return (path, vmin, vmax)."""
    from simnibs import mesh_io

    def report(message, percent):
        if progress_cb is not None:
            progress_cb(message, percent)

    size_mb = os.path.getsize(result_msh) / (1024 * 1024)
    report(f"Reading result mesh ({size_mb:.0f} MB)…", 5)
    mesh = mesh_io.read_msh(result_msh)

    report("Interpolating the E-field onto mesh nodes…", 15)
    field_name, node_values = _node_field(mesh)
    vector_name, node_vectors = _node_vector_field(mesh)

    report("Extracting the gray-matter surface…", 95)
    coords, tris, values, vectors = _extract_surface(mesh, node_values, node_vectors, surface_tag)

    if out_path is None:
        out_path = os.path.splitext(result_msh)[0] + "_efield.vtk"
    report(f"Writing {coords.shape[0]} points…", 97)
    _write_legacy_vtk(out_path, coords, tris, field_name, values, vector_name, vectors)

    return out_path, float(values.min()), float(values.max())


def _ncomp(data) -> int:
    ncomp = getattr(data, "nr_comp", None)
    if ncomp:
        return int(ncomp)
    value = np.asarray(data.value)
    return 1 if value.ndim == 1 else int(value.shape[1])


def _node_field(mesh):
    """Return (name, values) with one E-field magnitude per mesh node."""
    # 1) a scalar field already defined on nodes
    for nd in mesh.nodedata:
        if nd.field_name in _SCALAR_CANDIDATES and _ncomp(nd) == 1:
            return nd.field_name, np.asarray(nd.value, dtype=float).reshape(-1)

    # 2) a scalar field defined on elements -> interpolate to nodes
    for ed in mesh.elmdata:
        if ed.field_name in _SCALAR_CANDIDATES and _ncomp(ed) == 1:
            nd = ed.as_nodedata()
            return ed.field_name, np.asarray(nd.value, dtype=float).reshape(-1)

    # 3) a vector field on nodes -> magnitude
    for nd in mesh.nodedata:
        if nd.field_name in _VECTOR_CANDIDATES and _ncomp(nd) == 3:
            v = np.asarray(nd.value, dtype=float).reshape(-1, 3)
            return "magn" + nd.field_name, np.linalg.norm(v, axis=1)

    # 4) a vector field on elements -> to nodes, then magnitude
    for ed in mesh.elmdata:
        if ed.field_name in _VECTOR_CANDIDATES and _ncomp(ed) == 3:
            nd = ed.as_nodedata()
            v = np.asarray(nd.value, dtype=float).reshape(-1, 3)
            return "magn" + ed.field_name, np.linalg.norm(v, axis=1)

    available = [nd.field_name for nd in mesh.nodedata] + [ed.field_name for ed in mesh.elmdata]
    raise ValueError(
        f"No E-field found in mesh (looked for {_SCALAR_CANDIDATES + _VECTOR_CANDIDATES}; "
        f"available fields: {available or 'none'})"
    )


def _node_vector_field(mesh):
    """Return (name, vectors) with one E-field vector per node, or (None, None)."""
    for nd in mesh.nodedata:
        if nd.field_name in _VECTOR_CANDIDATES and _ncomp(nd) == 3:
            return nd.field_name, np.asarray(nd.value, dtype=float).reshape(-1, 3)

    for ed in mesh.elmdata:
        if ed.field_name in _VECTOR_CANDIDATES and _ncomp(ed) == 3:
            nd = ed.as_nodedata()
            return ed.field_name, np.asarray(nd.value, dtype=float).reshape(-1, 3)

    log.info("no vector field in mesh; the surface will carry magnitudes only")
    return None, None


def _extract_surface(mesh, node_values: np.ndarray, node_vectors, surface_tag: int):
    """Pull the triangle surface for ``surface_tag`` and remap to compact node ids."""
    node_number_list = mesh.elm.node_number_list  # (M, 4), 1-based; tris use first 3, 4th == -1
    elm_type = mesh.elm.elm_type
    tag1 = mesh.elm.tag1

    selection = (elm_type == _TRIANGLE_TYPE) & (tag1 == surface_tag)
    if not selection.any():
        # Fall back to every triangle (the outer surface of the mesh).
        log.warning("surface tag %s not found; using all triangles", surface_tag)
        selection = elm_type == _TRIANGLE_TYPE
    if not selection.any():
        raise ValueError("mesh has no triangle elements to build a surface from")

    tri_nodes = node_number_list[selection][:, :3]  # 1-based node ids
    used = np.unique(tri_nodes)                      # sorted, 1-based
    coords = np.asarray(mesh.nodes.node_coord, dtype=float)[used - 1]
    values = node_values[used - 1]
    vectors = node_vectors[used - 1] if node_vectors is not None else None
    tris = np.searchsorted(used, tri_nodes).astype(np.int64)  # 0-based, compact
    return coords, tris, values, vectors


def _write_legacy_vtk(path, coords, tris, field_name, values, vector_name, vectors) -> None:
    n = coords.shape[0]
    ntri = tris.shape[0]
    connectivity = np.hstack([np.full((ntri, 1), 3, dtype=np.int64), tris])
    with open(path, "w", encoding="ascii") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("SimNIBS E-field surface\n")
        fh.write("ASCII\n")
        fh.write("DATASET POLYDATA\n")
        fh.write(f"POINTS {n} float\n")
        np.savetxt(fh, coords, fmt="%.6g")
        fh.write(f"POLYGONS {ntri} {ntri * 4}\n")
        np.savetxt(fh, connectivity, fmt="%d")
        fh.write(f"POINT_DATA {n}\n")
        fh.write(f"SCALARS {field_name} float 1\n")
        fh.write("LOOKUP_TABLE default\n")
        np.savetxt(fh, values.reshape(-1, 1), fmt="%.6g")
        if vectors is not None:
            fh.write(f"VECTORS {vector_name} float\n")
            np.savetxt(fh, vectors, fmt="%.6g")
