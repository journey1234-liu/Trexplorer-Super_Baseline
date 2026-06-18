import argparse
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

sys.setrecursionlimit(50000)

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import networkx as nx
    import numpy as np
    import SimpleITK as sitk
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from src.trxsuper.datasets.dataset_preprocessing import (
        add_points_type_list,
        convert_networkx_to_bigtree,
        densify_networkx,
        extract_full_trajectores,
    )
except ImportError as exc:
    raise SystemExit(
        "CoW preparation requires numpy, networkx, bigtree, SimpleITK, and vtk. "
        "Install project requirements first: pip install -r requirements.txt"
    ) from exc


DATASET_NAME = "Dataset033_BinTopCoW24_CT"
TREE_REQUIRED_KEYS = {
    "branches",
    "networkx",
    "all_ids",
    "bifur_ids",
    "endpts_ids",
    "interm_ids",
    "root_id",
    "num_branches",
    "num_points",
    "branch_ids",
    "trajectories",
    "traj_stats",
    "total_points",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare CoW CT data in Trexplorer dataset format.")
    parser.add_argument("--src", default="/home/lyyu/data/202605_topovst_rev")
    parser.add_argument("--dst", default="data/CoW")
    parser.add_argument("--radius-array", default="mis_radius")
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--limit",
        default=None,
        help="Optional smoke-test limits as N or train,val,test, e.g. 2,1,1.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files/symlinks under the destination.")
    return parser.parse_args()


def parse_limit(limit):
    if not limit:
        return None
    parts = [int(x.strip()) for x in limit.split(",")]
    if len(parts) == 1:
        return {"train": parts[0], "val": parts[0], "test": parts[0]}
    if len(parts) == 3:
        return {"train": parts[0], "val": parts[1], "test": parts[2]}
    raise ValueError("--limit must be N or train,val,test")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def replace_path(dst, overwrite):
    if not dst.exists() and not dst.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(
            f"{dst} already exists. Re-run with --overwrite to replace it.")
    if dst.is_dir() and not dst.is_symlink():
        raise IsADirectoryError(f"Refusing to replace directory: {dst}")
    dst.unlink()


def link_or_copy(src, dst, mode, overwrite):
    replace_path(dst, overwrite)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def discover_cases(raw_dir):
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"
    images_ts = raw_dir / "imagesTs"
    labels_ts = raw_dir / "labelsTs"
    train_ids = sorted(
        p.name.removesuffix("_0000.nii.gz")
        for p in images_tr.glob("*_0000.nii.gz"))
    test_ids = sorted(
        p.name.removesuffix("_0000.nii.gz")
        for p in images_ts.glob("*_0000.nii.gz"))
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "images_tr": images_tr,
        "labels_tr": labels_tr,
        "images_ts": images_ts,
        "labels_ts": labels_ts,
    }


def split_train_val(train_ids, val_count, seed):
    if val_count <= 0 or val_count >= len(train_ids):
        raise ValueError(
            f"val-count must be in [1, {len(train_ids) - 1}], got {val_count}")
    shuffled = sorted(train_ids)
    random.Random(seed).shuffle(shuffled)
    val_ids = set(shuffled[:val_count])
    return sorted([x for x in train_ids if x not in val_ids]), sorted(val_ids)


def read_image_size(path):
    image = sitk.ReadImage(str(path))
    return tuple(int(x) for x in image.GetSize())


def read_roi_location(path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("Location (Voxels):"):
            values = line.split(":", 1)[1].strip().split()
            if len(values) != 3:
                raise ValueError(f"Invalid ROI location line in {path}: {line}")
            return np.array([float(v) for v in values], dtype=float)
    return None


def positions_fit(points, image_size):
    size = np.asarray(image_size, dtype=float)
    lower_ok = np.all(points >= 0)
    upper_ok = np.all(points < size)
    return bool(lower_ok and upper_ok)


def physical_to_indices(points, image):
    return np.array([
        image.TransformPhysicalPointToContinuousIndex(
            tuple(float(coord) for coord in point))
        for point in points
    ], dtype=float)


def choose_coordinate_mode(points, image, roi_location):
    image_size = image.GetSize()
    if positions_fit(points, image_size):
        return "direct", points

    physical = physical_to_indices(points, image)
    if positions_fit(physical, image_size):
        return "physical_lps_to_index", physical

    ras_points = points * np.array([-1.0, -1.0, 1.0])
    ras_physical = physical_to_indices(ras_points, image)
    if positions_fit(ras_physical, image_size):
        return "physical_ras_to_lps_index", ras_physical

    if roi_location is not None:
        shifted = points - roi_location.reshape(1, 3)
        if positions_fit(shifted, image_size):
            return "roi_subtracted", shifted
    raise ValueError(
        "VTP points do not fit image bounds directly or after ROI subtraction. "
        f"point_min={points.min(axis=0).tolist()}, "
        f"point_max={points.max(axis=0).tolist()}, "
        f"image_size={list(image_size)}, "
        f"roi_location={None if roi_location is None else roi_location.tolist()}")


def read_cow_vtp(vtp_path, radius_array):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    model = reader.GetOutput()
    points_vtk = model.GetPoints()
    if points_vtk is None:
        raise ValueError(f"No points found in {vtp_path}")

    points = np.array([
        points_vtk.GetPoint(i)
        for i in range(points_vtk.GetNumberOfPoints())
    ], dtype=float)

    cell_data = model.GetCellData()
    radii_vtk = cell_data.GetArray(radius_array)
    if radii_vtk is None:
        arrays = [
            cell_data.GetArrayName(i)
            for i in range(cell_data.GetNumberOfArrays())
        ]
        raise ValueError(
            f"Radius array '{radius_array}' not found in {vtp_path}. "
            f"Available CellData arrays: {arrays}")
    radii = vtk_to_numpy(radii_vtk).astype(float).reshape(-1)

    lines = model.GetLines()
    lines.InitTraversal()
    line_ids = []
    for _ in range(lines.GetNumberOfCells()):
        cell = vtk.vtkIdList()
        lines.GetNextCell(cell)
        line_ids.append([cell.GetId(k) for k in range(cell.GetNumberOfIds())])

    if len(radii) != len(line_ids):
        raise ValueError(
            f"{vtp_path} has {len(line_ids)} line cells but "
            f"{len(radii)} radii in {radius_array}")

    return points, line_ids, radii


def add_edge_with_radius(graph, u, v, radius):
    if graph.has_edge(u, v):
        graph.edges[u, v]["radius_values"].append(float(radius))
        graph.edges[u, v]["radius"] = float(np.mean(
            graph.edges[u, v]["radius_values"]))
    else:
        graph.add_edge(u, v, radius=float(radius), radius_values=[float(radius)])


def make_node_radii(graph):
    node_radii = {}
    for node in graph.nodes:
        vals = []
        for _, _, data in graph.edges(node, data=True):
            vals.append(float(data["radius"]))
        node_radii[node] = float(np.mean(vals)) if vals else 1.0
    nx.set_node_attributes(graph, node_radii, "radius")


def cow_vtp_to_directed_graphs(vtp_path, image_path, roi_path, radius_array):
    points, line_ids, radii = read_cow_vtp(vtp_path, radius_array)
    image = sitk.ReadImage(str(image_path))
    image_size = image.GetSize()
    roi_location = read_roi_location(roi_path)
    coord_mode, points = choose_coordinate_mode(points, image, roi_location)

    undirected = nx.Graph()
    undirected.add_nodes_from(range(points.shape[0]))
    nx.set_node_attributes(
        undirected,
        {idx: points[idx].astype(float) for idx in range(points.shape[0])},
        "position")

    for line, radius in zip(line_ids, radii):
        for idx in range(len(line) - 1):
            add_edge_with_radius(undirected, line[idx], line[idx + 1], radius)

    make_node_radii(undirected)
    components = list(nx.connected_components(undirected))
    directed_graphs = []
    for component in components:
        sub = undirected.subgraph(component).copy()
        endpoints = [node for node in sub.nodes if sub.degree[node] == 1]
        if not endpoints:
            raise ValueError(f"No endpoint root candidate in {vtp_path}")
        endpoints.sort(
            key=lambda node: sub.nodes[node]["position"][-1], reverse=True)
        root = endpoints[0]

        directed = nx.DiGraph()
        for node, data in sub.nodes(data=True):
            directed.add_node(node, **data)
        for u, v in nx.bfs_edges(sub, source=root):
            directed.add_edge(u, v, radius=float(sub.edges[u, v]["radius"]))
        directed.remove_nodes_from(list(nx.isolates(directed)))
        if len(directed) and not nx.is_tree(directed.to_undirected()):
            raise ValueError(f"Directed component is not a tree in {vtp_path}")
        directed_graphs.append(directed)

    return directed_graphs, coord_mode, image_size


def graph_to_annotation(graphs):
    dense_graphs = [densify_networkx(graph.copy()) for graph in graphs]
    trees = [convert_networkx_to_bigtree(graph) for graph in dense_graphs]
    annot = {
        "branches": trees,
        "networkx": dense_graphs,
        "trajectories": [],
        "traj_stats": [],
    }
    annot = add_points_type_list(annot)
    for tree in trees:
        trajectories, stats = extract_full_trajectores(tree)
        annot["trajectories"].append(trajectories)
        annot["traj_stats"].append(stats)
    annot["total_points"] = [
        len(list(tree.descendants)) + 1
        for tree in trees
    ]

    missing = sorted(TREE_REQUIRED_KEYS - set(annot))
    if missing:
        raise ValueError(f"Converted annotation missing keys: {missing}")
    return annot


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_annotation(case_id, src_root, dst_centerlines, radius_array,
                     image_path, overwrite):
    vtp_path = src_root / "cow_graphs" / f"{case_id}.vtp"
    roi_path = src_root / "cow_roi_loc" / f"{case_id}.txt"
    if not vtp_path.exists():
        raise FileNotFoundError(vtp_path)

    graphs, coord_mode, image_size = cow_vtp_to_directed_graphs(
        vtp_path, image_path, roi_path, radius_array)
    annot = graph_to_annotation(graphs)
    out_path = dst_centerlines / f"{case_id}.pickle"
    replace_path(out_path, overwrite)
    if not out_path.exists():
        with out_path.open("wb") as handle:
            pickle.dump(annot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    graph_stats = {
        "num_components": len(graphs),
        "total_points": annot["total_points"],
        "traj_stats": annot["traj_stats"],
    }
    return out_path, coord_mode, image_size, to_jsonable(graph_stats)


def make_val_sub_vol_pickle(dst_root, val_ids, samples_per_case=200):
    from src.trxsuper.datasets.utils.generate_val_sub_vol_file import (
        create_sub_vol_eval_dataset,
    )

    samples = []
    per_case = max(1, samples_per_case // max(1, len(val_ids)))
    for case_id in val_ids:
        annot_path = dst_root / "annots_val_sub_vol" / f"{case_id}.pickle"
        samples.extend(create_sub_vol_eval_dataset(
            str(annot_path),
            bifur_prob=0.01,
            end_prob=0.01,
            root_prob=0.01,
            seq_len=10,
            num_samples_per_annot=per_case,
            seq_range_half=False))
    with (dst_root / "annots_val_sub_vol.pickle").open("wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return len(samples)


def prepare_dirs(dst_root):
    dirs = [
        "centerlines",
        "annots_train",
        "annots_val",
        "annots_test",
        "annots_val_sub_vol",
        "images_train",
        "images_val",
        "images_test",
        "images_val_sub_vol",
        "masks_train",
        "masks_val",
        "masks_test",
        "masks_val_sub_vol",
    ]
    for name in dirs:
        ensure_dir(dst_root / name)


def apply_limits(train_ids, val_ids, test_ids, limits):
    if not limits:
        return train_ids, val_ids, test_ids
    return (
        train_ids[:limits["train"]],
        val_ids[:limits["val"]],
        test_ids[:limits["test"]],
    )


def source_paths(raw_dir, case_id, split):
    if split == "test":
        return (
            raw_dir / "imagesTs" / f"{case_id}_0000.nii.gz",
            raw_dir / "labelsTs" / f"{case_id}.nii.gz",
        )
    return (
        raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz",
        raw_dir / "labelsTr" / f"{case_id}.nii.gz",
    )


def install_case(case_id, split, src_root, raw_dir, dst_root, args):
    image_src, mask_src = source_paths(raw_dir, case_id, split)
    if not image_src.exists():
        raise FileNotFoundError(image_src)
    if not mask_src.exists():
        raise FileNotFoundError(mask_src)

    annot_path, coord_mode, image_size, graph_stats = write_annotation(
        case_id,
        src_root,
        dst_root / "centerlines",
        args.radius_array,
        image_src,
        args.overwrite)

    split_suffix = {"train": "train", "val": "val", "test": "test"}[split]
    link_or_copy(
        annot_path,
        dst_root / f"annots_{split_suffix}" / f"{case_id}.pickle",
        args.link_mode,
        args.overwrite)
    link_or_copy(
        image_src,
        dst_root / f"images_{split_suffix}" / f"{case_id}.nii.gz",
        args.link_mode,
        args.overwrite)
    link_or_copy(
        mask_src,
        dst_root / f"masks_{split_suffix}" / f"{case_id}.nii.gz",
        args.link_mode,
        args.overwrite)

    if split == "val":
        link_or_copy(
            annot_path,
            dst_root / "annots_val_sub_vol" / f"{case_id}.pickle",
            args.link_mode,
            args.overwrite)
        link_or_copy(
            image_src,
            dst_root / "images_val_sub_vol" / f"{case_id}.nii.gz",
            args.link_mode,
            args.overwrite)
        link_or_copy(
            mask_src,
            dst_root / "masks_val_sub_vol" / f"{case_id}.nii.gz",
            args.link_mode,
            args.overwrite)

    return {
        "case_id": case_id,
        "split": split,
        "image": str(image_src),
        "mask": str(mask_src),
        "graph": str(src_root / "cow_graphs" / f"{case_id}.vtp"),
        "annotation": str(annot_path),
        "coordinate_mode": coord_mode,
        "image_size": list(image_size),
        "graph_stats": graph_stats,
    }


def main():
    args = parse_args()
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst)
    raw_dir = src_root / "nnUNet_raw" / DATASET_NAME
    limits = parse_limit(args.limit)

    prepare_dirs(dst_root)
    cases = discover_cases(raw_dir)
    train_ids, val_ids = split_train_val(
        cases["train_ids"], args.val_count, args.seed)
    test_ids = cases["test_ids"]
    train_ids, val_ids, test_ids = apply_limits(
        train_ids, val_ids, test_ids, limits)

    manifest = {
        "source_root": str(src_root),
        "raw_dataset": str(raw_dir),
        "destination": str(dst_root),
        "radius_array": args.radius_array,
        "seed": args.seed,
        "val_count": args.val_count,
        "link_mode": args.link_mode,
        "limits": limits,
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "cases": [],
        "skipped": [],
    }

    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for case_id in ids:
            print(f"Preparing {split}: {case_id}")
            try:
                manifest["cases"].append(install_case(
                    case_id, split, src_root, raw_dir, dst_root, args))
            except Exception as exc:
                manifest["skipped"].append({
                    "case_id": case_id,
                    "split": split,
                    "reason": repr(exc),
                })
                raise

    manifest["annots_val_sub_vol_count"] = make_val_sub_vol_pickle(
        dst_root, val_ids)

    manifest_path = dst_root / "cow_prepare_manifest.json"
    replace_path(manifest_path, args.overwrite)
    with manifest_path.open("w") as handle:
        json.dump(to_jsonable(manifest), handle, indent=2)

    print(
        "Prepared CoW dataset: "
        f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test. "
        f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
