import os
import sys
import pickle
import argparse
from typing import Dict
from glob import glob
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import networkx as nx
from tqdm import tqdm
from bigtree import Node, find_name, levelordergroup_iter, find_path, find
import SimpleITK as sitk


def get_affine(data):
    """
    Get or construct the affine matrix of the image, it can be used to correct
    spacing, orientation or execute spatial transforms.
    Args:
    data: an ITK image object loaded from an image file or props dictionary.

    """
    if isinstance(data, Dict):
        direction = np.array(data["sitk_stuff"]["direction"]).reshape(3, 3)
        spacing = np.array(data["sitk_stuff"]["spacing"])
        origin = np.array(data["sitk_stuff"]["origin"])

    else:
        direction = np.array(data.GetDirection()).reshape(3, 3)
        spacing = np.asarray(data.GetSpacing())
        origin = np.asarray(data.GetOrigin())

    sr = min(max(direction.shape[0], 1), 3)
    affine: np.ndarray = np.eye(sr + 1)
    affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
    affine[:sr, -1] = origin[:sr]
    return affine, spacing


def transform_points(
    points: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:

    points = np.concatenate(
        [points, np.ones([points.shape[0], 1])], axis=1).T
    return (affine @ points).T[:, :-1]


def fix_asoca_points(points: np.ndarray, affine: np.ndarray):

    cl_affine = affine.copy()
    cl_affine[0, 3] = -1 * cl_affine[0, 3]  # O_x to -O_x
    cl_affine[1, 3] = -1 * cl_affine[1, 3]  # O_y to -O_y

    trans_matrix = affine @ np.linalg.inv(cl_affine)
    points = transform_points(points, trans_matrix)

    return points


def read_centerline_model(file_name: str):
    """
    Read out centerline model in VTK or VTP format.
    """

    if not file_name.endswith((".vtk", ".vtp")):
        raise FileNotFoundError("Only .vtk and .vtp files supported.")

    if file_name.endswith(".vtk"):
        reader = vtk.vtkPolyDataReader()
        arr_name = "Radius"  # Slicer-VMTK generated model
    else:
        reader = vtk.vtkXMLPolyDataReader()
        arr_name = "MaximumInscribedSphereRadius"  # ASOCA
    reader.SetFileName(file_name)
    reader.Update()
    model: vtk.vtkPolyData = reader.GetOutput()

    # Extract vtkPoints and convert to numpy ndarray
    points: vtk.vtkPoints = model.GetPoints()
    points_arr = np.array([
        points.GetPoint(i)
        for i in range(points.GetNumberOfPoints())]).reshape(-1, 3)
    radii = vtk_to_numpy(
        model.GetPointData().GetArray(arr_name)).reshape(-1, 1)

    # Extract lines and separate them into cells(line segments)
    lines = model.GetLines()
    lines.InitTraversal()
    num_cells = lines.GetNumberOfCells()
    lines_list = []
    for i in range(num_cells):
        cell = vtk.vtkIdList()
        lines.GetNextCell(cell)
        point_ids = [cell.GetId(k) for k in range(cell.GetNumberOfIds())]
        lines_list.append(point_ids)

    return points_arr, lines_list, radii


def convert_networkx_to_bigtree_wrapper(args):

    id, graph = args
    return id, convert_networkx_to_bigtree(graph)


@lru_cache(maxsize=None)
def convert_networkx_to_bigtree(graph: nx.DiGraph):

    # Get all node positions from the graph
    node_positions = nx.get_node_attributes(graph, 'position')
    # Get all edge radii from the graph
    edge_radii = nx.get_edge_attributes(graph, 'radius')

    # Get the root point
    in_degrees = dict(graph.in_degree())
    root_node_id = [node for node,
                    in_degree in in_degrees.items() if in_degree == 0]
    assert len(root_node_id) == 1, "Graph has multiple root nodes!"
    root_node_id = root_node_id[0]

    out_edges = list(graph.out_edges(root_node_id))
    # node radius is the maximum radius of the edges that start from that node
    node_radius = max([edge_radii[edge] for edge in out_edges])

    # Initialize the branch and point id for the root node
    branch_id = 0
    point_id = 0
    root_node_id_bt = str(branch_id) + "-" + str(point_id)
    node_pos = np.array(node_positions[root_node_id]).astype(float).tolist()
    root_node_bt = Node(str(root_node_id_bt),
                        position=node_pos, radius=node_radius, label=1)
    branch_id_count = 0

    # create a dictionary to map the node id from networkx to bigtree
    nx_to_bt_node_id_dict = {root_node_id: root_node_id_bt}

    # Traverse the graph level by level using BFS and populate the bigtree nodes
    for i, level in enumerate(nx.bfs_layers(graph, root_node_id)):
        if i == 0:
            continue  # skip the root node
        for node in level:
            label = 1  # Intermediate
            parent = list(graph.predecessors(node))
            assert len(parent) <= 1, "Graph has multiple parents!"
            parent = parent[0]
            node_radius = edge_radii[(parent, node)]
            parent_id_bt = nx_to_bt_node_id_dict[parent]
            branch_id = parent_id_bt.split('-')[0]
            point_id = parent_id_bt.split('-')[1]
            is_bifur = True if len(list(graph.out_edges(node))) > 1 else False
            is_end = True if len(list(graph.out_edges(node))) == 0 else False
            if is_bifur:
                label = 2  # bifurcation
            if is_end:
                label = 0  # end
            bifurcation = True if len(
                list(graph.out_edges(parent))) > 1 else False
            # if parent is a bifurcation, then create a new branch id by incrementing the branch id count and set the point id to 0
            if bifurcation:
                branch_id_count += 1
                branch_id = str(branch_id_count)
                point_id = '0'
            else:
                # if parent is not a bifurcation, then increment the point id
                point_id = str(int(point_id) + 1)
            node_id_bt = branch_id + "-" + point_id
            # Create a new node in bigtree
            node_pos = np.array(node_positions[node]).astype(float).tolist()
            Node(str(node_id_bt), position=node_pos, radius=node_radius,
                 parent=find_name(root_node_bt, parent_id_bt), label=label)
            # Add the node id to the mapping dictionary
            nx_to_bt_node_id_dict[node] = node_id_bt

    return root_node_bt


def convert_model_to_networkx(model_file: str):
    """
    Given an input vtk file location, read the vtk file and convert centerline
    to networkx DiGraph.
    """

    positions, lines, radii = read_centerline_model(file_name=model_file)
    node_pos_dict = dict(enumerate(positions))
    # Create node names
    node_names = np.arange(positions.shape[0])
    node_radii_dict = dict(zip(node_names, radii))
    # Get edges from lines list
    edges = []
    for line in lines:
        line_edges = [(line[i], line[i + 1]) for i in range(len(line) - 1)]
        edges.extend(line_edges)
    edges = list(set(edges))

    # First we get an undirected Graph.
    G0 = nx.Graph()
    G0.add_nodes_from(node_names)
    G0.add_edges_from(edges)
    nx.set_node_attributes(G0, node_radii_dict, "radius")
    roots = [n for n in G0.nodes if G0.degree[n] == 1]

    conns = list(nx.connected_components(G0))
    print(f"{len(conns)} connected components in original centerline tree.")

    # Traverse the undirected graph to get a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    nx.set_node_attributes(G, node_pos_dict, 'position')
    nx.set_node_attributes(G, node_radii_dict, 'radius')
    for conn in conns:
        conn_roots = [r for r in roots if r in conn]
        conn_roots.sort(key=lambda r: G0.nodes[r]["radius"], reverse=True)
        forward_edges = list(nx.bfs_edges(G0, source=conn_roots[0]))
        forward_edge_radii = list(map(
            lambda t: np.mean([radii[t[0]], radii[t[1]]]), forward_edges))
        sub_edge_radii_dict = dict(zip(forward_edges, forward_edge_radii))

        G.add_edges_from(forward_edges)
        nx.set_edge_attributes(G, sub_edge_radii_dict, 'radius')

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    # Extract connected components
    connected_components = list(nx.weakly_connected_components(G))

    # Create subgraphs from connected components
    subgraphs = [G.subgraph(component).copy()
                 for component in connected_components]

    return subgraphs


def add_points_type_list(trees_with_metadata):
    """
    Add list of root point, bifurcation points, end points, and intermediate points
    """
    trees_with_metadata['all_ids'] = []
    trees_with_metadata['bifur_ids'] = []
    trees_with_metadata['endpts_ids'] = []
    trees_with_metadata['interm_ids'] = []
    trees_with_metadata['root_id'] = []
    trees_with_metadata['num_branches'] = []
    trees_with_metadata['num_points'] = []
    trees_with_metadata['branch_ids'] = []
    trees = trees_with_metadata['branches']
    for tree in trees:
        bifur_ids = []
        endpts_ids = []
        interm_ids = []
        all_ids = []
        for level in levelordergroup_iter(tree):
            for node in level:
                all_ids.append(node.node_name)
                if len(node.children) > 1:
                    bifur_ids.append(node.node_name)
                elif len(node.children) == 0:
                    endpts_ids.append(node.node_name)
                elif node.parent is None:
                    root_id = node.node_name
                else:
                    interm_ids.append(node.node_name)
        trees_with_metadata['all_ids'].append(all_ids)
        trees_with_metadata['bifur_ids'].append(bifur_ids)
        trees_with_metadata['endpts_ids'].append(endpts_ids)
        trees_with_metadata['interm_ids'].append(interm_ids)
        trees_with_metadata['root_id'].append(root_id)
        branch_ids = sorted(list(set([x.split('-')[0] for x in all_ids])))
        trees_with_metadata['branch_ids'].append(branch_ids)
        trees_with_metadata['num_branches'].append(len(branch_ids))

        # Count points for each branch
        branch_counts = {}
        for node in all_ids:
            branch_id = node.split('-')[0]
            if branch_id in branch_counts:
                branch_counts[branch_id] += 1
            else:
                branch_counts[branch_id] = 1
        # Create a list with counts for each corresponding branch in branch_ids
        num_points = [branch_counts[branch_id] for branch_id in branch_ids]
        trees_with_metadata['num_points'].append(num_points)

    return trees_with_metadata


def densify_networkx(graph: nx.DiGraph):

    # check if graph is tree
    assert nx.is_tree(graph), "Graph is not a tree!"

    # Get all node positions from the graph
    node_positions = nx.get_node_attributes(graph, 'position')
    node_radii = nx.get_node_attributes(graph, "radius")
    # Get all edges from the graph
    edges = list(graph.edges)
    # Get all edge radii from the graph
    edge_radii = nx.get_edge_attributes(graph, 'radius')

    # Go through each edge, and add nodes and edges in between
    for edge in edges:
        start_node = edge[0]
        end_node = edge[1]
        start_node_position = node_positions[start_node]
        start_node_radius = node_radii[start_node]
        end_node_position = node_positions[end_node]
        end_node_radius = node_radii[end_node]
        edge_radius = edge_radii[edge]

        # Calculate the inbetween node positions
        in_between_positions = bresenham3D(
            np.round(start_node_position), np.round(end_node_position))
        # computed by convex combination of end points
        mid_points = np.linspace(0, 1, 2 + len(in_between_positions))[1:-1]
        in_between_positions_cont = [
            (1 - x) * start_node_position + x * end_node_position for x in mid_points]
        in_between_radii_cont = [
            (1 - x) * start_node_radius + x * end_node_radius for x in mid_points]

        # Remove the original edge
        graph.remove_edge(start_node, end_node)

        # Add the nodes in between
        for position, r in zip(in_between_positions_cont, in_between_radii_cont):
            # Add the new node to the graph
            new_node = max(graph.nodes) + 1
            graph.add_node(new_node, position=position, radius=r)
            # Add the new edge to the graph
            graph.add_edge(start_node, new_node, radius=edge_radius)
            start_node = new_node

        # connect last inbetween to the end node
        graph.add_edge(start_node, end_node, radius=edge_radius)

    # check if graph is tree
    assert nx.is_tree(graph), "Graph is not a tree!"
    return graph


# Reference: https://github.com/balzer82/3D-OccupancyGrid-Python/blob/master/Timings.py
def bresenham3D(startPoint, endPoint):

    path = []

    startPoint = [int(startPoint[0]), int(startPoint[1]), int(startPoint[2])]
    endPoint = [int(endPoint[0]), int(endPoint[1]), int(endPoint[2])]

    steepXY = (np.abs(endPoint[1] - startPoint[1]) >
               np.abs(endPoint[0] - startPoint[0]))
    if (steepXY):
        startPoint[0], startPoint[1] = startPoint[1], startPoint[0]
        endPoint[0], endPoint[1] = endPoint[1], endPoint[0]

    steepXZ = (np.abs(endPoint[2] - startPoint[2]) >
               np.abs(endPoint[0] - startPoint[0]))
    if (steepXZ):
        startPoint[0], startPoint[2] = startPoint[2], startPoint[0]
        endPoint[0], endPoint[2] = endPoint[2], endPoint[0]

    delta = [np.abs(endPoint[0] - startPoint[0]), np.abs(endPoint[1] - startPoint[1]),
             np.abs(endPoint[2] - startPoint[2])]

    errorXY = delta[0] / 2
    errorXZ = delta[0] / 2

    step = [
        -1 if startPoint[0] > endPoint[0] else 1,
        -1 if startPoint[1] > endPoint[1] else 1,
        -1 if startPoint[2] > endPoint[2] else 1
    ]

    y = startPoint[1]
    z = startPoint[2]

    for x in range(startPoint[0], endPoint[0], step[0]):
        point = [x, y, z]

        if (steepXZ):
            point[0], point[2] = point[2], point[0]
        if (steepXY):
            point[0], point[1] = point[1], point[0]

        errorXY -= delta[1]
        errorXZ -= delta[2]

        if (errorXY < 0):
            y += step[1]
            errorXY += delta[0]

        if (errorXZ < 0):
            z += step[2]
            errorXZ += delta[0]

        path.append(np.array(point))

    return path[1:]


def extract_full_trajectores(tree: Node):

    trajectories = []
    root_id = tree.name
    leaf_nodes = list(tree.leaves)
    bifur_all = [
        node.name for node in tree.descendants if len(node.children) > 1]
    traj_lens = []
    for endpt in leaf_nodes:
        endpt_id = endpt.name
        path = find_path(tree, f"/{endpt_id}")
        path_list = path.path_name.split("/")
        bifur_ids = [p for p in path_list if p in bifur_all]
        trajectories.append(dict(root_id=root_id, endpt_id=endpt_id,
                                 bifur_ids=bifur_ids, path=path_list))
        traj_lens.append(len(path_list))
        # print(path_list)

    traj_stats = dict(min=min(traj_lens), max=max(
        traj_lens), mean=np.mean(traj_lens))

    return trajectories, traj_stats


def process_model_file(model_file: str, save_path: str):

    idx = model_file.split("/")[-2]
    sample_id = idx

    # Convert the centerlines from model to networkx directed graphs
    graphs = convert_model_to_networkx(model_file)

    # Fill the in-between nodes in the networkx directed graphs
    dense_graphs = [densify_networkx(graph) for graph in graphs]

    # Convert the networkx directed graphs to bigtree trees
    trees_with_metadata = {
        'branches': [],
        "trajectories": [],
        "traj_stats": [],
    }
    trees_with_metadata['networkx'] = dense_graphs

    # Parallel conversion of networkx graphs to bigtree trees
    args_list = [(idx, graph) for idx, graph in enumerate(dense_graphs)]
    max_workers = 4  # Set this to the number of CPU cores you want to use
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            convert_networkx_to_bigtree_wrapper, args_list))

    # Create the bigtree_dicts from the results
    bigtree_dicts = {idx: result for idx, result in results}
    trees_with_metadata['branches'] = [
        bigtree_dicts[idx] for idx in range(len(bigtree_dicts))]

    # Add lists of root point, bifurcation points, end points, and intermediate points
    # tqdm.write("Adding Points Type List")
    trees_with_metadata = add_points_type_list(trees_with_metadata)

    # Create the bigtree_dicts from the results
    bigtree_dicts = {idx: result for idx, result in results}
    trees_with_metadata['branches'] = [bigtree_dicts[idx]
                                       for idx in range(len(bigtree_dicts))]
    # Add additional info into trees_with_meta
    for idx in range(len(bigtree_dicts)):
        traj, stats = extract_full_trajectores(bigtree_dicts[idx])
        trees_with_metadata["trajectories"].append(traj)
        trees_with_metadata['traj_stats'].append(stats)
    trees_with_metadata["total_points"] = [
        len(list(bigtree_dicts[idx].descendants)) + 1
        for idx in range(len(bigtree_dicts))]

    # Save the bigtree trees
    with open(os.path.join(save_path, sample_id + ".pickle"), 'wb') as handle:
        pickle.dump(trees_with_metadata, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def convert_model_to_bigtree(directory: str):

    model_files = glob("*/skeleton.vtk", root_dir=directory, recursive=True)
    model_files = [os.path.join(directory, file) for file in model_files]
    dataset_name = "ASOCA"
    bigtree_path = os.path.join("data", dataset_name, "centerlines")
    os.makedirs(bigtree_path, exist_ok=True)

    # Sort the paths using the custom sorting key
    sorted_model_files = sorted(model_files)

    # Use ProcessPoolExecutor to parallelize the outermost loop
    # Set this to the number of CPU cores you want to use for the outer loop

    for model_file in sorted_model_files:
        print(f"Processing {model_file} ...")
        process_model_file(model_file, bigtree_path)


def main(dataset_dir):

    convert_model_to_bigtree(dataset_dir)


if __name__ == "__main__":

    os.system("clear")

    # Usage example:
    # python preprocess_synthetic_dataset.py /path/to/dataset
    sys.setrecursionlimit(50000)
    parser = argparse.ArgumentParser(
        description="Process a dataset directory.")
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to the dataset directory'
    )
    args = parser.parse_args()
    main(args.dataset_dir)
