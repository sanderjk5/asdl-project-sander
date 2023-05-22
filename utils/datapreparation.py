from utils.msgpackutils import load_msgpack_l_gz
from dpu_utils.utils import RichPath
import torch
from torch_geometric.data import Data

def prepareData(dataset_dir: RichPath):
    assert dataset_dir.exists()

    datalist = []

    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                nodes = []
                for node in graph["graph"]["nodes"]:
                    nodes.append(node)
                #print(len(graph["graph"]["nodes"]))
                # x = torch.tensor(nodes)
                Data(x=nodes)
                
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")
    print('test')