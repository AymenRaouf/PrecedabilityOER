from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear, MLP, SimpleConv
from torch_geometric.data import HeteroData
from torch import Tensor
import torch

class ModelReinjection(torch.nn.Module):
    def __init__(self, node_types, heads, hidden_channels, out_channels, entity_features, num_layers):
        super().__init__()
        self.gnn = self.HeteroGNN(node_types, heads, hidden_channels, out_channels, num_layers)
        self.gnn = self.gnn.float()
        self.classifier = self.Classifier(hidden_channels + entity_features)

    def forward(self, data : HeteroData) -> Tensor:
        node_dict_emb = {
            "OER" : data["OER"].x,
            "Concept" : data["Concept"].x,
            "Class" : data["Class"].x
        }
        node_dict = {
            "OER" : data["OER"].x,
            "Concept" : data["Concept"].x,
            "Class" : data["Class"].x
        }
        edge_dict = {
            ("OER", "before_sr", "OER"): data["OER", "before_sr", "OER"].edge_label_index,
            ("OER", "before_ep", "OER"): data["OER", "before_ep", "OER"].edge_index,
            ("OER", "covers", "Concept") : data["OER", "covers", "Concept"].edge_index,
            ("Concept", "belongs", "Class") : data["Concept", "belongs", "Class"].edge_index,
            ("Concept", "rev_covers", "OER") : data["Concept", "rev_covers", "OER"].edge_index,
            ("Class", "rev_belongs", "Concept") : data["Class", "rev_belongs", "Concept"].edge_index
        }

        node_dict_emb = self.gnn(node_dict_emb, edge_dict)
        node_dict = {
            "OER" : torch.cat((data["OER"].x, node_dict_emb["OER"]), dim = 1),
            "Concept" : torch.cat((data["Concept"].x, node_dict_emb["Concept"]), dim = 1),
            "Class" : torch.cat((data["Class"].x, node_dict_emb["Class"]), dim = 1)
        }
        pred = self.classifier(
            node_dict,
            edge_dict
        )

        return pred
    
    class Classifier(torch.nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.mlp = MLP([input_channels * 2, 512, 256, 128, 64, 1])

        def forward(self, node, edge) -> Tensor:
            edge_feat_oer_before = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][0]])
            edge_feat_oer_after = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][1]])
            edge_vec = torch.cat((edge_feat_oer_before, edge_feat_oer_after), dim = 1)
            prod = self.mlp(edge_vec)
            return torch.squeeze(prod)
        
    class HeteroGNN(torch.nn.Module):
        def __init__(self, node_types, heads, hidden_channels, out_channels, num_layers):
            super().__init__()


            self.lin_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('OER', 'before_ep', 'OER') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = True, cached = False),
                    ('OER', 'covers', 'Concept') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False),
                    ('Concept', 'belongs', 'Class') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False),
                    ('Concept', 'rev_covers', 'OER') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False),
                    ('Class', 'rev_belongs', 'Concept') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False)
                }, aggr = 'mean') #experiment with cat for aggr instead of mean
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                node_type: self.lin_dict[node_type](x)
                for node_type, x in x_dict.items()
            }
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            return x_dict
        

class ModelSAGE(torch.nn.Module):
    def __init__(self, node_types, heads, hidden_channels, out_channels, entity_features, num_layers):
        super().__init__()
        self.gnn = self.HeteroGNN(node_types, heads, hidden_channels, out_channels, num_layers)
        self.gnn = self.gnn.float()
        self.classifier = self.Classifier(hidden_channels)

    def forward(self, data : HeteroData) -> Tensor:
        node_dict = {
            "OER" : data["OER"].x,
            "Concept" : data["Concept"].x,
            "Class" : data["Class"].x
        }
        edge_dict = {
            ("OER", "before_sr", "OER"): data["OER", "before_sr", "OER"].edge_label_index,
            ("OER", "before_ep", "OER"): data["OER", "before_ep", "OER"].edge_index,
            ("OER", "covers", "Concept") : data["OER", "covers", "Concept"].edge_index,
            ("Concept", "belongs", "Class") : data["Concept", "belongs", "Class"].edge_index,
            ("Concept", "rev_covers", "OER") : data["Concept", "rev_covers", "OER"].edge_index,
            ("Class", "rev_belongs", "Concept") : data["Class", "rev_belongs", "Concept"].edge_index
        }

        node_dict = self.gnn(node_dict, edge_dict)
        pred = self.classifier(
            node_dict,
            edge_dict
        )

        return pred
    
    class Classifier(torch.nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.mlp = MLP([input_channels * 2, 512, 256, 128, 64, 1])

        def forward(self, node, edge) -> Tensor:
            edge_feat_oer_before = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][0]])
            edge_feat_oer_after = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][1]])
            edge_vec = torch.cat((edge_feat_oer_before, edge_feat_oer_after), dim = 1)
            prod = self.mlp(edge_vec)
            return torch.squeeze(prod)
        
    class HeteroGNN(torch.nn.Module):
        def __init__(self, node_types, heads, hidden_channels, out_channels, num_layers):
            super().__init__()


            self.lin_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('OER', 'before_ep', 'OER') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = True, cached = False),
                    ('OER', 'covers', 'Concept') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False),
                    ('Concept', 'belongs', 'Class') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False),
                    ('Concept', 'rev_covers', 'OER') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False),
                    ('Class', 'rev_belongs', 'Concept') : SAGEConv((-1, -1), hidden_channels, heads =heads,  add_self_loops = False, cached = False)
                }, aggr = 'mean') # experiment with cat for aggr instead of mean
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                node_type: self.lin_dict[node_type](x)
                for node_type, x in x_dict.items()
            }
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            return x_dict
        

class ModelGAT(torch.nn.Module):
    def __init__(self, node_types, heads, hidden_channels, out_channels, entity_features, num_layers):
        super().__init__()
        self.gnn = self.HeteroGNN(node_types, heads, hidden_channels, out_channels, num_layers)
        self.gnn = self.gnn.float()
        self.classifier = self.Classifier(hidden_channels * heads)

    def forward(self, data : HeteroData) -> Tensor:
        node_dict = {
            "OER" : data["OER"].x,
            "Concept" : data["Concept"].x,
            "Class" : data["Class"].x
        }
        edge_dict = {
            ("OER", "before_sr", "OER"): data["OER", "before_sr", "OER"].edge_label_index,
            ("OER", "before_ep", "OER"): data["OER", "before_ep", "OER"].edge_index,
            ("OER", "covers", "Concept") : data["OER", "covers", "Concept"].edge_index,
            ("Concept", "belongs", "Class") : data["Concept", "belongs", "Class"].edge_index,
            ("Concept", "rev_covers", "OER") : data["Concept", "rev_covers", "OER"].edge_index,
            ("Class", "rev_belongs", "Concept") : data["Class", "rev_belongs", "Concept"].edge_index
        }

        node_dict = self.gnn(node_dict, edge_dict)
        pred = self.classifier(
            node_dict,
            edge_dict
        )

        return pred
    
    class Classifier(torch.nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.mlp = MLP([input_channels * 2, 512, 256, 128, 64, 1])

        def forward(self, node, edge) -> Tensor:
            edge_feat_oer_before = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][0]])
            edge_feat_oer_after = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][1]])
            edge_vec = torch.cat((edge_feat_oer_before, edge_feat_oer_after), dim = 1)
            prod = self.mlp(edge_vec)
            return torch.squeeze(prod)
        
    class HeteroGNN(torch.nn.Module):
        def __init__(self, node_types, heads, hidden_channels, out_channels, num_layers):
            super().__init__()


            self.lin_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('OER', 'before_ep', 'OER') : GATConv((-1, -1), hidden_channels, add_self_loops = True, cached = False, heads = heads),
                    ('OER', 'covers', 'Concept') : GATConv((-1, -1), hidden_channels, add_self_loops = False, cached = False, heads = heads),
                    ('Concept', 'belongs', 'Class') : GATConv((-1, -1), hidden_channels, add_self_loops = False, cached = False, heads = heads),
                    ('Concept', 'rev_covers', 'OER') : GATConv((-1, -1), hidden_channels, add_self_loops = False, cached = False, heads = heads),
                    ('Class', 'rev_belongs', 'Concept') : GATConv((-1, -1), hidden_channels, add_self_loops = False, cached = False, heads = heads)
                }, aggr = 'mean') #experiment with cat for aggr instead of mean
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                node_type: self.lin_dict[node_type](x)
                for node_type, x in x_dict.items()
            }
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            return x_dict
        

class ModelConv(torch.nn.Module):
    def __init__(self, node_types, heads, hidden_channels, out_channels, entity_features, num_layers):
        super().__init__()
        self.gnn = self.HeteroGNN(node_types, heads, hidden_channels, out_channels, num_layers)
        self.gnn = self.gnn.float()
        self.classifier = self.Classifier(hidden_channels)

    def forward(self, data : HeteroData) -> Tensor:
        node_dict = {
            "OER" : data["OER"].x,
            "Concept" : data["Concept"].x,
            "Class" : data["Class"].x
        }
        edge_dict = {
            ("OER", "before_sr", "OER"): data["OER", "before_sr", "OER"].edge_label_index,
            ("OER", "before_ep", "OER"): data["OER", "before_ep", "OER"].edge_index,
            ("OER", "covers", "Concept") : data["OER", "covers", "Concept"].edge_index,
            ("Concept", "belongs", "Class") : data["Concept", "belongs", "Class"].edge_index,
            ("Concept", "rev_covers", "OER") : data["Concept", "rev_covers", "OER"].edge_index,
            ("Class", "rev_belongs", "Concept") : data["Class", "rev_belongs", "Concept"].edge_index
        }

        node_dict = self.gnn(node_dict, edge_dict)
        pred = self.classifier(
            node_dict,
            edge_dict
        )

        return pred
    
    class Classifier(torch.nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.mlp = MLP([input_channels * 2, 512, 256, 128, 64, 1])

        def forward(self, node, edge) -> Tensor:
            edge_feat_oer_before = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][0]])
            edge_feat_oer_after = torch.squeeze(node["OER"][edge[("OER", "before_sr", "OER")][1]])
            edge_vec = torch.cat((edge_feat_oer_before, edge_feat_oer_after), dim = 1)
            prod = self.mlp(edge_vec)
            return torch.squeeze(prod)
        
    class HeteroGNN(torch.nn.Module):
        def __init__(self, node_types, heads, hidden_channels, out_channels, num_layers):
            super().__init__()


            self.lin_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('OER', 'before_ep', 'OER') : SimpleConv(),
                    ('OER', 'covers', 'Concept') : SimpleConv(),
                    ('Concept', 'belongs', 'Class') : SimpleConv(),
                    ('Concept', 'rev_covers', 'OER') : SimpleConv(),
                    ('Class', 'rev_belongs', 'Concept') : SimpleConv()
                }, aggr = 'mean') #experiment with cat for aggr instead of mean
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                node_type: self.lin_dict[node_type](x)
                for node_type, x in x_dict.items()
            }
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            return x_dict