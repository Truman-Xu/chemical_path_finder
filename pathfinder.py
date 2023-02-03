import random
import numpy as np
from rdkit import Chem
from copy import copy

from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdchem import BondType
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolAlign import AlignMol

from copy import deepcopy

from path_finder import PrepMolFrags, RandAssemFrags, \
    MakeMolFragDict, DrawMolFragDict, CreatePathStateDict, \
    ReportPathDict, GenRandPath

import networkx as nx
import matplotlib.pyplot as plt

class MolGraph(nx.Graph):
    def __init__(self, mol, prefix = '', Gen3D = False) -> None:
        super().__init__()
        self.prefix = str(prefix)
        self.mol = mol
        self.mol_frags, self.atom_mapping = self.PrepMolFrags(Gen3D=Gen3D)
        self.frag_dict = MakeMolFragDict(self.mol_frags) 
        self.MakeMolGraph()

    def __repr__(self) -> str:
        return super().__repr__()

    def _repr_html_(self, options = None):
        ##TODO: incorporate rdkit.Draw and proper layout
        if options is None:
            # Use default options
            options = {
                'node_color': 'grey',
                'node_size': 500,
                'width': 3,
            }
        return nx.draw(
            self, 
            pos=nx.spring_layout(self), 
            with_labels=True, 
            **options
        )

    def ShowMolFrags(self):
        return DrawMolFragDict(
            self.frag_dict, 
            molsPerRow = 4, 
            subImgSize = (200,200), 
            useSVG=True
            )

    def IsInSameRing(self, atom_id1, atom_id2):
        for t in self.atom_ring_info:
            if (atom_id1 in t) and (atom_id2 in t):
                return True
        return False
    
    def PrepMolFrags(self, Gen3D = False):
        # Remove explicit Hs
        mol = Chem.RemoveHs(self.mol)
        # Remove Stereochemistry
        Chem.RemoveStereochemistry(mol)
        # Create hydrogen atom as substitution
        subs_atom = Chem.Atom(1)
        mH = AllChem.AddHs(mol, addCoords = Gen3D)
        rmol = Chem.RWMol(mH)
        self.atom_ring_info = mol.GetRingInfo().AtomRings()
        
        atom_connection_mapping = []
        if not Gen3D:
            # Create wild card [*] atom for substitution
            subs_atom = Chem.Atom(0)
            # Convert all hydrogens to wild card atoms
            for a in mH.GetAtoms():
                if a.GetSymbol() == "H":
                    H_idx = a.GetIdx()
                    assert len(a.GetNeighbors()) == 1
                    rmol.ReplaceAtom(H_idx, subs_atom)
                    
        # Cleave single bonds and replace with wild card atoms
        # Bonds from the original mol with no explicit hydrogens 
        # are used here to avoid cleaving the bonds with hydrogen
        for b in mol.GetBonds():
            if b.GetBondType() == BondType.SINGLE:
                bg_idx = b.GetBeginAtomIdx()
                bg_element = b.GetBeginAtom().GetSymbol()
                end_idx = b.GetEndAtomIdx()
                end_element = b.GetEndAtom().GetSymbol()
                # Do not break bonds in rings
                if self.IsInSameRing(bg_idx, end_idx):
                    continue
                # Do not break single bond between heteroatoms
                if (bg_element != 'C') and (end_element != 'C'):
                    continue
                rmol.RemoveBond(bg_idx,end_idx)
                atom_connection_mapping.append((bg_idx,end_idx))
                subs_idx_bg = rmol.AddAtom(subs_atom) 
                subs_idx_end = rmol.AddAtom(subs_atom) 
                rmol.AddBond(bg_idx,subs_idx_bg, BondType.SINGLE)
                rmol.AddBond(end_idx,subs_idx_end, BondType.SINGLE)
        
        frag_atom_mapping = []
        mol_frags = list(Chem.GetMolFrags(rmol.GetMol(), 
                                            asMols=True,
                                            fragsMolAtomMapping = frag_atom_mapping))
        
        atom_dict = dict()
        for frag_id, atom_ids in enumerate(frag_atom_mapping):
            for new_atom_id, orig_atom_id in enumerate(atom_ids):
                atom_dict[orig_atom_id] = (frag_id, new_atom_id)

        new_atom_id_map = dict()
        for orig_st, orig_end in atom_connection_mapping:
            st, end = atom_dict[orig_st], atom_dict[orig_end]
            st_node, end_node = self.prefix+str(st[0]), self.prefix+str(end[0])
            st_atom, end_atom = st[1], end[1]
            # node id mapping: (st_node, end_node)
            # atom mapping for between the two nodes: (st_atom, end_atom)
            new_atom_id_map[(st_node, end_node)] = (st_atom, end_atom)

        if Gen3D:
            for frag in mol_frags:
                AllChem.EmbedMolecule(frag, AllChem.srETKDGv3())
        
        return mol_frags, new_atom_id_map
                
    def GetAttchIdx(self, mol_frag):
        cand_idx_list = []
        for atom in mol_frag.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == '*' or symbol == 'H':
                cand_idx_list.append(atom.GetIdx())
        return cand_idx_list

    def MakeMolGraph(self):
        node_list = [
            (self.prefix+str(i),
            {
                'smiles':Chem.MolToSmiles(mol),
                'mol':mol,
                'fp':Chem.RDKFingerprint(mol),
                'cand_idx': self.GetAttchIdx(mol)
            })
            for i, mol in enumerate(self.mol_frags)
        ]
                
        self.add_nodes_from(node_list)
        self.add_edges_from(self.atom_mapping.keys())
        assert nx.is_tree(self) # make sure the graph is a spanning tree

class MolPathFinder:
    def __init__(self, start_graph: MolGraph, end_graph: MolGraph) -> None:
        self.start_graph = start_graph
        self.end_graph = end_graph
        self.cur_graph = deepcopy(start_graph)

    def IsSameNode(self, n1, n2):
        if n1['smiles'] == n2['smiles']:
            return True
        return False

    def NodeSubCost(self,n1,n2):
        return 1-TanimotoSimilarity(n1['fp'],n2['fp'])        

    def FindOptimalPathMoves(self, timeout = 60):
        optimizer = nx.optimize_edit_paths(
            self.start_graph, 
            self.end_graph, 
        #     node_match=IsSameNode,
            node_subst_cost=self.NodeSubCost,
        #     node_ins_cost = lambda x: 2,
        #     node_del_cost = lambda x: 2,
        #     edge_subst_cost = lambda x,y : 2,
            edge_del_cost = lambda x: 0.25,
            edge_ins_cost = lambda x: 0.25,
            timeout = timeout
        )
        print('Running Path Edit Move Optimization ...')
        for r in optimizer:
            self.node_ops, self.edge_ops, score = r
            print('Current Score:', round(score, 4), end='\r')
        print('Optimization Finished. Final Socre: {:.4f}'.format(score))

        self.sort_edge_ops()
        self.sort_node_ops()
        self.id_map_dict = dict(move for move in self.node_ops['identity'])
        self.edge_ops = self.recur_update_id(self.edge_ops, self.id_map_dict)
        self.convert_edge_dict_format()
        self.substitute_identity_nodes()
        st_atom_mapping = self.recur_update_id(self.cur_graph.atom_mapping, self.id_map_dict)
        end_atom_mapping = deepcopy(self.end_graph.atom_mapping)
        self.target_atom_mapping = {**st_atom_mapping,**end_atom_mapping}
        self.initial_graph = deepcopy(self.cur_graph)
        self.node_ops.pop('identity')
        self.edge_ops.pop('identity')
        self.initial_id_map_dict = deepcopy(self.id_map_dict)
        self.initial_ops = deepcopy(self.node_ops), deepcopy(self.edge_ops)

    def recur_update_id(self, x, id_map_dict):
        if isinstance(x, dict):
            return dict(self.recur_update_id(y, id_map_dict) for y in x.items())
        elif not isinstance(x, str) and hasattr(x, '__iter__'):
            return tuple(self.recur_update_id(y, id_map_dict) for y in x)
        elif x in id_map_dict:
            return id_map_dict[x]
        else:
            return x

    def convert_edge_dict_format(self):
        temp_dict = dict()
        for key, val in self.edge_ops.items():
            temp_dict[key] = list(val)
        self.edge_ops = temp_dict

    def sort_node_ops(self):
        node_ops_dict = {'identity':[], 'deletion':[],'insertion':[],'substitution':[]}
        for n1, n2 in self.node_ops:
            if n1 and not n2:
                node_ops_dict['deletion'].append(n1)
            elif not n1 and n2:
                node_ops_dict['insertion'].append(n2)
            elif n1 and n2 and \
                self.start_graph.nodes[n1]['smiles'] == self.end_graph.nodes[n2]['smiles']:
                node_ops_dict['identity'].append((n1, n2))
            else:
                node_ops_dict['substitution'].append((n1, n2))
        self.node_ops = node_ops_dict

    def sort_edge_ops(self):
        edge_ops_dict = {'identity':[], 'deletion':[],'insertion':[],'substitution':[]}
        for e1, e2 in self.edge_ops:
            if e1 and e2:
                n1a, n1b = e1
                n2a, n2b = e2
                if self.start_graph.nodes[n1a]['smiles'] == self.end_graph.nodes[n2a]['smiles'] and\
                self.start_graph.nodes[n1b]['smiles'] == self.end_graph.nodes[n2b]['smiles']:
                    edge_ops_dict['identity'].append((e1, e2))
                else:
                    edge_ops_dict['substitution'].append((e1, e2))
            elif not e1 and e2:
                edge_ops_dict['insertion'].append(e2)
            else:
                edge_ops_dict['deletion'].append(e1)
        self.edge_ops = edge_ops_dict

    def substitute_identity_nodes(self):
        for n1, n2 in self.node_ops['identity']:
            self.cur_graph.add_node(n2, **self.end_graph.nodes[n2])
            self.cur_graph.add_edges_from((n2, nei) for nei in self.cur_graph[n1])
            self.cur_graph.remove_node(n1)

    def is_graph_valencies_valid(self):
        for n in self.cur_graph.nodes:
            if len(self.cur_graph[n]) > len(self.cur_graph.nodes[n]['cand_idx']):
                return False
        return True

    def is_node_valencies_valid(self, g: MolGraph, node_id):
        return len(g[node_id]) > len(g.nodes[node_id]['cand_idx'])

    def is_merging_edge_available(self, t1_nodes, t2_nodes):
        # Check if there is an edge operation available to merge the two trees
        for edge in self.edge_ops['insertion']:
            if edge[0] in t1_nodes and edge[1] in t2_nodes:
                return True
            elif edge[1] in t1_nodes and edge[0] in t2_nodes:
                return True
        return False

    def find_merging_edges(self, t1_nodes, t2_nodes):
        av_edges = []
        # Check if there is an edge operation available to merge the two trees
        for edge in self.edge_ops['insertion']:
            if edge[0] in t1_nodes and edge[1] in t2_nodes:
                av_edges.append(edge)
            elif edge[1] in t1_nodes and edge[0] in t2_nodes:
                av_edges.append(edge)
        return av_edges

    def FindCurrentFeasibleMove(self):
        feasible_ops =  {
            'deletion': dict(),
            'insertion': dict(),
            'substitution': dict(),
            'rearrangement':dict()
                        }
        # All node substitutions are feasible 
        feasible_ops['substitution'] = deepcopy(self.node_ops['substitution'])
        
        if len(self.node_ops['substitution']) > 0:
            node_before, node_after = zip(*self.node_ops['substitution'])
            node_before, node_after= set(node_before), set(node_after)

            for edge_1, edge_2 in self.edge_ops['substitution']:
                edge_1 = set(edge_1)
                edge_2 = set(edge_2)
                # Reduncdant of node substitution
                if (edge_1 - edge_2).issubset(node_before) and \
                (edge_2 - edge_1).issubset(node_after):
                    pass
                elif edge_1.issubset(node_before) and edge_2.issubset(node_after):
                    pass
                # The case of graph rearrangement
                else:
                    feasible_ops['rearrangement'][edge_1] = [edge_2]
                
        for node in self.node_ops['deletion']:
            # Terminal node
            if len(self.cur_graph[node]) == 1:
                for edge in self.edge_ops['deletion']:
                    if node in edge:
                        feasible_ops['deletion'][node] = []
            
            elif len(self.cur_graph[node]) == 2:
                # Node with 2 neighbors
                temp_graph = deepcopy(self.cur_graph)
                temp_graph.remove_node(node)
                t1_nodes, t2_nodes = list(nx.connected_components(temp_graph))
                av_edges =  self.find_merging_edges(t1_nodes, t2_nodes)
                if len(av_edges) > 0:
                    feasible_ops['deletion'][node] = av_edges
                
        for node in self.node_ops['insertion']:
            avail_edges = []
            for edge in self.edge_ops['insertion']:
                n_attach_1, n_attach_2 = edge
                if node == n_attach_1 and n_attach_2 in self.cur_graph.nodes:
                    avail_edges.append(edge)
                elif node == n_attach_2 and n_attach_1 in self.cur_graph.nodes:
                    avail_edges.append(edge)
            if len(avail_edges) > 0:
                feasible_ops['insertion'][node] = avail_edges
        
        for edge in self.edge_ops['deletion']:
            temp_graph = deepcopy(self.cur_graph)
            temp_graph.remove_edge(*edge)
            t1_nodes, t2_nodes = list(nx.connected_components(temp_graph))
            av_edges = self.find_merging_edges(t1_nodes, t2_nodes)
            if len(av_edges) > 0:
                feasible_ops['rearrangement'][edge] = av_edges
            
        return feasible_ops

    def insert_node(self, node_insert, node_neighbor):
        self.cur_graph.add_node(node_insert, **self.end_graph.nodes[node_insert])
        self.cur_graph.add_edge(node_insert, node_neighbor)
        self.node_ops['insertion'].remove(node_insert)
        if (node_insert,node_neighbor) in self.edge_ops['insertion']:
            self.edge_ops['insertion'].remove((node_insert,node_neighbor))
        elif (node_neighbor,node_insert) in self.edge_ops['insertion']:
            self.edge_ops['insertion'].remove((node_neighbor,node_insert))

    def substitute_node(self, node_remove, node_add):
        self.cur_graph.add_node(node_add, **self.end_graph.nodes[node_add])
        self.cur_graph.add_edges_from((node_add, nei) for nei in self.cur_graph[node_remove])
        self.cur_graph.remove_node(node_remove)
        ## update ID in the dict and the move set
        self.id_map_dict[node_remove] = node_add
        self.edge_ops = self.recur_update_id(self.edge_ops, self.id_map_dict)
        self.convert_edge_dict_format()
        self.node_ops['substitution'].remove((node_remove, node_add))
        edges_remove = None
        for edge1, edge2 in self.edge_ops['substitution']:
            if node_remove in edge1 and node_add in edge2:
                edges_remove = (edge1, edge2)
        if edges_remove:
            self.edge_ops['substitution'].remove(edges_remove)

    def rearrange_node(self, edge_remove, edge_add):
        self.cur_graph.remove_edge(*edge_remove)
        self.cur_graph.add_edge(*edge_add)
        if (edge_remove, edge_add) in self.node_ops['substitution']:
            self.node_ops['substitution'].remove((edge_remove, edge_add))
        else:
            self.edge_ops['deletion'].remove(edge_remove)
            self.edge_ops['insertion'].remove(edge_add)

    def delete_node(self, node_remove, edge_add):
        self.cur_graph.remove_node(node_remove)
        if edge_add:
            self.cur_graph.add_edge(*edge_add)
            self.edge_ops['insertion'].remove(edge_add)
        self.node_ops['deletion'].remove(node_remove)
        # Remove edges associated with the node in the move set
        edges_remove = []
        for edge in self.edge_ops['deletion']:
            if node_remove in edge:
                edges_remove.append(edge)
        for edge in edges_remove:
            self.edge_ops['deletion'].remove(edge)

    def random_choose_move(self, move_set):
        keys = []
        for key, val in move_set.items():
            if val: keys.append(key)
        key = random.choice(list(keys))
        if key == 'deletion':
            move = random.choice(list(move_set[key].keys()))
            av_edges = list(move_set[key][move])
            if len(av_edges) > 0:
                edge_add = random.choice(list(move_set[key][move]))
            else: 
                edge_add = []
            return key, (move, edge_add) 
        if key == 'insertion' or key == 'rearrangement':
            move = random.choice(list(move_set[key].keys()))
            edge_add = random.choice(list(move_set[key][move]))
            return key, (move, edge_add)
        else:
            move = random.choice(move_set[key])
            return key, move

    def execute_move(self, move):
        key, move_tuple = move
        t1, t2 = move_tuple
        if key == 'deletion':
            self.delete_node(t1, t2)
        elif key == 'insertion':
            # node_insert, node_neiself.bor
            node_insert = t1
            for node in t2:
                if node != node_insert:
                    node_neighbor = node
            self.insert_node(node_insert, node_neighbor)
        elif key == 'rearrangement':
            # edself._remove, edself._add
            self.rearrange_node(t1, t2)
        elif key == 'substitution':
            # node_remove, node_add
            self.substitute_node(t1, t2)
        
    def GeneratePath(self):
        graph_path = [deepcopy(self.cur_graph)]
        moves = []
        while not nx.is_isomorphic(self.cur_graph, self.end_graph):
            cur_moves = self.FindCurrentFeasibleMove()
            move_choice = self.random_choose_move(cur_moves)
            moves.append(move_choice)
            self.execute_move(move_choice)
            assert nx.is_tree(self.cur_graph)
            self.cur_graph.atom_mapping = self.recur_update_id(self.cur_graph.atom_mapping, self.id_map_dict)
            graph_path.append(deepcopy(self.cur_graph))
        
        self.Reinit()
        return graph_path, moves, deepcopy(self.target_atom_mapping)

    def Reinit(self):
        self.node_ops, self.edge_ops = deepcopy(self.initial_ops)
        self.cur_graph = deepcopy(self.initial_graph)
        self.id_map_dict = deepcopy(self.initial_id_map_dict)

class GraphMolAssembler:
    def __init__(self, graph: MolGraph, atom_mapping: dict):
        self.graph = graph
        self.atom_mapping = atom_mapping
        self.init_node_id = self.find_terminal_node()
        self.node_ids = [self.init_node_id]
        self.recur_traverse_tree(self.init_node_id, self.node_ids)
        self.assembled_nodes = [self.init_node_id]
        init_fragment = self.graph.nodes[self.node_ids[0]]['mol']
        self.ed_mol = Chem.RWMol(init_fragment)
        
        self.node_atom_dict = dict(
            (node_id, self.get_atom_dict(graph.nodes[node_id]['mol']))\
            for node_id in self.node_ids
        )
        
    def find_terminal_node(self):
        for node_id in self.graph.nodes:
            if len(self.graph[node_id]) == 1:
                return node_id
        
    def recur_traverse_tree(self, node, node_list):
        neighbors = list(self.graph[node].keys())
        for nei in neighbors:
            if nei not in node_list:
                node_list.append(nei)
                self.recur_traverse_tree(nei, node_list)
        
    def get_atom_dict(self, mol):
        atom_idx_dict = {}
        for a in mol.GetAtoms():
            if a.GetSymbol() == '*' or a.GetSymbol() == 'H':
                assert len(a.GetNeighbors()) == 1
                Nei_idx = a.GetNeighbors()[0].GetIdx()
                if Nei_idx in atom_idx_dict:
                    atom_idx_dict[Nei_idx].append(a.GetIdx())
                else: 
                    atom_idx_dict[Nei_idx] = [a.GetIdx()]
        return atom_idx_dict

    def select_mol_attach_pt(self, node_id):
        nei_idx1 = None
        if (self.cur_node_id, node_id) in self.atom_mapping.keys():
            nei_idx1, nei_idx2 = self.atom_mapping[(self.cur_node_id, node_id)]
        elif (node_id, self.cur_node_id) in self.atom_mapping.keys():
            nei_idx2, nei_idx1 = self.atom_mapping[(node_id, self.cur_node_id)]

        if nei_idx1 in self.node_atom_dict[self.cur_node_id]: 
            wild_idx1 = random.choice(self.node_atom_dict[self.cur_node_id][nei_idx1])
            wild_idx2 = random.choice(self.node_atom_dict[node_id][nei_idx2])
        else:
            print(self.cur_node_id, node_id)
            wild_idx1, nei_idx1 = self.rand_select_mol_attach_pt(self.cur_node_id)
            wild_idx2, nei_idx2 = self.rand_select_mol_attach_pt(node_id)
        return wild_idx1, nei_idx1, wild_idx2, nei_idx2

    def rand_select_mol_attach_pt(self, node_id):
        ## Currently only does random selection on attachment point
        atom_idx_dict = self.node_atom_dict[node_id]
        nei_idx = random.choice(list(atom_idx_dict.keys()))
        wild_idx = random.choice(atom_idx_dict[nei_idx])
        return wild_idx, nei_idx
    
    def attach_node(self, node_id, is3D = False):
        fragment = self.graph.nodes[node_id]['mol']
        wild_idx1, nei_idx1, wild_idx2, nei_idx2 = self.select_mol_attach_pt(node_id)
        # wild_idx1, nei_idx1 = self.select_mol_attach_pt(self.cur_node_id)
        # wild_idx2, nei_idx2 = self.select_mol_attach_pt(node_id)

        if is3D:
        # align the fragament to the ed_mol 
        # fragment will move to the ed_mol
            _rmsd = AlignMol(fragment, self.ed_mol, 
                atomMap=(
                    (wild_idx2, nei_idx1),
                    (nei_idx2, wild_idx1)
                    )
                )
        # Num of atoms before insert the choice fragment
        cur_num_atoms = self.ed_mol.GetNumAtoms()
        cur_num_heavy_atoms = self.ed_mol.GetNumHeavyAtoms()
        num_frag_heavy_atoms = fragment.GetNumHeavyAtoms()
        
        self.ed_mol.InsertMol(fragment)
        self.ed_mol.AddBond(
            nei_idx1, 
            # fragament atom idx increase by the num of atom in ed_mol
            # in the previous state
            nei_idx2+cur_num_atoms, 
            BondType.SINGLE
            )
        # remove the hydrogens or wild card from the ed_mol and the fragment
        # starting from the higher idx atom
        self.ed_mol.RemoveAtom(wild_idx2+cur_num_atoms)
        self.ed_mol.RemoveAtom(wild_idx1)
        
        # Once the fragment is attached
        # update heavy atom : wild card atom mapping dict
        new_dict = {}
        for orig_nei_idx, orig_wild_list in self.node_atom_dict[node_id].items():
            new_nei_idx = orig_nei_idx+cur_num_atoms-1
            new_wild_idx_list = []
            for orig_wild_idx in orig_wild_list:
                if orig_wild_idx > wild_idx2:
                    new_wild_idx_list.append(orig_wild_idx+cur_num_atoms-2)
                elif orig_wild_idx < wild_idx2:
                    new_wild_idx_list.append(orig_wild_idx+cur_num_atoms-1)
                    
            if len(new_wild_idx_list) > 0:
                new_dict[new_nei_idx] = new_wild_idx_list

            # update the target atom mapping dict
            for node_id_pairs in self.atom_mapping.keys():
                if node_id in node_id_pairs:
                    idx = node_id_pairs.index(node_id)
                    nei_atom_ids = list(self.atom_mapping[node_id_pairs])
                    nei_atom_ids[idx] = new_nei_idx
                    self.atom_mapping[node_id_pairs] = tuple(nei_atom_ids)

        self.node_atom_dict[node_id] = new_dict
        
        # Update the mapping dict for the current node
        self.node_atom_dict[self.cur_node_id][nei_idx1].remove(wild_idx1)
        if len(self.node_atom_dict[self.cur_node_id][nei_idx1]) == 0:
            self.node_atom_dict[self.cur_node_id].pop(nei_idx1)
            
        # Update the mapping dict for all wild card index in the assembled fragments
        for assembled_node_id in self.assembled_nodes:
            new_assembled_dict = {}
            for key, vals in self.node_atom_dict[assembled_node_id].items():
                if key > wild_idx1:
                    key -= 1
                new_vals = []
                for v in vals:
                    if v > wild_idx1:
                        v -= 1
                    new_vals.append(v)
                new_assembled_dict[key] = new_vals
            self.node_atom_dict[assembled_node_id] = new_assembled_dict
                
    def assemble(self, is3D = False):
        for node_id in self.node_ids:
            self.cur_node_id = node_id
            neighbors = self.graph[node_id]
            for nei_node_id in neighbors.keys():
                if nei_node_id not in self.assembled_nodes:
                    self.attach_node(nei_node_id, is3D)
                    self.assembled_nodes.append(nei_node_id)
        
        wild_card_idx = []
        for atom in self.ed_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                wild_card_idx.append(atom.GetIdx())
                
        for i in wild_card_idx:
            self.ed_mol.ReplaceAtom(i,Chem.Atom(1))
        
        final_mol = Chem.RemoveHs(self.ed_mol.GetMol())
        return final_mol