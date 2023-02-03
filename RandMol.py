import random
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdchem import EditableMol, BondType
from rdkit.Chem.rdMolAlign import AlignMol
from itertools import repeat
from copy import deepcopy

def IsInSameRing(atom_id1, atom_id2, atom_ring_info):
    for t in atom_ring_info:
        if (atom_id1 in t) and (atom_id2 in t):
            return True
    return False

def PrepMolFrags(mol, Gen3D = False):
    # Remove Stereochemistry
    Chem.RemoveStereochemistry(mol)
    # Create hydrogen atom as substitution
    subs_atom = Chem.Atom(1)
    mH = AllChem.AddHs(mol)
    rmol = EditableMol(mH)
    atom_ring_info = mol.GetRingInfo().AtomRings()
    
    if not Gen3D:
        # Create wild card [*] atom for substitution
        subs_atom = Chem.Atom(0)
        # Convert all hydrogens to wild card atoms
        for a in mH.GetAtoms():
            if a.GetSymbol() == "H":
                H_idx = a.GetIdx()
                assert len(a.GetNeighbors()) == 1
                Nei_idx = a.GetNeighbors()[0].GetIdx()
                rmol.ReplaceAtom(H_idx, subs_atom)
                
    # Cleave single bonds and replace with wild card atoms
    # Bonds from the original mol with implicit hydrogens 
    # are used here to avoid cleaving the bonds with hydrogen
    for b in mol.GetBonds():
        if b.GetBondType() == BondType.SINGLE:
            bg_idx = b.GetBeginAtomIdx()
            bg_element = b.GetBeginAtom().GetSymbol()
            end_idx = b.GetEndAtomIdx()
            end_element = b.GetEndAtom().GetSymbol()
            # Do not break bonds in rings
            if IsInSameRing(bg_idx, end_idx, atom_ring_info):
                continue
            # Do not break single bond between heteroatoms
            if (bg_element != 'C') and (end_element != 'C'):
                continue
            rmol.RemoveBond(bg_idx,end_idx)
            subs_idx_bg = rmol.AddAtom(subs_atom) 
            subs_idx_end = rmol.AddAtom(subs_atom) 
            rmol.AddBond(bg_idx,subs_idx_bg, BondType.SINGLE)
            rmol.AddBond(end_idx,subs_idx_end, BondType.SINGLE)
                
    out_mol_frags = list(Chem.GetMolFrags(rmol.GetMol(), asMols=True))
    
    if Gen3D:
        for frag in out_mol_frags:
            AllChem.EmbedMolecule(frag, AllChem.srETKDGv3())
    
    return out_mol_frags 

def MakeMolFragDict(fragment_list):
    frag_dict = {}
    for frag in fragment_list:
        smi = Chem.MolToSmiles(frag)
        if smi not in frag_dict:
            frag_dict[smi] = {'n':1, 'mol':frag}
        else:
            frag_dict[smi]['n'] += 1
    return frag_dict

def DrawMolFragDict(frag_dict, **kwargs):
    mols, labels = [], []
    for key, sub_dict in frag_dict.items():
        stripped_smi = key.strip('*').replace('(*)','')
        label = '{}\nn={}'.format(stripped_smi, sub_dict['n'])
        labels.append(label)
        mols.append(AllChem.AddHs(Chem.MolFromSmiles(key)))
    return Draw.MolsToGridImage(mols, legends = labels, **kwargs)

def GetAtomDict(mol: Chem.Mol):
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

def PrepAttachPts(mol: Chem.RWMol):
    atom_idx_dict = GetAtomDict(mol)
    heavy_idx = random.choice(list(atom_idx_dict.keys()))
    wild_idx = random.choice(atom_idx_dict[heavy_idx])
    return wild_idx, heavy_idx

def PrepRandFrag(mol_fragment_list):
    choice_fragment = random.choice(mol_fragment_list)
    atom_idx_dict = GetAtomDict(choice_fragment)
    heavy_idx = random.choice(list(atom_idx_dict.keys()))
    wild_idx = random.choice(atom_idx_dict[heavy_idx])
    return choice_fragment, wild_idx, heavy_idx

def _RandAssemFrags(temp_frag_list, is3D):
    
    init_mol = random.choice(temp_frag_list)
    temp_frag_list.remove(init_mol)
    ed_mol = Chem.RWMol(init_mol)
    
    while len(temp_frag_list)>0:
        subs_idx1, heavy_idx1 = PrepAttachPts(ed_mol)
        
        choice_fragment, subs_idx2, heavy_idx2 = \
        PrepRandFrag(temp_frag_list)
        
        if is3D:
        # align the fragament to the ed_mol 
        # fragment will move to the ed_mol
            _rmsd = AlignMol(choice_fragment, ed_mol, 
                atomMap=(
                    (subs_idx2, heavy_idx1),
                    (heavy_idx2, subs_idx1)
                    )
                )
        # Num of atoms before insert the choice fragment
        cur_num_atoms = ed_mol.GetNumAtoms()

        ed_mol.InsertMol(choice_fragment)
        ed_mol.AddBond(
            heavy_idx1, 
            # fragament atom idx increase by the num of atom in ed_mol
            # in the previous state
            heavy_idx2+cur_num_atoms, 
            BondType.SINGLE
            )
        # remove the hydrogens or wild card from the ed_mol and the fragment
        # starting from the higher idx atom
        ed_mol.RemoveAtom(subs_idx2+cur_num_atoms)
        ed_mol.RemoveAtom(subs_idx1)
        
        temp_frag_list.remove(choice_fragment)

    for a in ed_mol.GetAtoms():
        if a.GetSymbol() == '*':
            ed_mol.ReplaceAtom(a.GetIdx(), Chem.Atom(1))

    final_mol = ed_mol.GetMol()
    final_mol.UpdatePropertyCache()
    Chem.SanitizeMol(final_mol)
    if is3D:
        AllChem.UFFOptimizeMolecule(final_mol)
    final_mol = Chem.RemoveHs(final_mol)

    return final_mol

def RandAssemFrags(fragment_dict, is3D):
    temp_frag_list = []
    for smiles, sub_dict in fragment_dict.items():
        for i in range(sub_dict['n']):
            temp_frag_list.append(sub_dict['mol'])
    try:
        return _RandAssemFrags(temp_frag_list, is3D)
    
    except IndexError:
        # redo random assemble if the structure 
        # runs out of wild card before includes all fragments
        # e.g. F-F, CF4, CCl4, etc formed
        return RandAssemFrags(fragment_dict, is3D)
        
def CreatePathStateDict(start_mol, end_mol, Gen3D = False):
    start_dict = MakeMolFragDict(PrepMolFrags(start_mol, Gen3D = Gen3D))
    end_dict = MakeMolFragDict(PrepMolFrags(end_mol, Gen3D = Gen3D))

    st_set = set(start_dict.keys())
    end_set = set(end_dict.keys())
    common = st_set.intersection(end_set)
    remove = st_set.difference(end_set)
    add = end_set.difference(st_set)

    common_dict = {}
    remove_dict = {}
    add_dict = {}

    for smi in common:
        mol = start_dict[smi]['mol']
        st_n, end_n = start_dict[smi]['n'], end_dict[smi]['n']
        n_common = min(st_n, end_n)
        common_dict[smi] = {'n': n_common, 'mol': mol}
        n_diff = st_n - end_n
        if n_diff > 0:
            remove_dict[smi] = {'n': n_diff, 'mol': mol}
        elif n_diff < 0:
            add_dict[smi] = {'n': -n_diff, 'mol': mol}

    remove_dict = {
        **remove_dict, 
        **{smi: deepcopy(start_dict[smi]) for smi in remove}
    }

    add_dict = {
        **add_dict, 
        **{smi: deepcopy(end_dict[smi]) for smi in add}
    }
    
    init_state_dict = {
        **deepcopy(start_dict),
        **{smi: {'n': 0, 'mol': end_dict[smi]['mol']} for smi in add}
    }
    
    path_state_dict = {
        'start': start_dict,
        'end': end_dict,
        'common': common_dict,
        'remove': remove_dict,
        'add': add_dict,
        'cur_state': init_state_dict
    }
    
    return path_state_dict

def ReportPathDict(path_state_dict):
    svgs_dict = {}
    for key, frag_dict in path_state_dict.items():
        if frag_dict:
#             print(key)
            svgs_dict[key] = DrawMolFragDict(frag_dict, molsPerRow = 4, useSVG=True)
        else:
#             print(key, 'is EMPTY')
            svgs_dict[key] = None
    return svgs_dict
            
def GenRandPath(perm_path_state_dict, molsPerState = 100, Gen3D = False):
    path_state_dict = deepcopy(perm_path_state_dict)
    choices = []
    for op in ('add','remove'):
        for smi, sub_dict in path_state_dict[op].items():
            for i in range(sub_dict['n']):
                choices.append((op, smi))

    path = {}
    state_id = 0
    while len(choices) > 0:
        op, smi = random.choice(choices)
        choices.remove((op, smi))
        if op == 'add':
            path_state_dict['cur_state'][smi]['n'] += 1
        else:
            path_state_dict['cur_state'][smi]['n'] -= 1

        path_state_dict[op][smi]['n'] -= 1

        if path_state_dict[op][smi]['n'] == 0:
            path_state_dict[op].pop(smi)

        cur_state_mols = []
        for i in range(molsPerState):
            cur_state_mols.append(RandAssemFrags(path_state_dict['cur_state'], is3D = Gen3D))

        path[state_id] = {
            'state': deepcopy(path_state_dict['cur_state']),
            'mols': cur_state_mols
        }
        state_id += 1
    return path