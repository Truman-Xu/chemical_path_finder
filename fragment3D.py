from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

class MolCleaver:
    def __init__(self, frag_level = 1):
        # A higher fragmentation level contains all the exceptions from the lower ones
        frag_rule_exceptions = {
            3:'Halogen',
            2:'CarbonChain',
            1:'HeteroAtom',
            0:'SameRing'
        }
        # Check for invalid fragmentation level
        if frag_level not in range(len(frag_rule_exceptions)+1):
            raise ValueError('Fragmentation level has to be in {}'.format(tuple(range(len(frag_rule_exceptions)+1))))
            
        self.frag_exceptions = [frag_rule_exceptions[i] for i in range(frag_level)]
        
    def IsInSameRing(self):
        for t in self.atom_ring_info:
            if (self.bg_idx in t) and (self.end_idx in t):
                return True
        return False

    def IsBondBreakable(self):
        bond_check_dict = {
            # check bonds in the same ring
            'SameRing': self.IsInSameRing(),
            # check single bond between heteroatoms
            'HeteroAtom': (self.bg_element != 'C') and (self.end_element != 'C'),
            # check carbon chains
            'CarbonChain': (self.bg_element == 'C') and (self.end_element == 'C'),
            # check bond between carbon and halogens
            'Halogen': ((self.bg_element == 'C') and (self.end_element in ('F','Cl','Br'))) \
                       or ((self.bg_element in ('F','Cl','Br')) and (self.end_element == 'C'))
        }
        
        for key in self.frag_exceptions:
            # If it's a fragmentation rule exception, do not break bond
            if bond_check_dict[key]:
                return False

        return True


    def MakeMolFragments(self, mol, separate = False):
        
        mol = Chem.RemoveHs(mol)
        self.atom_ring_info = mol.GetRingInfo().AtomRings()
        self.rmol = Chem.RWMol(mol) 
        
        # Cleave single bonds and replace with hydrogen to set correct valencies
        # Bonds from the original mol with no explicit hydrogens 
        # are used here to avoid cleaving the bonds with hydrogen
        for b in mol.GetBonds():
            if b.GetBondType() == BondType.SINGLE:
                self.bg_idx = b.GetBeginAtomIdx()
                self.bg_element = b.GetBeginAtom().GetSymbol()
                self.end_idx = b.GetEndAtomIdx()
                self.end_element = b.GetEndAtom().GetSymbol()
                if self.IsBondBreakable():
                    self.rmol.RemoveBond(self.bg_idx, self.end_idx)

                subs_idx_bg = self.rmol.AddAtom(Chem.Atom(1)) 
                subs_idx_end = self.rmol.AddAtom(Chem.Atom(1))
                self.rmol.AddBond(self.bg_idx,subs_idx_bg, BondType.SINGLE)
                self.rmol.AddBond(self.end_idx,subs_idx_end, BondType.SINGLE)
        
        # Remove the hydrogen because they have no 3D coords
        out_mol = Chem.RemoveAllHs(self.rmol.GetMol())

        if separate:
            out_mol = list(Chem.GetMolFrags(out_mol, asMols=True))

        return out_mol
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fragmentation tool on molecule with 3D coordinates')
    parser.add_argument('-i','--infile', type=str,
                        help='input .sdf file path')
    parser.add_argument('-o','--outfile', type=str, default="./frag3Dout.sdf",
                        help='path for file output. Default to the ./frag3Dout.sdf')
    parser.add_argument('-s','--separate', action='store_true',
                        help='save fragments as separate mol objects')
    parser.add_argument('-l','--level', type=int, default = 2,
                        help='fragmentation level. The lower the value, the smaller the fragments. Default to 2')
    
    args = parser.parse_args()
    in_mol = Chem.MolFromMolFile(args.infile)
    
    if not in_mol:
        raise FileError('Invalid sdf mol file: {}'.format(args.infile))
        
    cleaver = MolCleaver(frag_level = args.level)
    out_mol = cleaver.MakeMolFragments(in_mol, separate = args.separate)
    
    with open(args.outfile, 'w') as f:
        w = Chem.SDWriter(f)
        if isinstance(out_mol, list):
            for m in out_mol: 
                w.write(m)
        else:
            w.write(out_mol)
        w.close()