# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : ReadMol.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

class Stream(object):
    pass


class MolGeomStream(Stream):
    def __init__(self, GeomStream=NotImplemented, natom=0):
        """

        :param GeomStream: iter
        iter with molecule geom data: z_num x y z\n
        :param natom:
        """
        super().__init__()
        self.stream = GeomStream
        self.natom = natom
        self.atom_list, self.atom_geom = self._mol_geom_split(self.stream)
        if len(self.atom_list) == self.natom:
            pass
        else:
            self.natom = len(self.atom_list)

    def _mol_geom_split(self, Stream):
        atom_list = []
        atom_geom = []
        for item in Stream:
            if not len(item.split()) == 0:
                # print(item)
                x = float(item.split()[1])
                y = float(item.split()[2])
                z = float(item.split()[3])
                atom_list.append(item.split()[0])
                atom_geom.append([x, y, z])
        return atom_list, atom_geom


if __name__ == "__main__":
    str = iter("""
    6  0.000000000000     0.000000000000     0.000000000000\n\n
6  0.000000000000     0.000000000000     2.845112131228\n
8  1.899115961744     0.000000000000     4.139062527233\n
1 -1.894048308506     0.000000000000     3.747688672216\n
1  1.942500819960     0.000000000000    -0.701145981971\n
1 -1.007295466862    -1.669971842687    -0.705916966833\n
1 -1.007295466862     1.669971842687    -0.705916966833\n
""".split("\n"))
    Geom = MolGeomStream(str, 7)
    print(Geom.atom_list, Geom.atom_geom)
    from QuanChemComp.core.Molecule.Molecule import Molecule

    mol = Molecule(atom_list=Geom.atom_list, geom=Geom.atom_geom)
    print("geom:\n", mol.geom)
    print("bond:\n", mol.geom.bond_martix)
    print("Angles")
    natom = Geom.natom
    for i in range(natom):
        for j in range(natom):
            for k in range(j + 1, natom):
                print("{}\t{}\t{}\t{}".format(i, j, k, mol.geom.angle(i, j, k)))
    print("COM:\n", mol.center_of_mass)
    # print("MOI:\n",mol.moments_of_inertia)
