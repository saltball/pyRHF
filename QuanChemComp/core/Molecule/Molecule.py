# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : Molecule.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import sys, uuid

import numpy as np

from QuanChemComp.core.classAttr import lazyattr
from QuanChemComp.core.GeomCalcu import tensor_angle, martrix_dis_calcu
from QuanChemComp.core.Molecule.PointGroup.PointGroup import *
from QuanChemComp.lib.elements import ELEMENTS

BOHR = 0.5291772083


class Molecule(object):
    """ The Molecule Object.

    """

    def __init__(self, mol_name: str = None, atom_list: list = None, geom: list = None, charge: int = 0,
                 multi: int = 0, **kwargs):
        # Initialize the molecular name.
        if mol_name is None:
            self.mol_name = uuid.uuid1()
        else:
            self.mol_name = mol_name
        self.charge = charge
        # Check whether charge legal.
        if not isinstance(charge, int):
            raise TypeError(
                "Molecule must have charge by int type. Got {} in Molecule {}".format(self.charge, self.mol_name))
        # Define multiple
        if self.charge != 0:
            self.multi = abs(self.charge) * 2 + 1
        else:
            self.multi = 1
        # Add and translate atom_list to atom_name_list.
        self.atom_list = atom_list
        self.natom = len(self.atom_list)  # Number of atoms.
        self.geom = MolGeom(geom=geom, natom=self.natom, unit='Bohr')  # Get the MolGeom object as Geom data.
        self.point_group = PointGroup(atom_list=self.atom_list, geom=self.geom)  # Get the Point Group.
        self.basis = None
        self.__dict__.update(kwargs)  # Any other para.

    @lazyattr
    def atom_name_list(self) -> list:
        """
        Return atom name list in order of store.
        :return: atom_name_list
        The atom name list.
        """
        atom_name_list = []
        for atom in self.atom_list:
            if isinstance(atom, int) or not ("[" in atom or "(" in atom):
                try:  # Check if znum input.
                    atom_name_list.append(ELEMENTS[atom].symbol)
                except KeyError:
                    try:
                        atom = int(atom)
                        atom_name_list.append(ELEMENTS[atom].symbol)
                    except ValueError:
                        raise
            else:
                raise NotImplementedError(
                    "Atom name with [ or ( statement is not Implemented.")  # TODO(yanbohan98@gmail.com): Add isotope element.
        return atom_name_list

    @lazyattr
    def atom_num_list(self) -> list:
        atom_num_list = []
        for atom in self.atom_list:
            atom_num_list.append(ELEMENTS[atom].number)
        return atom_num_list

    @lazyattr
    def molecular_electron_num(self) -> int:
        return sum(self.atom_num_list) - self.charge

    @lazyattr
    def atom_geom_dict(self) -> dict:
        """

        :return: atom_geom_dict
        Dict with atomname as keys, geom as values.
        """
        atom_geom_dict = {}
        for i, atom in enumerate(self.atom_list):
            atom_geom_dict[(ELEMENTS[atom].symbol + str(i + 1))] = self.geom.array[i]
        return atom_geom_dict

    @lazyattr
    def atom_mass_list(self) -> list:
        """
        Return the list of mass of atoms in molecule.
        :return: atom_mass_list
        List of mass of atoms.
        """
        atom_mass_list = []
        for i, atom in enumerate(self.atom_list):
            atom_mass_list.append(ELEMENTS[self.atom_num_list[i]].isotopes[ELEMENTS[
                atom].nominalmass].mass)  # Return the most abundance isotope of element.
        return atom_mass_list

    @lazyattr
    def mol_mass(self) -> list:
        """
        The relative mass of molecule.
        :return: sum(self.atom_mass_list)
        """
        return sum(self.atom_mass_list)

    @lazyattr
    def center_of_mass(self) -> np.array:
        """
        Return the center of mass(COM).
        :return:
        """
        return (np.array(self.atom_mass_list) * self.geom.array.transpose()).sum(1) / self.mol_mass

    @lazyattr
    def moments_of_inertia(self, center="COM"):
        """
        Calculate the moments of inertia to COM.
        :param center:
        :return:
        """
        geom = self.geom.array
        I = -np.ones([3, 3]) + 2 * np.eye(3)
        if not center == "COM":
            geom = geom.translate(*center)
        return I * np.multiply(
            self.geom.array.reshape([self.natom, 3, 1]) * self.geom.array.reshape([self.natom, 1, 3]),
            np.array(self.atom_mass_list).reshape([self.natom, 1, 1])).sum(0)

    def __repr__(self):
        atom_geom_data = ""
        for item, value in self.atom_geom_dict.items():
            atom_geom_data = atom_geom_data + "{}\t\t| {}\t| {}\t| {}\t|\n".format(item, value[0], value[1], value[2])
        description = "The Molecule [{}]\nCharged {} \n##### \tThe Molecule atom-geom list:     \t#####\nATOM \t|\tx \t|\ty \t|\tz \t|\n{}##### \tThe Molecule atom-geom list END \t#####".format(
            self.mol_name, self.charge, atom_geom_data)
        return description

    def __str__(self):
        return self.__repr__()


class MolGeom(object):
    """The Geometry Object.

    """

    def __init__(self, geom=[], natom=0, unit='Bohr', **kwargs):
        self.natom = natom
        if isinstance(geom, list):
            self.array = np.array(geom).reshape((natom, 3))
            if self.array.size != 3 * natom:
                print("Geom Data is not fit the number of atom. Try to clip for 3*atom_number.", file=sys.stderr)
        if unit == 'Bohr':
            self.array = 1/BOHR * self.array
        elif unit != "A":
            raise ValueError("What is unit {}?".format(unit))
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Number of atoms: {}\nInput Cartesian coordinates:\n{}".format(self.natom,
                                                                              self.array.tolist())

    def rotate(self, phi):
        raise NotImplementedError("The molecule Rotate function is not Implemented Yet.")

    # TODO (yanbohan98@gmail.com): Implemented the Rotate Function.

    def translate(self, x, y, z):
        """
        Planer translate of the molecule geom.
        :return:
        The geom after translation.
        """
        return self.array + np.array([x, y, z])

    def _translate(self, x, y, z):
        """
        Change the geom of molecule.
        :return: None
        """
        self.array = self.translate(x, y, z)

    def bond(self, atom1, atom2):
        return np.linalg.norm(self.array[atom1 - 1] - self.array[atom2 - 1])

    def angle(self, atom1, atom2, atom3):
        """
        Return the angle from 12 to 13
        """
        direct12 = self.array[atom2 - 1] - self.array[atom1 - 1]
        direct13 = self.array[atom3 - 1] - self.array[atom1 - 1]
        return tensor_angle(direct12, direct13)

    def torsion(self, atom1, atom2, atom3, atom4):
        """
        Return the torsion/dihedral of 1-2-3-4
        """
        direct12 = self.array[atom2 - 1] - self.array[atom1 - 1]
        direct23 = self.array[atom3 - 1] - self.array[atom2 - 1]
        direct34 = self.array[atom4 - 1] - self.array[atom3 - 1]
        surface123n = np.cross(direct12 / np.linalg.norm(direct12), direct23 / np.linalg.norm(direct23))
        surface423n = np.cross(direct23 / np.linalg.norm(direct23), direct34 / np.linalg.norm(direct34))
        return tensor_angle(surface123n, surface423n)

    @lazyattr
    def bond_martix(self):
        return martrix_dis_calcu(self.array)

    @lazyattr
    def angle_tensor(self):
        raise NotImplementedError("The Angle_tensor Function is not work now.")

    # TODO(yanbohan98@gmail.com): To implement the Angle_tensor Function.

    @lazyattr
    def torsion_tensor(self):
        raise NotImplementedError("The Torsion_tensor Function is not work now.")
        # TODO(yanbohan98@gmail.com): To implement the Torsion_tensor Function.


if __name__ == "__main__":
    mol = Molecule(atom_list=[9, 9, 9, 9], geom=[[0.2, 0.3, 0.1], [1, -0.1, -0.1], [1, 2, 3], [-1, 2, 3]])
    print(mol)
    print("###\tmol.geom:\n", mol.geom)
    print("###\tmol.atom_name_list:\n", mol.atom_name_list)
    print("###\tmol.atom_geom_dict:\n", mol.atom_geom_dict)
    print("###\tmol.geom.bond:\n", mol.geom.bond(1, 2))
    print("###\tmol.center_of_mass:\n", mol.center_of_mass)
    print("###\tmol.geom.bond_martix:\n", mol.geom.bond_martix)
