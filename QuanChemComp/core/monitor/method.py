# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : method.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from QuanChemComp.conf.conf import _erfE, _erfP, _MAXiternum
from QuanChemComp.lib.elements import ELEMENTS
from QuanChemComp.core.basis.AO import AO
import basis_set_exchange as bse
import numba
# import json
from QuanChemComp.core.Molecule.Molecule import Molecule
from scipy.linalg import sqrtm

import numpy as np

angulName = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
lListDict = {"s": [(0, 0, 0)  # s
                   ],
             "p": [(1, 0, 0),  # px
                   (0, 1, 0),  # py
                   (0, 0, 1)  # pz
                   ],
             "d": [(2, 0, 0),  # dx^2
                   (0, 2, 0),  # dy^2
                   (0, 0, 2),  # dz^2
                   (1, 0, 1),  # dxz
                   (0, 1, 1),  # dyz
                   (1, 1, 0)  # dxy
                   ],
             "f": [(3, 0, 0),  # fx^3
                   (0, 3, 0),  # fy^3
                   (0, 0, 3),  # fz^3
                   (2, 1, 0),  # fx^2y
                   (2, 0, 1),  # fx^2z
                   (1, 2, 0),  # fxy^2
                   (1, 0, 2),  # fxz^2
                   (0, 2, 1),  # fy^2z
                   (0, 1, 2),  # fyz^2
                   (1, 1, 1)  # fxyz
                   ]
             }
lListNameDict = {"s": [""  # s
                       ],
                 "p": ["x",  # px
                       "y",  # py
                       "z"  # pz
                       ],
                 "d": ["x^2",  # dx^2
                       "y^2",  # dy^2
                       "z^2",  # dz^2
                       "xz",  # dxz
                       "yz",  # dyz
                       "xy"  # dxy
                       ],
                 "f": ["x^3",  # fx^3
                       "y^3",  # fy^3
                       "z^3",  # fz^3
                       "x^2y",  # fx^2y
                       "x^2z",  # fx^2z
                       "xy^2",  # fxy^2
                       "xz^2",  # fxz^2
                       "y^2z",  # fy^2z
                       "yz^2",  # fyz^2
                       "xyz"  # fxyz
                       ]
                 }


class CalcuMethod(object):

    def __init__(self, mol: Molecule = None):
        self.molecular = mol
        self.AO = None
        if mol is not None:
            self._molecularInit()
        pass

    def _molecularInit(self):
        self.Int = {"overlap": None, "kinet": None, "nucAttr": None, "twoEle": None}
        self.AO = []
        try:
            basisKind = self.molecular.basis
        except NotImplementedError:
            raise NotImplementedError("No basis defined for {}".format(self.molecular.mol_name))

        for index, elem in enumerate(self.molecular.atom_name_list):
            _elem = ELEMENTS[elem]
            basis = bse.get_basis(basisKind, elem, header=False)
            n_Obital = {"s": 0, "p": 0, "d": 0, "f": 0, "g": 0}
            for ele_key, ele_value in basis['elements'].items():  # travel all elements(actually one)
                # store numPrime of orbitals
                for shell_value in ele_value['electron_shells']:  # travel all electron shells
                    for i, magOrbit in enumerate(shell_value['coefficients']):  # travel all groups of coeff
                        angul = shell_value['angular_momentum'][i]
                        for indexlNum, lNum in enumerate(lListDict[angulName[angul]]):
                            n_Obital[angulName[angul]] += 1
                            self.AO.append(
                                AO(id="{}_{}{}{}".format(list(self.molecular.atom_geom_dict.keys())[index],
                                                         n_Obital["s"], angulName[angul],
                                                         lListNameDict[angulName[angul]]
                                                         [indexlNum]
                                                         ),
                                   z=_elem.number,
                                   r=list(self.molecular.atom_geom_dict.values())[index],
                                   angum=angul,
                                   l=lNum,
                                   norm=np.array([1., 1., 1.]),
                                   coef=np.array([float(i) for i in magOrbit]),
                                   expon=np.array([float(i) for i in shell_value['exponents']])))


class NotImplementedCalcuMethod(CalcuMethod):
    def __init__(self):
        super(NotImplementedCalcuMethod, self).__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("CalcuMethod not implenmented!")


class RHF(CalcuMethod):
    def __init__(self, mol):
        super(CalcuMethod, self).__init__()
        self.molecular = mol
        self._molecularInit()
        self._check()

    def _check(self):
        # Define Check
        if self.molecular.basis is None:
            raise NotImplementedError("No basis defined for molecular \"{}\"".format(self.molecular.mol_name))

    def eNN(self):
        result = 0
        Z = self.molecular.atom_num_list
        D = self.molecular.geom.bond_martix
        for i in range(self.molecular.natom):
            for j in range(i + 1, self.molecular.natom):
                result += Z[i] * Z[j] / D[i][j]
        return result

    def calcuInt(self):
        import time
        print("Calculating overlap and kinetic integration...")
        st=time.time()
        self.Int["overlap"], self.Int["kinet"] = self._cal_K_S()
        et=time.time()
        print("Overlap and kinetic integration Done. {}".format(et-st))
        print("Calculating nucAttr integration...")
        st=time.time()
        self.Int["nucAttr"] = self._cal_nucAttr()
        et=time.time()
        print("NucAttr integration Done.{}".format(et-st))
        print("Calculating twoEle integration...")
        st=time.time()
        self.Int["twoEle"] = self._cal_2e_4nuc()
        et=time.time()
        print("TwoEle integration Done.{}".format(et-st))

    def scf(self):
        print("SCF For Mol {} Start.".format(self.molecular.mol_name))
        self.calcuInt()
        print(self.Int)
        nao = len(self.AO)
        nele = self.molecular.molecular_electron_num
        nocc = int(nele / 2)
        # calculate {S},{h},{g}

        # so = slice(0, nocc)
        # E_nuc = self.eNN()

        S = self.Int["overlap"]
        Hcore = self.Int["kinet"] - self.Int["nucAttr"]
        g = self.Int["twoEle"]
        # initial of Cij: Huckel Method
        eps0, c0 = np.linalg.eigh(S)
        Sinv2 = c0 @ np.diag(np.sqrt(1 / eps0)) @ c0.transpose()  # {S}^{-1/2}
        Fock1 = Sinv2 @ Hcore @ Sinv2.transpose()
        eps0, c1 = np.linalg.eigh(Fock1)
        c0 = Sinv2.transpose() @ c1
        c0 = self._c_norm(c0, nao)
        P1 = self._cal_P(c0, nao, nocc)
        # P1=np.eye(nAO)
        # print("P1",P1.__repr__())
        # caculate {G}
        E0 = np.inf
        P0 = np.inf
        CONV = 0
        eps1 = None
        import time
        print("Start SCF...")
        print("     iter     \tE     \tdE     \trms")
        st = time.time()
        for i in range(_MAXiternum + 1):
            G = self._cal_G(P1)
            F = Hcore + G
            E1 = self._cal_E(F, Hcore, P1) + self.eNN()
            # E1 = (Hcore*P1).sum()+0.5*np.einsum("uvkl,uv,kl ->",self.Int["twoEle"],P1,P1)- 0.25 * np.einsum("ukvl, uv, kl ->", self.Int["twoEle"],P1,P1)
            # E1 =E1+self.eNN()
            dE = E1 - E0
            E0 = E1
            rms = np.linalg.norm(P0 - P1) / (nao - 1)
            P0 = P1
            print("     {}\t     {}\t     {}\t     {}".format(i, E1, dE, rms))
            if (dE) ** 2 < _erfE ** 2 and rms < _erfP:
                CONV = 1
                print("SCF Done!")
                print("\n")
                et = time.time()
                print("SCF Done = {}\t RMS={}\t time cost: {}s".format(E0, rms,et-st))
                print("The HF orbital energies:\n")
                for i in range(nao):
                    print("The {} MO:\t{}".format(i + 1, eps1[i]))
                # print("The Fock_Mat:\n", F.__repr__())
                # print("The c", c1.__repr__())
                # print("The P:\n", P1.__repr__())
                break
            Fock1 = Sinv2 @ F @ Sinv2.transpose()
            eps1, c1 = np.linalg.eigh(Fock1)
            c0 = Sinv2.transpose() @ c1
            c0 = self._c_norm(c0, nao)
            P1 = self._cal_P(c0, nao, nocc)
        if not CONV == 1:
            print("****convergence FAILURE!****")

    def _cal_E(self, F, H, P):
        return 0.5 * (P * (F + H)).sum()

    def _c_norm(self, c, nAO):
        result = c
        for i in range(nAO):
            ci = 0
            for j in range(nAO):
                for k in range(nAO):
                    ci += c[j][i] * c[k][i] * self.Int["overlap"][j][k]
            for j in range(nAO):
                result[j][i] /= ci ** 0.5
        # for i in range(nAO):
        #     c[:,i]=c[:,i]/np.sqrt((c @ np.ones([nAO, nAO]) @ c.transpose()*self.Int["overlap"]).sum())
        return result

    def _cal_overlap(self):
        from QuanChemComp.core.AnalyticInteg.overlap import SxyzDefold
        nAO = len(self.AO)
        S_mat = np.zeros((nAO, nAO))
        for i in range(nAO):
            AO1 = self.AO[i]
            for j in range(i, nAO):  # symmetry
                AO2 = self.AO[j]
                Stemp = self._coeff2_mat(i, j)
                for i1, p1 in enumerate(AO1.expon):
                    for i2, p2 in enumerate(AO2.expon):
                        Stemp[i1][i2] *= SxyzDefold(p1, p2, AO1.r, AO2.r, *AO1.l, *AO2.l)
                S_mat[i][j] = Stemp.sum()
                S_mat[j][i] = Stemp.sum()
        return S_mat

    def _cal_kinet(self):
        from QuanChemComp.core.AnalyticInteg.kinetic import kinetDefold
        nAO = len(self.AO)
        S_mat = np.zeros((nAO, nAO))
        for i in range(nAO):
            AO1 = self.AO[i]
            for j in range(i, nAO):
                AO2 = self.AO[j]
                Stemp = self._coeff2_mat(i, j)
                for i1, p1 in enumerate(AO1.expon):
                    for i2, p2 in enumerate(AO2.expon):
                        Stemp[i1][i2] *= kinetDefold(p1, p2, AO1.r, AO2.r, *AO1.l, *AO2.l)
                S_mat[i][j] = Stemp.sum()
                S_mat[j][i] = Stemp.sum()
        return S_mat

    def _cal_K_S(self):
        from QuanChemComp.core.AnalyticInteg.overlap import SxyzDefold
        from QuanChemComp.core.AnalyticInteg.kinetic import kinetDefold
        nAO = len(self.AO)
        S_mat = np.zeros((nAO, nAO))
        K_mat = np.zeros((nAO, nAO))

        for i in range(nAO):
            AO1 = self.AO[i]
            for j in range(nAO):
                AO2 = self.AO[j]
                Stemp = self._coeff2_mat(i, j)
                Ktemp = self._coeff2_mat(i, j)
                for i1, p1 in enumerate(AO1.expon):
                    for i2, p2 in enumerate(AO2.expon):
                        Stemp[i1][i2] *= SxyzDefold(p1, p2, AO1.r, AO2.r, *AO1.l, *AO2.l)
                        Ktemp[i1][i2] *= kinetDefold(p1, p2, AO1.r, AO2.r, *AO1.l, *AO2.l)
                S_mat[i][j] = Stemp.sum()
                K_mat[i][j] = Ktemp.sum()
        return [S_mat, K_mat]

    def _cal_nucAttr(self):
        from QuanChemComp.core.AnalyticInteg.potential.potential import phi_2c1e
        nAO = len(self.AO)
        natom = self.molecular.natom
        nA_mat = np.zeros((nAO, nAO))
        atomNucArray = [np.array(rC) for rC in self.molecular.geom.array]
        NucChargeArray = [z for z in self.molecular.atom_num_list]
        for i in range(nAO):
            AO1 = self.AO[i]
            for j in range(i, nAO):  # notice symmetry
                AO2 = self.AO[j]
                nAtemp = self._coeff2_mat(i, j)
                for i1, p1 in enumerate(AO1.expon):
                    for i2, p2 in enumerate(AO2.expon):
                        nuc12C = 0
                        for indexC, rC in enumerate(atomNucArray):
                            nuc12C += phi_2c1e(*AO1.l, *AO2.l, p1, p2, AO1.r, AO2.r, rC) * NucChargeArray[indexC]
                        nAtemp[i1][i2] *= nuc12C
                nA_mat[i][j] = nAtemp.sum()
                nA_mat[j][i] = nAtemp.sum()
        return nA_mat

    def _cal_2e_4nuc(self):
        from QuanChemComp.core.AnalyticInteg.potential.potential import gabcd
        nAO = len(self.AO)
        twoEle_mat = np.zeros((nAO, nAO, nAO, nAO))
        for i1 in range(nAO - 1, -1, -1):
            AO1 = self.AO[i1]
            for i2 in range(i1, -1, -1):  # notice symmetry
                AO2 = self.AO[i2]
                for i3 in range(nAO - 1, -1, -1):  # notice symmetry
                    AO3 = self.AO[i3]
                    for i4 in range(i3, -1, -1):  # notice symmetry
                        AO4 = self.AO[i4]
                        gabcdtemp = self._coeff4_mat(i1, i2, i3, i4)
                        for phi1, p1 in enumerate(AO1.expon):
                            for phi2, p2 in enumerate(AO2.expon):
                                for phi3, p3 in enumerate(AO3.expon):
                                    for phi4, p4 in enumerate(AO4.expon):
                                        gabcdtemp[phi1][phi2][phi3][phi4] *= gabcd(AO1.l[0], AO2.l[0], AO3.l[0],
                                                                                   AO4.l[0],
                                                                                   AO1.l[1], AO2.l[1], AO3.l[1],
                                                                                   AO4.l[1],
                                                                                   AO1.l[2], AO2.l[2], AO3.l[2],
                                                                                   AO4.l[2],
                                                                                   p1, p2, p3, p4, AO1.r, AO2.r, AO3.r,
                                                                                   AO4.r)
                        twoEle_mat_temp = gabcdtemp.sum()
                        twoEle_mat[i1][i2][i3][i4] = twoEle_mat_temp  # 1234
                        twoEle_mat[i2][i1][i3][i4] = twoEle_mat_temp  # 2134
                        twoEle_mat[i1][i2][i4][i3] = twoEle_mat_temp  # 1243
                        twoEle_mat[i2][i1][i4][i3] = twoEle_mat_temp  # 2143
                        twoEle_mat[i3][i4][i1][i2] = twoEle_mat_temp  # 3412
                        twoEle_mat[i4][i3][i1][i2] = twoEle_mat_temp  # 4312
                        twoEle_mat[i3][i4][i2][i1] = twoEle_mat_temp  # 3421
                        twoEle_mat[i4][i3][i2][i1] = twoEle_mat_temp  # 4321
        return twoEle_mat

    def _coeff2_mat(self, i, j):
        # generate the coefficient matrix of AO1 and AO2
        AO1 = self.AO[i]
        AO2 = self.AO[j]
        coef1 = AO1.coef * np.ones([1, len(AO1.coef)])
        coef2 = AO2.coef * np.ones([1, len(AO2.coef)])
        return coef1.transpose() @ coef2

    def _coeff4_mat(self, i1, i2, i3, i4):
        # generate the coefficient matrix of AO1|AO2|AO3|AO4
        AO1 = self.AO[i1]
        AO2 = self.AO[i2]
        AO3 = self.AO[i3]
        AO4 = self.AO[i4]
        n1 = len(AO1.coef)
        n2 = len(AO2.coef)
        n3 = len(AO3.coef)
        n4 = len(AO4.coef)
        matC = np.zeros((n1, n2, n3, n4))
        for i1 in numba.prange(n1):
            for i2 in numba.prange(n2):
                for i3 in numba.prange(n3):
                    for i4 in numba.prange(n4):
                        matC[i1][i2][i3][i4] = AO1.coef[i1] * AO2.coef[i2] * AO3.coef[i3] * AO4.coef[i4]
        return matC

    def _cal_P(self, c, nAO, nocc):
        # Oslice = slice(0, nocc)
        # p = 2 * c[:, Oslice] @ c[:, Oslice].T
        p = np.zeros([nAO, nAO])
        for i in range(nAO):
            for j in range(nAO):
                for l in range(nocc):
                    p[i][j] += 2 * c[i][l] * c[j][l]
        return p

    def _cal_G(self, P):
        nAO = len(self.AO)
        g = self.Int["twoEle"]
        G=np.einsum("uvkl,kl->uv",g,P)- 0.5 * np.einsum("ukvl, kl -> uv", g,P)
        # G = np.zeros([nAO, nAO])
        # for i in range(nAO):
        #     for j in range(nAO):
        #         Gij = 0
        #         for k in range(nAO):
        #             for l in range(nAO):
        #                 Gij += P[k][l] * (g[i][j][k][l] - 0.5 * g[i][k][j][l])
        #         G[i][j] = Gij
        return G


if __name__ == "__main__":
    # mol = Molecule(atom_list=[6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
    #                geom=[[-0.5036726, 0.0786988, 0],
    #                      [0.8914874, 0.0786988, 0],
    #                      [1.5890254, 1.2864498, 0],
    #                      [0.8913714, 2.4949588, -0.0011990],
    #                      [-0.5034536, 2.4948808, -0.0016780],
    #                      [-1.2010546, 1.2866748, -0.0006820],
    #                      [-1.0534316, -0.8736182, 0.0004500],
    #                      [1.4409954, -0.8738142, 0.0013150],
    #                      [2.6887054, 1.2865298, 0.0006340],
    #                      [1.4415714, 3.4471018, -0.0012580],
    #                      [-1.0535756, 3.4471618, -0.0026310],
    #                      [-2.3006586, 1.2868578, -0.0008620]],
    #                basis="sto-3g")
    # mol = Molecule(atom_list=[1,1],
    #            geom=[[0.,0.,0.35612],
    #                  [0.,0.,-0.35612]],
    #            basis="3-21g")
    mol = Molecule(atom_list=[7, 1, 1, 1],
                   geom=[[-1.2067156, 1.1385100, 0],
                         [-0.8733937, 0.1956969, 0],
                         [-0.8733765, 1.6099101, 0.8164967],
                         [-0.8733765, 1.6099101, -0.8164967]],
                   basis="sto-3g")
    Handle = RHF(mol)
    # Handle.calcuInt()
    # print([i.id for i in Handle.AO])
    # print([i.r for i in Handle.AO])
    import timeit
    import scipy

    # print("Energy allclose   ", np.allclose(E_tot, scf_eng.e_tot))
    # print("Density allclose  ", np.allclose(D, scf_eng.make_rdm1()))
    Handle.scf()
    # print(Handle.eNN())
    # print(Handle.Int)
    # print(Handle.Int["overlap"].__repr__())
    # print(
    # (Handle._cal_K_S()[1]-Handle._cal_nucAttr()).__repr__())
    # print(timeit.repeat("Handle._cal_2e_4nuc()", "from __main__ import Handle", number=100, repeat=5))
