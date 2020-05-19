# What's this?


# How to install?
## Requirements
This package needs some third-part package to ensure fundamental functions. 
- Python 3.x
- Numpy
- Numba


```shell script
$ conda install numpy numba sympy
```
- [Basis_set_exchange](https://github.com/MolSSI-BSE) [^1]: (Without in conda distribution by April, 2020.) 
```shell script
$ pip install basis_set_exchange
```
## Get this soft
Now For Source only.
```shell script
$ git clone [$the git Repository URL]
```

# Fast Start



# Code Structure
## `/QuanChemComp`:
### `/core`
- ***file*** `math.py`
    
    -   Some important math functions.\
        Offering:
        - **Func** `Factorial(n)`
            Factorial of integer. Also called `$N!$`
            - No Error will raise.
        - **Func** `DoubleFactorial(n)`
            Double Factorial of integer. Also called `$N!!$`
            - No Error will raise.
        - **Func** `Comb(n: int, m: int)`
            Combination of m in n. Notice 0<=m<=n.
            - No Error will raise.
#### `/core/AnalyticInteg`
Most Integration **Func** use `numpy.array` values `a_array` and `b_array`. 
And `***notice the unit***` of distance data.

Most methods here using `numba.njit` to accelerate. `# TODO: GPU numba and other Parallel.`
- ***file*** `gtoMath.py`

    Focus on Common Dependency, eg. Constant, coefficients 
    - Offering some Common math functions for GTO.\
        Offering:
        - **Func** `K_GTO(a, b, dAB_2)`, `dAB_2`:x^2+y^2+z^2, not |AB|,but |AB|^2
        - **Func** `norm_GTO(a: float, la=0, ma=0, na=0)`
            for Normalization of GTO |aAlmn> if necessary. (Most always.)

- ***file*** `GammaFunc.py`

    Focus on particular Gamma Functions: Boys Func `$F_m(w)$`
    - **Func** `$F_m(w)$`.\
        Offering:
        - Analytical **Func** `_Fm(m,w)`: use bultin function from `sympy` package.
        - **- Default & Prefer -** Approximate **Func** `FmDefold(m,w)`: use Pade approximate, data from the book [^2]

- ***file*** `overlap.py`

    Focus on `$\left<aAlmn\mid bBlmn\right>$`
    - The Realization of Overlap Integration in Quantum Chemistry. \
        Offering:
        - **IMPORTANT** **Func**`SxyzDefold(a, b, a_array, b_array, la, ma, na, lb, mb, nb)`
            - for l,m,n<4,there are builtin Functions by values.

- ***file*** `kinetic.py`

    Focus on `$\left<aAlmn\mid -\frac{1}{2} \nabla^2 \mid bBlmn\right>$`
    - Depended on **Func**`SxyzDefold` in `overlap.py`
    - The Realization of Kinetic Integration in Quantum Chemistry.\    
        Offering:
        - **IMPORTANT** **Func**`kinetDefold(a, b, a_array, b_array, la, ma, na, lb, mb, nb)`
            - Use Values `Sxyz` from `overlap.py`
            
- ***file*** `potential/potential.py`

    Focus on `$\left<aAlmn\mid \frac{1}{r_{C}}\mid bBlmn\right>$` 
    and `$\left<aAlmn bBlmn\mid \frac{1}{r_{C}}\mid cClmn dDlmn \right>$`
    - The Realization of Potential Integration in Quantum Chemistry. 
    Including one-electron Integrate and two-electron Integrate. Most algorithm is from book [^3]\
        Offering:
        - **Func** `_Eijt(i: int, j: int, t: int, p, PAx, PBx)`
            - return the expansion coefficients `$E_t^{ij}/E_0^{00}$` of McMurchine-Davidson scheme.
            - Require Func `Comb` from `../math.py`
        - **Func** `_Rtuvn(t, u, v, n, p, PCx, PCy, PCz, PC2)`
            - return the Hermite Coulomb integrals `$R_{tuv}$`
            - Require Func `FmDefold` from `../GammaFunc.py`
        - **Func** `V_tuvab(i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)`
            - return the basic integrate `V^{000}_ab`
        - **IMPORTANT**  **Func** `phi_2c1e_nNorm(t, u, v, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)`
            - **Warning**: NOT NORMALIZED.
            - return all kinds of one-electron integrate with differential operators.
            - Require Func `V_tuvab(i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)` in the same .py file.
        - **Func** `phi_2c1e(i, j, k, l, m, n, a, b, rA, rB, rC, Norm=False)`
            - **NORMALIZED** version of **Func** `phi_2c1e_nNorm`
        - **IMPORTANT**  **Func** `gabcd(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, a, b, c, d, rA, rB, rC, rD)`
            - **NORMALIZED**
            - return two-electron integrate  <aAbB|1/r_{12}|cCdD>.
            - Require Func `_Rtuvn` and `_Eijt`
            
### `/core`
- ***file*** `elements.py`\
Author: `Christoph Gohlke` <https://www.lfd.uci.edu/~gohlke/>\
License in file.

#### `/core/monitor`
- ***file*** `method.py`

    Focus on HF methods. \
    Offering:
    - ***IMPORTANT*** **Class** `RHF(mol = <:class:> Molecule)`\
        instance methods:\
        - ***IMPORTANT*** **Func** `.scf()` \
            Do scf and log info in scf. # TODO: logger control

#### `/core/Molecular`
- ***file*** `Molecule.py`
    
    Focus on Molecule objects.
    Offering: \
    - ***IMPORTANT*** **Class** `Molecule(mol_name: str,atom_list: list,geom: list,charge: int,multi: int,**kwargs))`
        instance methods:\
        - `.atom_name_list()`: return a list like `["C","N"]`, contains the name of each elements
        - `.atom_num_list()`: return a list like `[6,7]`, contains the elements number (Z) of each atoms, int.
        - `.molecular_electron_num()`: return int calculated from charge and atom numbers. ($\Sigma Z-charge$)
        - `.atom_geom_dict()`: return a np.array like `array([[1,1,1],[2,2,-2]])`, order by `atom_name_list` or `atom_num_list`.
        - `.atom_mass_list()`: order by `atom_name_list` or `atom_num_list`.
        - `.mol_mass()`
        - `.center_of_mass()`
        - `.moments_of_inertia()`

# Frequently Q&A


# Rerferences

[^1]: A New Basis Set Exchange: An Open, Up-to-date Resource for the Molecular Sciences Community. Benjamin P. Pritchard, Doaa Altarawy, Brett Didier, Tara D. Gibson, Theresa L. Windus. J. Chem. Inf. Model. 2019, 59(11), 4814-4820, doi:10.1021/acs.jcim.9b00725.

[^2]: 徐光宪; 黎乐民; 王德民. 量子化学：基本原理和从头计算法, 第二版.; 科学出版社: 北京, 2007. C.2 p.67

[^3]: Helgaker T., Jørgensen P., Olsen J. (2000). Molecular Electronic-Structure Theory . Chichester: Wiley; doi:10.1002/9781119019572

[^4]: `elements.py` Author: `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_