# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : get_basis.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
import json

import basis_set_exchange as bse

"""
This script defines some methods to get the basis value from 
"""

def get_basis(basis_name, elements_identity, input_format='json', output_format="matrix"):
    '''
    Get basis values from BSE(basis_set_exchange) database.
    :param basis_name: str
        Name of the basis, such as "6-31G", "6-31+g*".
        It is not case sensitive.
        Check `basis_set_exchange.api` for more information.
    :param elements_identity: str or list
        List of elements that you want the basis set for.
    :param input_format: str
        "json" for now.
    :param output_format: str
        "matrix" as default.
        This defines the Output basis set format in the return value of the dict.
        # TODO (@saltball) Maybe before 2020/03 : More formats as interfaces to be done. Maybe numpy.array, pytorch.Tensor and so on.
    :return: {"element_name":[`object_of_the_basis`],...}
        The value of the dict depends on the parameter `output_format`.
    '''
    if input_format == "json":
        basis_json = bse.get_basis(basis_name, elements_identity, fmt='json', header=False)
        return _resolve_basis_from_json(basis_json, output_format)


def _resolve_basis_from_json(basis_json, format="matrix"):
    basis_dict = {}
    # FIXME
    if format == "matrix":
        return basis_dict  # FIXME
    else:
        raise NotImplementedError("Format {} isn't Implemented.".format(format))


if __name__ == '__main__':
    # try:
    #     basis = json.loads(bse.get_basis("cc-pVDz", "H,C,S", fmt='json', header=False))
    #     print(basis)
    #     for ele_key, ele_value in basis['elements'].items():
    #         print(ele_key)
    #         for shell_value in ele_value['electron_shells']:
    #             print("angular_momentum: {}".format(shell_value['angular_momentum']))
    #             print("exponents: {}".format(shell_value['exponents']))
    #             print("coefficients: {}".format(shell_value['coefficients']))
    # except Exception:
    #     raise
    from pprint import pprint
    bs=bse.get_basis("6-31G", "H,C,S", header=False, uncontract_spdf=True)
    pprint(bs["elements"]["6"])
