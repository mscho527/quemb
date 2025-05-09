"""
This script tests the correlation energies of sample restricted molecular
BE calculations from density matching

Author(s): Minsik Cho
"""

import unittest

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.fragment import ChemGenArgs


class TestBE_restricted(unittest.TestCase):
    # TODO: Add test against known values (molecular_restrict_test)
    def test_h8_sto3g_ben_trustRegion(self):
        # Test consistency between two QN methods
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(7)]
        mol.atom.append(["H", (0.0, 0.0, 4.2)])
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        self.molecular_QN_test(
            mol,
            2,
            "H8 (BE2)",
            "chemgen",
            only_chem=False,
            additional_args=ChemGenArgs(treat_H_different=False),
        )

    def molecular_QN_test(
        self,
        mol,
        n_BE,
        test_name,
        frag_type,
        delta=1e-6,
        only_chem=True,
        additional_args=None,
    ):
        mf = scf.RHF(mol)
        mf.max_cycle = 100
        mf.kernel()
        fobj = fragmentate(
            frag_type=frag_type,
            n_BE=n_BE,
            mol=mol,
            additional_args=additional_args,
        )
        mybe1 = BE(mf, fobj)
        mybe1.optimize(
            solver="CCSD", method="QN", trust_region=False, only_chem=only_chem
        )
        mybe2 = BE(mf, fobj)
        mybe2.optimize(
            solver="CCSD", method="QN", trust_region=True, only_chem=only_chem
        )
        self.assertAlmostEqual(
            mybe1.ebe_tot,
            mybe2.ebe_tot,
            msg="BE Correlation Energy (DM) for "
            + test_name
            + " does not return comparable results from two QN optimization!",
            delta=delta,
        )


if __name__ == "__main__":
    unittest.main()
