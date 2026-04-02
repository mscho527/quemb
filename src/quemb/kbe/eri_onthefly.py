import logging

from numpy import complex128, zeros
from pyscf import lib
from pyscf.ao2mo.addons import restore
from pyscf.pbc.df.df import make_auxcell, make_modrho_basis
from pyscf.pbc.df.ft_ao import ft_ao, ft_aopair
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
from pyscf.pbc.df.incore import aux_e2
from pyscf.pbc.tools import get_coulG
from scipy.linalg import cholesky, solve_triangular

from quemb.molbe.eri_onthefly import block_step_size

logger = logging.getLogger(__name__)


def integral_direct_DF(mf, Fobjs, file_eri, auxbasis=None):
    r"""Calculate AO density-fitted 3-center integrals on-the-fly and transform to
    Schmidt space for given fragment objects

    Parameters
    ----------
    mf : pyscf.scf.hf.KRHF
        Mean-field object for the chemical system (typically BE.mf)
    Fobjs : list of quemb.molbe.autofrag.FragPart
        List containing fragment objects (typically BE.Fobjs)
        The MO coefficients are taken from Frags.TA and the transformed ERIs are stored
        in Frags.dname as h5py datasets.
    file_eri : h5py.File
        HDF5 file object to store the transformed fragment ERIs
    auxbasis : str, optional
        Auxiliary basis used for density fitting. If not provided, use pyscf's default
        choice for the basis set used to construct mf object; by default None
    """

    def calculate_Lpq_pbcRS(aux_range):
        r"""Internal function to calculate the 3-center integrals for a given range of
        auxiliary indices for periodic systems (with charge compensation in real space)

        Parameters
        ----------
        aux_range : tuple of int
            (start index, end index) of the auxiliary basis functions
            to calculate the 3-center integrals, i.e.
            (:math:`(pq|L)`) with L :math:`\in [start, end)` is returned
        """
        logger.debug("Start calculating (μν|P) for range %s", aux_range)
        p0, p1 = aux_range
        shls_slice = (
            0,
            mf.cell.nbas,
            0,
            mf.cell.nbas,
            p0,
            p1,
        )  # for pbc, we use aux_e2, which doesn't take in concatenated env

        ints = aux_e2(
            mf.cell,
            auxcell,
            mf.cell._add_suffix("int3c2e"),
            "s1",
            # kptij_list = , TODO: add kptij_list for k-point sampling
            shls_slice=shls_slice,
        ) - aux_e2(
            mf.cell,
            chgcell,
            mf.cell._add_suffix("int3c2e"),
            "s1",
            shls_slice=shls_slice,
        )  # Remove the part calculated in the reciprocal space to avoid double counting

        logger.debug("Finish calculating (μν|P) for range %s", aux_range)

        return ints.reshape(-1, mf.cell.nao, mf.cell.nao)  # TODO: ij pair to i, j

    def calculate_Gpq_pbcFS(pw_range):
        r"""Internal function to calculate the 3-center integrals for a given range of
        plane wave indices for periodic systems (in Fourier space)

        Parameters
        ----------
        aux_range : tuple of int
            (start index, end index) of the plane wave basis functions
            to calculate the 3-center integrals, i.e.
            (:math:`(pq|G)`) with G :math:`\in [start, end)` is returned
        """
        logger.debug("Start calculating (μν|G) for range %s", pw_range)
        g0, g1 = pw_range
        Gv_batch = Gv[g0:g1]
        coulG_batch = coulG[g0:g1]

        ft_aoao_batch = ft_aopair(mf.cell, Gv_batch)  # (G, nao, nao) # TODO kpts

        ints = ft_aoao_batch * coulG_batch.reshape(-1, 1, 1).conj()
        logger.debug("Finish calculating (μν|G) for range %s", pw_range)

        return ints  # (G, nao, nao)

    logger.info("Evaluating fragment ERIs on-the-fly using density fitting...")
    logger.info(
        "In this case, note that HF-in-HF error includes DF error on top of "
        "numerical error from embedding."
    )

    auxcell = make_auxcell(mf.cell, auxbasis)
    chgcell = make_modrho_basis(mf.cell, auxbasis)

    ccgdfbuilder = _CCGDFBuilder(mf.cell, auxcell, mf.kpts)
    ccgdfbuilder.build()

    # Prepare plane waves to evaluate in the reciprocal space for long-range contrib
    Gv, Gv_weights, kws = mf.cell.get_Gv_weights(ccgdfbuilder.mesh)
    coulG = get_coulG(mf.cell, mesh=ccgdfbuilder.mesh)  # kpt, exxdiv TODO
    coulG *= kws

    # Prepare storage for (pq|L) and (pq|G)
    pqG_frag = [
        zeros((Gv.shape[0], fragobj.nao, fragobj.nao), dtype=complex128)
        for fragobj in Fobjs
    ]
    pqL_frag = [
        zeros((auxcell.nao, fragobj.nao, fragobj.nao), dtype=complex128)
        for fragobj in Fobjs
    ]  # place to store fragment (pq|L)

    j2c = ccgdfbuilder.get_2c2e(
        zeros((1, 3))
    )  # (P|Q) accounting for periodic images and G=0 divergence
    # TODO: reexamine kpoint, as (P|Q) depends on momentum transfer
    low = cholesky(j2c[0], lower=True)

    end = 0
    Granges = [
        (x, y)
        for x, y in lib.prange(
            0,
            len(Gv),
            block_step_size(len(Fobjs), len(Gv), mf.cell.nao, dtype=complex128),
        )
    ]

    for idx, ints in enumerate(lib.map_with_prefetch(calculate_Gpq_pbcFS, Granges)):
        logger.debug("Calculating pq|G block #%d %s", idx, Granges[idx])
        # Transform pq (AO) to fragment space (ij)
        start = end
        end += ints.shape[0]
        for fragidx in range(len(Fobjs)):
            logger.debug("(μν|G) -> (ij|G) for frag #%d", fragidx)
            Gqi = ints @ Fobjs[fragidx].TA
            Giq = Gqi.transpose(0, 2, 1)
            pqG_frag[fragidx][start:end, :, :] = Giq @ Fobjs[fragidx].TA

    end = 0
    blockranges = [
        (x, y)
        for x, y in lib.prange(
            0, auxcell.nbas, block_step_size(len(Fobjs), auxcell.nbas, mf.cell.nao)
        )
    ]
    logger.debug("Aux Basis Block Info: %s", blockranges)

    for idx, ints in enumerate(lib.map_with_prefetch(calculate_Lpq_pbcRS, blockranges)):
        logger.debug("Calculating pq|L block #%d %s", idx, blockranges[idx])
        # Transform pq (AO) to fragment space (ij)
        start = end
        end += ints.shape[0]

        # Calculate (G|L) # TODO: can we move this and addition to pbcFS to save memory?
        # (G|ij) is not large but also not small, as G is typically large
        ft_auxL = ft_ao(chgcell, Gv, shls_slice=blockranges[idx])

        for fragidx in range(len(Fobjs)):
            logger.debug("(μν|P) -> (ij|P) for frag #%d", fragidx)
            Lqi = ints @ Fobjs[fragidx].TA
            Liq = Lqi.transpose(0, 2, 1)
            Lij = Liq @ Fobjs[fragidx].TA

            # Add in contributions from the reciprocal space
            pqL_frag[fragidx][start:end, :, :] = Lij + (
                ft_auxL.conj().T @ pqG_frag[fragidx].reshape(len(Gv), -1)
            ).reshape(-1, Fobjs[fragidx].nao, Fobjs[fragidx].nao)

    # Fit to get B_{ij}^{L}
    for fragidx in range(len(Fobjs)):
        logger.debug("Fitting B_{ij}^{L} for frag #%d", fragidx)
        b = pqL_frag[fragidx].reshape(auxcell.nao, -1)
        bb = solve_triangular(low, b, lower=True, overwrite_b=True, check_finite=False)
        logger.debug("Finished obtaining B_{ij}^{L} for frag #%d", fragidx)
        eri_nosym = bb.T @ bb
        if (eri_nosym.imag > 1e-6).any():
            raise ValueError(
                f"Imaginary part of ERI is larger than 1e-6 for frag #{fragidx}."
            )
        else:
            eri_nosym = eri_nosym.real
        eri = restore("4", eri_nosym, Fobjs[fragidx].nao)  # TODO kpt
        file_eri.create_dataset(Fobjs[fragidx].dname, data=eri)
