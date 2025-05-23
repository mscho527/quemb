# Author(s): Minsik Cho, Leah Weisburn

import os
import time

import h5py
from numpy import einsum, ix_, loadtxt
from pyscf import ao2mo, df, gto, qmmm, scf
from pyscf.lib import chkfile
from pyscf.tools import fcidump

from quemb.molbe.fragment import fragmentate


def libint2pyscf(
    xyzfile,
    hcore,
    basis,
    hcore_skiprows=1,
    use_df=False,
    unrestricted=False,
    spin=0,
    charge=0,
):
    """Build a pyscf Mole and RHF/UHF object using the given xyz file
    and core Hamiltonian (in libint standard format)
    c.f.
    In libint standard format, the basis sets appear in the order
    atom#   n   l   m
    0       1   0   0   1s
    0       2   0   0   2s
    0       2   1   -1  2py
    0       2   1   0   2pz
    0       2   1   1   2px
    ...
    In pyscf, the basis sets appear in the order
    atom #  n   l   m
    0       1   0   0   1s
    0       2   0   0   2s
    0       2   1   1   2px
    0       2   1   -1  2py
    0       2   1   0   2pz
    ...
    For higher angular momentum, both use [-l, -l+1, ..., l-1, l] ordering.


    Parameters
    ----------
    xyzfile : str
        Path to the xyz file
    hcore : str
        Path to the core Hamiltonian
    basis : str
        Name of the basis set
    hcore_skiprows : int, optional
        # of first rows to skip from the core Hamiltonian file, by default 1
    use_df : bool, optional
        If true, use density-fitting to evaluate the two-electron integrals
    unrestricted : bool, optional
        If true, use UHF bath
    spin : int, optional
        2S, Difference between the number of alpha and beta electrons
    charge : int, optional
        Total charge of the system

    Returns
    -------
    (pyscf.gto.mole.Mole, pyscf.scf.hf.RHF, or pyscf.pbc.scf.uhf.UHF)
    """
    # Check input validity
    if not os.path.exists(xyzfile):
        raise ValueError("Input xyz file does not exist")
    if not os.path.exists(hcore):
        raise ValueError("Input core Hamiltonian file does not exist")

    mol = gto.M(atom=xyzfile, basis=basis, spin=spin, charge=charge)
    hcore_libint = loadtxt(hcore, skiprows=hcore_skiprows)

    libint2pyscf = []
    for labelidx, label in enumerate(mol.ao_labels()):
        # pyscf: px py pz // 1 -1 0
        # libint: py pz px // -1 0 1
        if "p" not in label.split()[2]:
            libint2pyscf.append(labelidx)
        else:
            if "x" in label.split()[2]:
                libint2pyscf.append(labelidx + 2)
            elif "y" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)
            elif "z" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)

    hcore_pyscf = hcore_libint[ix_(libint2pyscf, libint2pyscf)]

    mol.incore_anyway = True
    if use_df:
        mf = scf.UHF(mol).density_fit() if unrestricted else scf.RHF(mol).density_fit()
        mydf = df.DF(mol).build()
        mf.with_df = mydf
    else:
        mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)

    mf.get_hcore = lambda *args: hcore_pyscf  # noqa: ARG005

    return mol, mf


def be2fcidump(be_obj, fcidump_prefix, basis):
    """Construct FCIDUMP file for each fragment in a given BE object
    Assumes molecular, restricted BE calculation

    Parameters
    ----------
    be_obj : molbe.mbe.BE
        BE object
    fcidump_prefix : str
        Prefix for path & filename to the output fcidump files
        Each file is named [fcidump_prefix]_f0, ...
    basis : str
        'embedding' to get the integrals in the embedding basis
        'fragment_mo' to get the integrals in the fragment MO basis
    """
    for fidx, frag in enumerate(be_obj.Fobjs):
        # Read in eri
        with h5py.File(frag.eri_file, "r") as read:
            eri = read[frag.dname][()]  # 2e in embedding basis
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx),
            h1e,
            h2e,
            frag.TA.shape[1],
            frag.nsocc,
            ms=0,
        )


def ube2fcidump(be_obj, fcidump_prefix, basis):
    """Construct FCIDUMP file for each fragment in a given BE object
    Assumes molecular, restricted BE calculation

    Parameters
    ----------
    be_obj : molbe.mbe.BE
        BE object
    fcidump_prefix : str
        Prefix for path & filename to the output fcidump files
        Each file is named [fcidump_prefix]_f0, ...
    basis : str
        'embedding' to get the integrals in the embedding basis
        'fragment_mo' to get the integrals in the fragment MO basis
    """
    for fidx, frag in enumerate(be_obj.Fobjs_a):
        # Read in eri
        with h5py.File(frag.eri_file, "r") as read:
            eri = read[frag.dname][()]  # 2e in embedding basis
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx) + "a",
            h1e,
            h2e,
            frag.TA.shape[1],
            frag.nsocc,
            ms=0,
        )

    for fidx, frag in enumerate(be_obj.Fobjs_b):
        # Read in eri
        with h5py.File(frag.eri_file, "r") as read:
            eri = read[frag.dname][()]  # 2e in embedding basis
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx) + "b",
            h1e,
            h2e,
            frag.TA.shape[1],
            frag.nsocc,
            ms=0,
        )


def be2puffin(
    xyzfile,
    basis,
    hcore=None,
    libint_inp=False,
    pts_and_charges=None,
    jk=None,
    use_df=False,
    charge=0,
    spin=0,
    nproc=1,
    ompnum=1,
    n_BE=1,
    df_aux_basis=None,
    frozen_core=True,
    localization_method="lowdin",
    unrestricted=False,
    from_chk=False,
    checkfile=None,
    ecp=None,
    frag_type="chemgen",
):
    """Front-facing API bridge tailored for SCINE Puffin

    Returns the CCSD oneshot energies
    - QM/MM notes: Using QM/MM alongside big basis sets, especially with a frozen
    core, can cause localization and numerical stability problems. Use with
    caution. Additional work to this end on localization, frozen core, ECPs,
    and QM/MM in this capacity is ongoing.
    - If running unrestricted QM/MM calculations, with ECPs, in a large basis set,
    do not freeze the core. Using an ECP for heavy atoms improves the localization
    numerics, but this is not yet compatible with frozen core on the rest of the atoms.

    Parameters
    ----------
    xyzfile : str
        Path to the xyz file
    basis : str
        Name of the basis set
    hcore : numpy.ndarray
        Two-dimensional array of the core Hamiltonian
    libint_inp : bool
        True for hcore provided in Libint format. Else, hcore input is in PySCF format
        Default is False, i.e., hcore input is in PySCF format
    pts_and_charges : tuple of numpy.ndarray
        QM/MM (points, charges). Use pyscf's QM/MM instead of starting Hamiltonian
    jk : numpy.ndarray
        Coulomb and Exchange matrices (pyscf will calculate this if not given)
    use_df : bool, optional
        If true, use density-fitting to evaluate the two-electron integrals
    charge : int, optional
        Total charge of the system
    spin : int, optional
        Total spin of the system, pyscf definition
    nproc : int, optional
    ompnum : int, optional
        Set number of processors and ompnum for the jobs
    frozen_core : bool, optional
        Whether frozen core approximation is used or not, by default True
    localization_method : str, optional
        For now, lowdin is best supported for all cases. IAOs to be expanded
        By default 'lowdin'
    unrestricted : bool, optional
        Unrestricted vs restricted HF and CCSD, by default False
    from_chk : bool, optional
        Run calculation from converged RHF/UHF checkpoint. By default False
    checkfile : str, optional
        if not None:
        - if from_chk: specify the checkfile to run the embedding calculation
        - if not from_chk: specify where to save the checkfile
        By default None
    ecp : str, optional
        specify the ECP for any atoms, accompanying the basis set
        syntax; for example :python:`{'Na': 'bfd-pp', 'Ru': 'bfd-pp'}`
        By default None
    """
    # The following imports have to happen here to avoid
    # circular dependencies.
    from quemb.molbe.mbe import BE  # noqa: PLC0415
    from quemb.molbe.ube import UBE  # noqa: PLC0415

    # Check input validity
    assert os.path.exists(xyzfile), "Input xyz file does not exist"

    mol = gto.M(atom=xyzfile, basis=basis, charge=charge, spin=spin, ecp=ecp)

    if not from_chk:
        if hcore is None:  # from point charges OR with no external potential
            hcore_pyscf = None
        else:  # from starting Hamiltonian in Libint format
            if libint_inp:
                libint2pyscf = []
                for labelidx, label in enumerate(mol.ao_labels()):
                    # pyscf: px py pz // 1 -1 0
                    # libint: py pz px // -1 0 1
                    if "p" not in label.split()[2]:
                        libint2pyscf.append(labelidx)
                    else:
                        if "x" in label.split()[2]:
                            libint2pyscf.append(labelidx + 2)
                        elif "y" in label.split()[2]:
                            libint2pyscf.append(labelidx - 1)
                        elif "z" in label.split()[2]:
                            libint2pyscf.append(labelidx - 1)

                hcore_pyscf = hcore[ix_(libint2pyscf, libint2pyscf)]
            else:
                # Input hcore is in PySCF format
                hcore_pyscf = hcore
        if jk is not None:
            jk_pyscf = (
                jk[0][ix_(libint2pyscf, libint2pyscf, libint2pyscf, libint2pyscf)],
                jk[1][ix_(libint2pyscf, libint2pyscf, libint2pyscf, libint2pyscf)],
            )

        mol.incore_anyway = True
        if unrestricted:
            if use_df and jk is None:
                raise ValueError("UHF and df are incompatible: use_df = False")
            if hcore is None:
                if pts_and_charges:
                    print(
                        "Using QM/MM Point Charges: Assuming QM structure in Angstrom "
                        "and MM Coordinates in Bohr !!!"
                    )
                    mf1 = scf.UHF(mol).set(
                        max_cycle=200
                    )  # using SOSCF is more reliable
                    # mf1 = scf.UHF(mol).set(max_cycle = 200, level_shift = (0.3, 0.2))
                    # using level shift helps, but not always. level_shift and
                    # scf.addons.dynamic_level_shift do not seem to work with QM/MM
                    # note: from the SCINE database, the structure is in Angstrom but
                    # the MM point charges are in Bohr !!
                    mf = qmmm.mm_charge(
                        mf1, pts_and_charges[0], pts_and_charges[1], unit="bohr"
                    ).newton()  # mf object, coordinates, charges
                else:
                    mf = scf.UHF(mol).set(max_cycle=200, level_shift=(0.3, 0.2))
            else:
                mf = scf.UHF(mol).set(max_cycle=200).newton()
        else:  # restricted
            if pts_and_charges:  # running QM/MM
                print(
                    "Using QM/MM Point Charges: Assuming QM structure in Angstrom and "
                    "MM Coordinates in Bohr !!!"
                )
                mf1 = scf.RHF(mol).set(max_cycle=200)
                mf = qmmm.mm_charge(
                    mf1, pts_and_charges[0], pts_and_charges[1], unit="bohr"
                ).newton()
                if use_df or jk is not None:
                    raise ValueError(
                        "Setting use_df to false and jk to none: have not tested DF "
                        "and QM/MM from point charges at the same time"
                    )
            elif use_df and jk is None:
                mf = scf.RHF(mol).density_fit(auxbasis=df_aux_basis)
            else:
                mf = scf.RHF(mol)

        if hcore is not None:
            mf.get_hcore = lambda *args: hcore_pyscf  # noqa: ARG005
        if jk is not None:
            mf.get_jk = lambda *args: jk_pyscf  # noqa: ARG005

        if checkfile:
            print("Saving checkfile to:", checkfile)
            mf.chkfile = checkfile
        time_pre_mf = time.time()
        mf.kernel()
        time_post_mf = time.time()
        if mf.converged:
            print("Reference HF Converged", flush=True)
        else:
            raise ValueError("Reference HF Unconverged -- stopping the calculation")
        if use_df:
            print(
                "Using auxillary basis in density fitting: ",
                mf.with_df.auxmol.basis,
                flush=True,
            )
            print("DF auxillary nao_nr", mf.with_df.auxmol.nao_nr(), flush=True)
        print("Time for mf kernel to run: ", time_post_mf - time_pre_mf, flush=True)

    elif from_chk:
        print("Running from chkfile", checkfile, flush=True)
        scf_result_dic = chkfile.load(checkfile, "scf")
        if unrestricted:
            mf = scf.UHF(mol)
        else:
            mf = scf.RHF(mol)
        if hasattr(mf, "with_df"):
            raise ValueError("Running from chkfile not tested with density fitting")
        mf.__dict__.update(scf_result_dic)
        if hcore:
            if libint_inp:
                libint2pyscf = []
                for labelidx, label in enumerate(mol.ao_labels()):
                    # pyscf: px py pz // 1 -1 0
                    # libint: py pz px // -1 0 1
                    if "p" not in label.split()[2]:
                        libint2pyscf.append(labelidx)
                    else:
                        if "x" in label.split()[2]:
                            libint2pyscf.append(labelidx + 2)
                        elif "y" in label.split()[2]:
                            libint2pyscf.append(labelidx - 1)
                        elif "z" in label.split()[2]:
                            libint2pyscf.append(labelidx - 1)

                hcore_pyscf = hcore[ix_(libint2pyscf, libint2pyscf)]
            else:
                # Input hcore is in PySCF format
                hcore_pyscf = hcore
            mf.get_hcore = lambda *args: hcore_pyscf  # noqa: ARG005
        elif pts_and_charges:
            print(
                "Using QM/MM Point Charges: Assuming QM structure in Angstrom and "
                "MM Coordinates in Bohr !!!"
            )
            mf = qmmm.mm_charge(
                mf,
                pts_and_charges[0],
                pts_and_charges[1],
                unit="bohr",
            ).newton()
        time_post_mf = time.time()
        print("Chkfile electronic energy:", mf.energy_elec(), flush=True)
        print("Chkfile e_tot:", mf.e_tot, flush=True)

    # Finished initial reference HF: now, fragmentation step

    fobj = fragmentate(n_BE=n_BE, frag_type=frag_type, mol=mol, frozen_core=frozen_core)
    time_post_fragmentate = time.time()
    print(
        "Time for fragmentation to run: ",
        time_post_fragmentate - time_post_mf,
        flush=True,
    )

    # Run embedding setup

    if unrestricted:
        mybe = UBE(mf, fobj, lo_method=localization_method)
        solver = "UCCSD"
    else:
        mybe = BE(mf, fobj, lo_method=localization_method)
        solver = "CCSD"

    # Run oneshot embedding and return system energy

    mybe.oneshot(solver=solver, nproc=nproc, ompnum=ompnum)
    return mybe.ebe_tot - mybe.ebe_hf


def print_energy_cumulant(ecorr, e_V_Kapprox, e_F_dg, e_hf):
    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with cumulant-based expression", flush=True)

    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
    print(f" E_HF            : {e_hf:>14.8f} Ha", flush=True)
    print(f" Tr(F del g)     : {e_F_dg:>14.8f} Ha", flush=True)
    print(f" Tr(V K_approx)  : {e_V_Kapprox:>14.8f} Ha", flush=True)
    print(f" E_BE            : {ecorr + e_hf:>14.8f} Ha", flush=True)
    print(f" Ecorr BE        : {ecorr:>14.8f} Ha", flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)


def print_energy_noncumulant(be_tot, e1, ec, e2, e_hf, e_nuc):
    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with non-cumulant expression", flush=True)
    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_1 + E_C + E_2 + E_nuc", flush=True)
    print(f" E_HF            : {e_hf:>14.8f} Ha", flush=True)
    print(f" E_Nuc           : {e_nuc:>14.8f} Ha", flush=True)
    print(f" E_BE total      : {be_tot + e_nuc:>14.8f} Ha", flush=True)
    print(f" E_1             : {e1:>14.8f} Ha", flush=True)
    print(f" E_C             : {ec:>14.8f} Ha", flush=True)
    print(f" E_2             : {e2:>14.8f} Ha", flush=True)
    print(f" Ecorr BE        : {be_tot + e_nuc - e_hf:>14.8f} Ha", flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)
