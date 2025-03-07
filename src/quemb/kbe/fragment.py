# Author(s): Oinam Romesh Meitei


from quemb.kbe.autofrag import autogen
from quemb.kbe.chain import polychain as _ext_polychain
from quemb.molbe.helper import get_core
from quemb.shared.helper import copy_docstring


class fragpart:
    def __init__(
        self,
        natom=0,
        frag_type="autogen",
        unitcell=1,
        gamma_2d=False,
        gamma_1d=False,
        interlayer=False,
        long_bond=False,
        perpend_dist=4.0,
        perpend_dist_tol=1e-3,
        nx=False,
        ny=False,
        nz=False,
        kpt=None,
        iao_valence_basis=None,
        be_type="be2",
        mol=None,
        frozen_core=False,
        self_match=False,
        allcen=True,
    ):
        """Fragment/partitioning definition

        Interfaces two main fragmentation functions (autogen & polychain) in MolBE.
        It defines edge & center for density matching and energy estimation.
        It also forms the base for IAO/PAO partitioning for
        a large basis set bootstrap calculation. Fragments are constructed based
        on atoms within a unitcell.

        Parameters
        ----------
        frag_type : str
            Name of fragmentation function. 'autogen', 'hchain_simple', and 'chain' are
            supported. Defaults to 'autogen'
            For systems with only hydrogen, use 'chain';
            everything else should use 'autogen'
        be_type : str
            Specifies order of bootsrap calculation in the atom-based fragmentation.
            'be1', 'be2', 'be3', & 'be4' are supported.
            Defaults to 'be2'
            For a simple linear system A-B-C-D,
            be1 only has fragments [A], [B], [C], [D]
            be2 has [A, B, C], [B, C, D]
            ben ...
        mol : pyscf.pbc.gto.cell.Cell
            pyscf.pbc.gto.cell.Cell object. This is required for the options, 'autogen',
            and 'chain' as frag_type.
        iao_valence_basis: str
            Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
        frozen_core: bool
            Whether to invoke frozen core approximation. This is set to False by default
        print_frags: bool
            Whether to print out list of resulting fragments. True by default
        write_geom: bool
            Whether to write 'fragment.xyz' file which contains all the fragments in
            cartesian coordinates.
        kpt : list of int
            No. of k-points in each lattice vector direction. This is the same as kmesh.
        interlayer : bool
            Whether the periodic system has two stacked monolayers.
        long_bond : bool
            For systems with longer than 1.8 Angstrom covalent bond, set this to True
            otherwise the fragmentation might fail.
        """
        # No. of unitcells to use for fragment construction
        self.unitcell = unitcell
        self.mol = mol
        self.frag_type = frag_type
        self.fsites = []
        self.Nfrag = 0
        self.edge = []
        self.center = []
        self.ebe_weight = []
        self.edge_idx = []
        self.center_idx = []
        self.centerf_idx = []
        self.be_type = be_type
        self.natom = natom
        self.frozen_core = frozen_core
        self.self_match = self_match
        self.allcen = allcen
        self.iao_valence_basis = iao_valence_basis
        self.kpt = kpt
        self.molecule = False  # remove this

        # Check for frozen core approximation
        if frozen_core:
            self.ncore, self.no_core_idx, self.core_list = get_core(mol)

        if frag_type == "polychain":
            if mol is None:
                raise ValueError(
                    "Provide pyscf gto.Cell object in fragpart() and restart!"
                )
            self.polychain(mol, frozen_core=frozen_core, unitcell=unitcell)
        elif frag_type == "autogen":
            if mol is None:
                raise ValueError(
                    "Provide pyscf gto.Cell object in fragpart() and restart!"
                )
            if kpt is None:
                raise ValueError("Provide kpt mesh in fragpart() and restart!")

            fgs = autogen(
                mol,
                kpt,
                be_type=be_type,
                frozen_core=frozen_core,
                iao_valence_basis=iao_valence_basis,
                unitcell=unitcell,
                nx=nx,
                ny=ny,
                nz=nz,
                long_bond=long_bond,
                perpend_dist=perpend_dist,
                perpend_dist_tol=perpend_dist_tol,
                gamma_2d=gamma_2d,
                gamma_1d=gamma_1d,
                interlayer=interlayer,
            )

            (
                self.fsites,
                self.edge,
                self.center,
                self.edge_idx,
                self.center_idx,
                self.centerf_idx,
                self.ebe_weight,
            ) = fgs
            self.Nfrag = len(self.fsites)

        else:
            raise ValueError(f"Fragmentation type = {frag_type} not implemented!")

    @copy_docstring(_ext_polychain)
    def polychain(self, mol, frozen_core=False, unitcell=1):
        return _ext_polychain(self, mol, frozen_core=frozen_core, unitcell=unitcell)
