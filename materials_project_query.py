import os
from mp_api.client import MPRester
from pathlib import Path
from pymatgen.io.cif import CifWriter


def save_structure_to_cif(structure, output_file):
    """This function is a helper function to save the cif files.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The structure to save as a CIF file.
    file : os.PathLike
        The path to save the structure to.
    """

    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    cif_writer = CifWriter(structure, symprec=0.1)
    cif_writer.write_file(output_file)


def pull_data_from_Materials_Project(
    output_directory,
    elements=["Pb", "S"],
    chemsys="Pb-S",
    fields=["material_id", "structure"],
    api_key=os.environ["PMG_API_KEY"]
):
    """Executes a query to the Materials Project, and saves those materials
    to disk as CIF files.
    
    Parameters
    ----------
    output_directory : os.PathLike
        Path to the directory to save the resultant CIF files.
    elements : list
        Elements to include in the query.
    chemsys : str
        The type of chemical system to pull. For example, "Pb-S-*".
    fields : list, optional
        The fields to keep from the Materials Project.
    api_key : str, optional
        Materials Project API key. Defaults to an environment variable
        "PMG_API_KEY".
    """

    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.search(
            elements=elements,
            chemsys=chemsys,
            fields=fields
        )

    for d in docs:
        output_file = Path(output_directory) / f"{d.material_id}_structure.cif"
        save_structure_to_cif(d.structure, output_file)
