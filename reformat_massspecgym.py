import argparse
import json
import typing as T
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from cache_decorator import Cache
from huggingface_hub import hf_hub_download
from matchms import Spectrum
from matchms.filtering import default_filters
from matchms.logging_functions import set_matchms_logger_level
from pandarallel import pandarallel
from tqdm import tqdm

set_matchms_logger_level("ERROR")
# Initialize pandarallel (add progress bar if you want)
pandarallel.initialize(progress_bar=True)
import re
from itertools import groupby
from typing import Iterator

from matchms import Spectrum
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ms_pred import common

NAME_STRING = r"<(.*)>"
COLLISION_REGEX = "([0-9]+)"
VALID_ELS = set(
    [
        "C",
        "N",
        "P",
        "O",
        "S",
        "Si",
        "I",
        "H",
        "Cl",
        "F",
        "Br",
        "B",
        "Se",
        "Fe",
        "Co",
        "As",
        "Na",
        "K",
    ]
)
ION_MAP = {
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M+NH4]+": "[M+H3N+H]+",
    "[M+H-2H2O]+": "[M-H4O2+H]+",
}


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.

    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
        filename="data/" + file_name,
        repo_type="dataset",
    )


@Cache(use_approximated_hash=True)
def load_massspecgym(fold: T.Optional[str] = None) -> pd.DataFrame:
    """
    Load the MassSpecGym dataset.

    Args:
        fold (str, optional): Fold name to load. If None, the entire dataset is loaded.
    """
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df["mzs"] = df["mzs"].apply(parse_spec_array)
    df["intensities"] = df["intensities"].apply(parse_spec_array)
    if fold is not None:
        df = df[df["fold"] == fold]

    df["spectrum"] = df.apply(
        lambda row: np.array([row["mzs"], row["intensities"]]), axis=1
    )
    return df


def to_spectrum(row: pd.Series) -> Spectrum:
    """
    Convert a DataFrame row to a Spectrum object.
    """
    return Spectrum(
        mz=np.array(row["mzs"]),
        intensities=np.array(row["intensities"]),
        metadata={
            "identifier": row.name,
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "formula": row["formula"],
            "precursor_formula": row["precursor_formula"],
            "parent_mass": row["parent_mass"],
            "precursor_mz": row["precursor_mz"],
            "adduct": row["adduct"],
            "instrument_type": row["instrument_type"],
            "collision_energy": row["collision_energy"],
            "fold": row["fold"],
            "simulation_challenge": row["simulation_challenge"],
        },
    )


@Cache(
    cache_path="cache/{function_name}/{_hash}/spectra.pkl",
    use_approximated_hash=True,
)
def to_spectra(df: pd.DataFrame) -> T.List[Spectrum]:
    # Apply to_spectrum + default_filters in parallel
    spectra = df.parallel_apply(
        lambda row: default_filters(to_spectrum(row)), axis=1
    ).tolist()
    return spectra


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def fails_filter(
    entry,
    valid_adduct=list(common.ion2mass.keys()),
    max_mass=1500,
):
    """fails_filter."""
    if entry["PRECURSOR TYPE"] not in valid_adduct:
        return True

    if "EXACT MASS" not in entry or float(entry["EXACT MASS"]) > max_mass:
        return True

    # QTOF, HCD,
    # if entry["INSTRUMENT TYPE"].upper() != "HCD":
    #    return True

    form_els = get_els(entry["FORMULA"])
    if len(form_els.intersection(VALID_ELS)) != len(form_els):
        return True

    return False


def get_els(form):
    return {i[0] for i in re.findall("([A-Z][a-z]*)([0-9]*)", form)}


def process_spectrum(spectrum: Spectrum) -> dict:
    output_dict = {}
    output_dict["Peaks"] = list(zip(spectrum.mz, spectrum.intensities))
    output_dict["INCHIKEY"] = spectrum.get("inchikey")
    output_dict["FORMULA"] = spectrum.get("formula")
    output_dict["spec_id"] = spectrum.get("identifier")
    output_dict["PRECURSOR TYPE"] = spectrum.get("adduct")
    output_dict["INSTRUMENT TYPE"] = spectrum.get("instrument_type")
    output_dict["COLLISION ENERGY"] = spectrum.get("collision_energy")
    output_dict["smiles"] = spectrum.get("smiles")
    output_dict["PRECURSOR M/Z"] = spectrum.get("precursor_mz")
    output_dict["EXACT MASS"] = spectrum.get("parent_mass")
    output_dict["FOLD"] = spectrum.get("fold")
    output_dict["SYNONYMS"] = spectrum.get("inchikey")

    if np.isnan(spectrum.get("collision_energy")):
        return {}

    # Apply filter before converting
    if fails_filter(output_dict):
        return {}

    return output_dict


def merge_data(collision_dict: dict):
    base_dict = None
    out_peaks = {}
    num_peaks = 0
    energies = []
    for energy, sub_dict in collision_dict.items():
        if base_dict is None:
            base_dict = sub_dict
        if energy in out_peaks:
            print(f"Unexpected to see {energy} in {json.dumps(sub_dict, indent=2)}")
            raise ValueError()
        out_peaks[energy] = np.array(sub_dict["Peaks"])
        energies.append(energy)
        num_peaks += len(out_peaks[energy])

    # if 'nan' not in energies:
    #     energies.append('nan')  # add a "nan" entry for merged spectrum

    base_dict["Peaks"] = out_peaks
    base_dict["COLLISION ENERGY"] = energies
    base_dict["NUM PEAKS"] = num_peaks

    peak_list = list(base_dict.pop("Peaks").items())
    info_dict = base_dict
    return (info_dict, peak_list)


def dump_fn(entry: tuple) -> T.Tuple[dict, dict]:
    # Create output entry
    entry, peaks = entry
    output_name = entry["spec_id"]
    common_name = entry.get("SYNONYMS", "")
    formula = entry["FORMULA"]
    ionization = entry["PRECURSOR TYPE"]
    parent_mass = entry["PRECURSOR M/Z"]
    out_entry = {
        "dataset": "massspecgym",  # or "nist2023",
        "spec": output_name,
        "name": common_name,
        "formula": formula,
        "ionization": ionization,
        "smiles": entry["smiles"],
        "inchikey": entry["INCHIKEY"],
        "precursor": parent_mass,
        "collision_energies": [k for k, v in peaks],
    }

    # create_output_file
    # All keys to exclude from the comments
    exclude_comments = {"Peaks"}

    header_str = [
        f">compound {common_name}",
        f">formula {formula}",
        f">ionization {ionization}",
        f">parentmass {parent_mass}",
    ]
    header_str = "\n".join(header_str)
    comment_str = "\n".join(
        [f"#{k} {v}" for k, v in entry.items() if k not in exclude_comments]
    )

    # Maps collision energy to peak set
    peak_list = []
    for k, v in peaks:
        peak_entry = []
        peak_entry.append(f">collision {k}")
        peak_entry.extend([f"{row[0]} {row[1]}" for row in v])
        peak_list.append("\n".join(peak_entry))

    peak_str = "\n\n".join(peak_list)
    out_str = header_str + "\n" + comment_str + "\n\n" + peak_str

    return out_entry, {f"{output_name}.ms": out_str}


def build_mgf_str(
    meta_spec_list: T.List[T.Tuple[dict, T.List[T.Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=("PEPMASS", "parentmass", "PRECURSOR_MZ"),
    precision=4,
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec in tqdm(meta_spec_list):
        str_rows = ["BEGIN IONS"]

        # Try to add precusor mass
        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            mz_to_inten = {}
            for i, j in spec_ar:
                i = np.round(i, precision)
                mz_to_inten[i] = mz_to_inten.get(i, 0) + j

            spec_ar = [[i, j] for i, j in mz_to_inten.items()]
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])

        else:
            raise NotImplementedError()
        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targ-dir", action="store", default="../processed_data/")
    args = parser.parse_args()
    target_directory = args.targ_dir

    target_directory = Path(target_directory)
    target_directory.mkdir(exist_ok=True, parents=True)
    target_ms = target_directory / "spec_files.hdf5"
    target_mgf = target_directory / "mgf_files"
    target_labels = target_directory / "labels.tsv"
    target_mgf.mkdir(exist_ok=True, parents=True)

    df = load_massspecgym()
    spectra = to_spectra(df)

    output_dicts = [process_spectrum(spectrum) for spectrum in tqdm(spectra)]
    parsed_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    )

    print("Shuffling dict before merge")
    for output_dict in tqdm(output_dicts):
        if len(output_dict) == 0:
            continue
        inchikey = output_dict["INCHIKEY"]
        precursor_type = output_dict["PRECURSOR TYPE"]
        instrument_type = output_dict["INSTRUMENT TYPE"]
        collision_energy = output_dict["COLLISION ENERGY"]
        precursor_mass = output_dict["PRECURSOR M/Z"]
        # if len(col_energies) == 2:
        #     collision_energy = col_energies[-1]  # take eV
        # elif len(col_energies) == 1:  # only NCE exists
        #     collision_energy = f'{float(col_energies[0]) * float(precursor_mass) / 500:.1f}'
        # else:
        #     raise ValueError('collision energy values not understood:', collision_energy)
        parsed_data[inchikey][precursor_type][instrument_type][
            collision_energy
        ] = output_dict

    # merge entries
    merged_entries = []
    print("Merging dicts")
    for inchikey, adduct_dict in tqdm(parsed_data.items()):
        for adduct, instrument_dict in adduct_dict.items():
            for instrument, collision_dict in instrument_dict.items():
                output_dict = merge_data(collision_dict)
                merged_entries.append(output_dict)

    output_tuples = [dump_fn(i) for i in merged_entries]

    output_entries = []
    ms_entries = {}
    for tup in output_tuples:
        output_entries.append(tup[0])
        ms_entries.update(tup[1])

    h5 = common.HDF5Dataset(target_ms, "w")
    h5.write_dict(ms_entries)
    h5.close()

    mgf_out = build_mgf_str(merged_entries)
    with open(target_mgf / "massspecgym.mgf", "w") as f:
        f.write(mgf_out)

    labels = pd.DataFrame(output_entries)

    # Transform ions
    labels["ionization"] = [ION_MAP.get(i, i) for i in labels["ionization"].values]
    labels.to_csv(target_labels, sep="\t", index=False)

    # add the splits of training, validation, and test
    split_dir = target_directory / "splits"
    split_dir.mkdir(exist_ok=True, parents=True)

    spec = df.loc[labels.spec].index
    fold = df.loc[spec, "fold"].values
    splits = pd.DataFrame({"spec": spec, "Fold_0": fold})
    splits.to_csv(split_dir / "split_1.tsv", sep="\t", index=False)
