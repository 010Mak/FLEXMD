from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterable, Union
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as Descriptors

try:
    from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
    _INCHI_AVAILABLE = True
except Exception:
    _INCHI_AVAILABLE = False

try:
    from rdkit.Chem import rdDetermineBonds as _RDB
    _HAS_RDB = True
except Exception:
    _HAS_RDB = False

try:
    from utilities.names import resolve_molecule_names, resolve_element_name
    _NAMES_AVAILABLE = True
except Exception:
    _NAMES_AVAILABLE = False

_LOG = logging.getLogger(__name__)
_PT = Chem.GetPeriodicTable()

_TRIVIAL: Dict[Tuple[str, int], str] = {
    ("H2O", 0): "Water",
    ("NH3", 0): "Ammonia",
    ("CO2", 0): "Carbon dioxide",
    ("CO", 0): "Carbon monoxide",
    ("O2", 0): "Oxygen",
    ("O3", 0): "Ozone",
    ("H2", 0): "Hydrogen",
    ("N2", 0): "Nitrogen",
    ("Cl2", 0): "Chlorine",
    ("F2", 0): "Fluorine",
    ("Br2", 0): "Bromine",
    ("I2", 0): "Iodine",

    ("Cl", -1): "Chloride",
    ("Br", -1): "Bromide",
    ("I", -1): "Iodide",
    ("F", -1): "Fluoride",
    ("Na", +1): "Sodium",
    ("K", +1): "Potassium",
    ("Ca", +2): "Calcium",
    ("Mg", +2): "Magnesium",
    ("Al", +3): "Aluminum",
    ("S", -2): "Sulfide",

    ("HO", -1): "Hydroxide",
    ("OH", -1): "Hydroxide",
    ("H3O", +1): "Hydronium",
    ("NH4", +1): "Ammonium",
    ("NO3", -1): "Nitrate",
    ("NO2", -1): "Nitrite",
    ("CO3", -2): "Carbonate",
    ("HCO3", -1): "Bicarbonate",
    ("SO4", -2): "Sulfate",
    ("SO3", -2): "Sulfite",
    ("PO4", -3): "Phosphate",
    ("HPO4", -2): "Hydrogen phosphate",
    ("H2PO4", -1): "Dihydrogen phosphate",
    ("CN", -1): "Cyanide",
    ("SCN", -1): "Thiocyanate",
    ("HS", -1): "Hydrosulfide",
}

def _trivial_name(formula_ascii: Optional[str], net_charge: Optional[int]) -> Optional[str]:
    if not formula_ascii or net_charge is None:
        return None
    key = (formula_ascii, int(net_charge))
    name = _TRIVIAL.get(key)
    if name:
        return name
    if len(formula_ascii) == 2:
        return _TRIVIAL.get((formula_ascii[::-1], int(net_charge)))
    return None


def identify(
    *,
    smiles: Optional[str] = None,
    inchi: Optional[str] = None,
    inchikey: Optional[str] = None,
    molblock: Optional[str] = None,
    allow_online_names: bool = True,
    include_synonyms: bool = False,
    allow_online: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    
    issues: List[Dict[str, str]] = []

    if allow_online is not None:
        allow_online_names = bool(allow_online)
    if kwargs:
        _LOG.debug("identify(): ignoring unexpected kwargs: %s", list(kwargs.keys()))

    mol: Optional[Chem.Mol] = None
    input_type = None

    if molblock:
        mol = _mol_from_molblock(molblock, issues); input_type = "molblock"
    if mol is None and smiles:
        mol = _mol_from_smiles(smiles, issues); input_type = input_type or "smiles"
    if mol is None and inchi:
        m, used_inchi = _mol_from_inchi(inchi, issues)
        mol = m
        if used_inchi: input_type = input_type or "inchi"
    if mol is None and inchikey:
        input_type = input_type or "inchikey"

    if mol is None and input_type is None:
        issues.append(_issue("NO_INPUT", "No valid chemical input supplied. Provide SMILES, InChI, InChIKey, or a MOL/SDF block."))
        return _empty_identity("Unknown species", issues)

    return _identity_from_mol(
        mol=mol,
        input_type=input_type,
        provided_inchikey=inchikey,
        allow_online_names=allow_online_names,
        include_synonyms=include_synonyms,
        net_charge_hint=None,
        spin_mult_hint=None,
        issues=issues,
    )


def identify_from_atoms(
    atoms: Union[str, Iterable[Union[dict, tuple, list]]],
    *,
    net_charge: Optional[int] = None,
    spin_multiplicity: Optional[int] = None,
    allow_online_names: bool = False,
    allow_online: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    issues: List[Dict[str, str]] = []

    if allow_online is not None:
        allow_online_names = bool(allow_online)
    if kwargs:
        _LOG.debug("identify_from_atoms(): ignoring unexpected kwargs: %s", list(kwargs.keys()))

    mol, parse_issues = _mol_from_atoms(atoms)
    issues.extend(parse_issues)
    if mol is None:
        issues.append(_issue("ATOMS_PARSE_ERROR", "Could not construct a molecule from the provided atoms."))
        return _empty_identity("Unknown species", issues)

    if _HAS_RDB:
        try:
            if net_charge is not None:
                _RDB.DetermineBonds(mol, charge=net_charge)
            else:
                _RDB.DetermineBonds(mol)
        except TypeError:
            try:
                _RDB.DetermineBonds(mol)
            except Exception as e:
                issues.append(_issue("BOND_PERCEPTION_FAILED", f"Bond perception skipped: {e}"))
        except Exception as e:
            issues.append(_issue("BOND_PERCEPTION_FAILED", f"Bond perception skipped: {e}"))
    else:
        issues.append(_issue("BOND_PERCEPTION_UNAVAILABLE", "RDKit built without rdDetermineBonds; proceeding without perceived bonds."))

    ident = _identity_from_mol(
        mol=mol,
        input_type="atoms",
        provided_inchikey=None,
        allow_online_names=allow_online_names,
        include_synonyms=False,
        net_charge_hint=net_charge,
        spin_mult_hint=spin_multiplicity,
        issues=issues,
    )

    if net_charge is not None and ident.get("net_charge") != int(net_charge):
        ident["net_charge"] = int(net_charge)
        ident.setdefault("issues", []).append(_issue(
            "NET_CHARGE_HINT_USED", "Used supplied net_charge hint (formal charges could not be reliably inferred)."
        ))

    if spin_multiplicity is not None and ident.get("spin_multiplicity") != int(spin_multiplicity):
        ident["spin_multiplicity"] = int(spin_multiplicity)
        ident.setdefault("issues", []).append(_issue(
            "SPIN_MULT_HINT_USED", "Used supplied spin_multiplicity hint."
        ))

    return ident

def _identity_from_mol(
    *,
    mol: Optional[Chem.Mol],
    input_type: Optional[str],
    provided_inchikey: Optional[str],
    allow_online_names: bool,
    include_synonyms: bool,
    net_charge_hint: Optional[int],
    spin_mult_hint: Optional[int],
    issues: List[Dict[str, str]],
) -> Dict[str, Any]:
    
    if mol is not None:
        mol, frag_count, _ = _select_largest_fragment(mol)
        if frag_count > 1:
            issues.append(_issue(
                "MULTIFRAGMENT",
                f"Input contains {frag_count} fragments; using the largest fragment (by heavy-atom count)."
            ))
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            issues.append(_issue("SANITIZE_WARNING", f"Sanitization warning: {e}"))
        has_3d_out = any(conf.Is3D() for conf in mol.GetConformers())
        frag_count_out = frag_count
    else:
        has_3d_out = False
        frag_count_out = 0

    formula = _safe_formula(mol, issues) if mol is not None else None
    can_smiles = _safe_canonical_smiles(mol, issues) if mol is not None else None
    rdinchi, rdinchikey = _safe_inchi(mol, issues) if mol is not None else (None, None)
    if rdinchikey is None and provided_inchikey:
        rdinchikey = provided_inchikey.strip()

    computed_charge = _net_formal_charge(mol) if mol is not None else None
    computed_mult = _spin_multiplicity(mol) if mol is not None else None

    net_charge = int(net_charge_hint) if net_charge_hint is not None else computed_charge
    spin_mult = int(spin_mult_hint) if spin_mult_hint is not None else computed_mult

    species_kind = _classify_species(mol, net_charge) if mol is not None else None

    common_name = None
    iupac_name = None
    name_source = None

    if mol is not None and mol.GetNumAtoms() == 1:
        sym = mol.GetAtomWithIdx(0).GetSymbol()
        try:
            if _NAMES_AVAILABLE:
                elem = resolve_element_name(sym)
                common_name = elem.get("common_name")
                name_source = elem.get("name_source") or "mendeleev"
            else:
                common_name = sym
                name_source = "element-symbol"
        except Exception:
            common_name = sym
            name_source = "element-symbol"

    if (common_name is None and allow_online_names and _NAMES_AVAILABLE):
        try:
            names = resolve_molecule_names(
                inchikey=rdinchikey,
                smiles=can_smiles,
                inchi=rdinchi,
                timeout=4.0,
                include_synonyms=False,
            )
            common_name = names.get("common_name") or None
            iupac_name = names.get("iupac_name") or None
            name_source = names.get("name_source") or name_source
        except Exception as e:
            issues.append(_issue("NAMING_FAILED", f"Online name resolution failed: {e}"))

    if common_name is None and iupac_name is None:
        trivial = _trivial_name(formula, net_charge)
        if trivial:
            common_name = trivial
            name_source = name_source or "trivial-table"

    formula_pretty = _pretty_formula(formula) if formula else None
    display_name, display_source = _derive_display_name(
        common_name=common_name,
        iupac_name=iupac_name,
        element_name=(common_name if (mol is not None and mol.GetNumAtoms()==1) else None),
        formula_pretty=formula_pretty,
        formula_ascii=formula,
        net_charge=net_charge
    )
    final_name_source = name_source or display_source

    identity: Dict[str, Any] = {
        "formula": formula,
        "formula_pretty": formula_pretty,
        "smiles": can_smiles,
        "inchi": rdinchi,
        "inchikey": rdinchikey,
        "net_charge": net_charge,
        "spin_multiplicity": spin_mult,
        "species_kind": species_kind,
        "display_name": display_name,
        "input_type": input_type,
        "has_3d": bool(has_3d_out),
        "fragment_count": int(frag_count_out),
        "issues": issues,
    }
    if common_name:
        identity["common_name"] = common_name
    if iupac_name:
        identity["iupac_name"] = iupac_name
    if final_name_source:
        identity["name_source"] = final_name_source

    if mol is not None:
        try:
            if not AllChem.MMFFHasAllMoleculeParams(mol):
                identity.setdefault("issues", []).append(_issue(
                    "MMFF_UNSUPPORTED",
                    "MMFF94 parameters incomplete for this molecule; UFF is recommended for geometry."
                ))
        except Exception:
            pass
        try:
            if not AllChem.UFFHasAllMoleculeParams(mol):
                identity.setdefault("issues", []).append(_issue(
                    "UFF_UNSUPPORTED",
                    "UFF parameters incomplete (unusual element or valence). Some MM methods may fail."
                ))
        except Exception:
            pass

    return identity

def _norm_symbol(sym: str) -> str:
    s = (sym or "").strip()
    if not s:
        return s
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()

def _symbol_from_item(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        sym = item.get("symbol") or item.get("element") or item.get("name")
        if sym:
            return _norm_symbol(str(sym))
        z = item.get("atomicNumber") or item.get("Z") or item.get("z")
        if z is not None:
            try:
                z = int(z)
                if z > 0:
                    return _PT.GetElementSymbol(z)
            except Exception:
                return None
        return None
    if isinstance(item, (list, tuple)):
        if len(item) >= 1 and isinstance(item[0], str):
            return _norm_symbol(item[0])
    return None

def _coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

def _extract_coords_from_dict(d: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    for kx, ky, kz in (("x","y","z"), ("X","Y","Z")):
        if kx in d or ky in d or kz in d:
            return _coerce_float(d.get(kx)), _coerce_float(d.get(ky)), _coerce_float(d.get(kz))
    for key in ("coords","coord","xyz","r","position","pos","geometry","point"):
        v = d.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            xs = _coerce_float(v[0]) if len(v) >= 1 else None
            ys = _coerce_float(v[1]) if len(v) >= 2 else None
            zs = _coerce_float(v[2]) if len(v) >= 3 else None
            return xs, ys, zs
        if isinstance(v, dict):
            return _extract_coords_from_dict(v)
    return None, None, None

def _extract_coords(item: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if isinstance(item, dict):
        return _extract_coords_from_dict(item)
    if isinstance(item, (list, tuple)):
        if len(item) == 4 and isinstance(item[0], str):
            return _coerce_float(item[1]), _coerce_float(item[2]), _coerce_float(item[3])
        if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (list, tuple)):
            v = item[1]
            xs = _coerce_float(v[0]) if len(v) >= 1 else None
            ys = _coerce_float(v[1]) if len(v) >= 2 else None
            zs = _coerce_float(v[2]) if len(v) >= 3 else None
            return xs, ys, zs
        if len(item) == 3 and isinstance(item[0], str):
            return _coerce_float(item[1]), _coerce_float(item[2]), 0.0
    return None, None, None

def _mol_from_atoms(
    atoms: Union[str, Iterable[Union[dict, tuple, list]]]
) -> Tuple[Optional[Chem.Mol], List[Dict[str, str]]]:
    
    issues: List[Dict[str, str]] = []

    parsed: List[Tuple[str, float, float, float]] = []

    if isinstance(atoms, str):
        parsed, xyz_issues = _parse_xyz_block(atoms)
        issues.extend(xyz_issues)
    else:
        try:
            got_any = False
            for idx, item in enumerate(atoms):
                sym = _symbol_from_item(item)
                if not sym:
                    issues.append(_issue("ATOM_FORMAT", f"Atom {idx}: missing/unknown element in {item}"))
                    continue
                try:
                    _ = Chem.Atom(sym)
                except Exception:
                    issues.append(_issue("BAD_ELEMENT", f"Atom {idx}: unknown element symbol '{sym}'"))
                    continue

                x, y, z = _extract_coords(item)
                defaulted = False
                if x is None: x = 0.0; defaulted = True
                if y is None: y = 0.0; defaulted = True
                if z is None: z = 0.0; defaulted = True
                if defaulted:
                    issues.append(_issue("ATOM_COORD_DEFAULTED", f"Atom {idx} '{sym}': missing coords defaulted to (0,0,0)."))
                parsed.append((sym, float(x), float(y), float(z)))
                got_any = True

            if not got_any:
                issues.append(_issue("NO_ATOMS", "No valid atoms provided."))
                return None, issues

        except Exception as e:
            issues.append(_issue("ATOM_PARSE_ERROR", f"Failed to parse atoms: {e}"))
            return None, issues

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(parsed))
    conf.Set3D(True)

    for i, (sym, x, y, z) in enumerate(parsed):
        atom = Chem.Atom(sym)
        rw_idx = rw.AddAtom(atom)
        conf.SetAtomPosition(rw_idx, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))

    mol = rw.GetMol()
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol, issues


def _parse_xyz_block(xyz: str) -> Tuple[List[Tuple[str, float, float, float]], List[Dict[str, str]]]:
    issues: List[Dict[str, str]] = []
    lines = [ln.strip() for ln in (xyz or "").splitlines() if ln.strip()]
    if not lines:
        return [], [ _issue("XYZ_EMPTY", "Empty XYZ content.") ]
    start_idx = 0
    try:
        _ = int(lines[0])
        start_idx = 2 if len(lines) >= 2 else 1
    except Exception:
        start_idx = 0

    parsed: List[Tuple[str, float, float, float]] = []
    for ln in lines[start_idx:]:
        parts = ln.split()
        if len(parts) < 4:
            issues.append(_issue("XYZ_LINE", f"Malformed XYZ line: '{ln}'"))
            continue
        sym = _norm_symbol(parts[0])
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            issues.append(_issue("XYZ_COORDS", f"Non-numeric coordinates in line: '{ln}'"))
            continue
        parsed.append((sym, x, y, z))
    return parsed, issues



def _issue(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": str(message)}

def _empty_identity(display_name: str, issues: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "formula": None,
        "formula_pretty": None,
        "smiles": None,
        "inchi": None,
        "inchikey": None,
        "net_charge": None,
        "spin_multiplicity": None,
        "species_kind": None,
        "display_name": display_name,
        "input_type": None,
        "has_3d": False,
        "fragment_count": 0,
        "issues": issues,
    }

def _mol_from_smiles(smi: str, issues: List[Dict[str, str]]) -> Optional[Chem.Mol]:
    s = (smi or "").strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s, sanitize=True)
    if mol is None:
        issues.append(_issue("PARSE_ERROR", "Failed to parse SMILES."))
    return mol

def _mol_from_inchi(inchi: str, issues: List[Dict[str, str]]) -> Tuple[Optional[Chem.Mol], bool]:
    
    if not inchi:
        return None, False
    if not _INCHI_AVAILABLE:
        issues.append(_issue("INCHI_UNAVAILABLE", "RDKit built without InChI support; cannot parse InChI."))
        return None, False
    try:
        mol = Chem.MolFromInchi(inchi, treatWarningAsError=False)
    except Exception as e:
        issues.append(_issue("PARSE_ERROR", f"Failed to parse InChI: {e}"))
        return None, True
    if mol is None:
        issues.append(_issue("PARSE_ERROR", "Failed to parse InChI."))
        return None, True
    return mol, True

def _mol_from_molblock(molblock: str, issues: List[Dict[str, str]]) -> Optional[Chem.Mol]:
    block = (molblock or "").strip()
    if not block:
        return None
    try:
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
        if mol is None:
            issues.append(_issue("PARSE_ERROR", "Failed to parse MOL/SDF block."))
        return mol
    except Exception as e:
        issues.append(_issue("PARSE_ERROR", f"Failed to parse MOL/SDF block: {e}"))
        return None

def _select_largest_fragment(mol: Chem.Mol) -> Tuple[Chem.Mol, int, List[int]]:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return mol, 1, list(range(mol.GetNumAtoms()))
    if not frags:
        return mol, 1, list(range(mol.GetNumAtoms()))
    if len(frags) == 1:
        return frags[0], 1, list(range(frags[0].GetNumAtoms()))
    def heavy_count(m: Chem.Mol) -> int:
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
    best = max(frags, key=lambda m: (heavy_count(m), m.GetNumAtoms()))
    return best, len(frags), list(range(best.GetNumAtoms()))

def _safe_formula(mol: Optional[Chem.Mol], issues: List[Dict[str, str]]) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Descriptors.CalcMolFormula(mol)
    except Exception as e:
        issues.append(_issue("FORMULA_FAILED", f"Failed to compute formula: {e}"))
        return None

def _safe_canonical_smiles(mol: Optional[Chem.Mol], issues: List[Dict[str, str]]) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        issues.append(_issue("SMILES_FAILED", f"Failed to generate canonical SMILES: {e}"))
        return None

def _safe_inchi(mol: Optional[Chem.Mol], issues: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    if mol is None:
        return None, None
    if not _INCHI_AVAILABLE:
        return None, None
    try:
        inchi = MolToInchi(mol)
        ik = MolToInchiKey(mol) if inchi else None
        return inchi, ik
    except Exception as e:
        issues.append(_issue("INCHI_FAILED", f"Failed to compute InChI/InChIKey: {e}"))
        return None, None

def _net_formal_charge(mol: Optional[Chem.Mol]) -> Optional[int]:
    if mol is None:
        return None
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))

def _spin_multiplicity(mol: Optional[Chem.Mol]) -> Optional[int]:
    
    if mol is None:
        return None
    rad_e = sum(int(a.GetNumRadicalElectrons()) for a in mol.GetAtoms())
    return int(max(1, rad_e + 1))

def _classify_species(mol: Optional[Chem.Mol], net_charge: Optional[int]) -> Optional[str]:
    if mol is None:
        return None
    rad_e = sum(int(a.GetNumRadicalElectrons()) for a in mol.GetAtoms())
    if mol.GetNumAtoms() == 1:
        return "element"
    if (net_charge or 0) != 0 and rad_e > 0:
        return "ion-radical"
    if (net_charge or 0) != 0:
        return "ion"
    if rad_e > 0:
        return "radical"
    return "molecule"


_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _pretty_formula(formula: Optional[str]) -> Optional[str]:
 
    if not formula:
        return None
    out = []
    for ch in formula:
        if ch.isdigit():
            out.append(ch.translate(_SUBSCRIPT))
        else:
            out.append(ch)
    return "".join(out)

def _format_charge_unicode(q: Optional[int]) -> str:

    if not q:
        return ""
    sign = "⁺" if q > 0 else "⁻"
    mag = abs(q)
    if mag == 1:
        return sign
    supers = str(mag).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{supers}{sign}"

def _derive_display_name(
    *,
    common_name: Optional[str],
    iupac_name: Optional[str],
    element_name: Optional[str],
    formula_pretty: Optional[str],
    formula_ascii: Optional[str],
    net_charge: Optional[int],
) -> Tuple[str, Optional[str]]:

    if common_name:
        return common_name, None
    if iupac_name:
        return iupac_name, None
    if element_name:
        return element_name, "mendeleev"
    if formula_pretty:
        return f"{formula_pretty}{_format_charge_unicode(net_charge)}", "computed"
    if formula_ascii:
        if net_charge:
            sign = "+" if net_charge > 0 else "-"
            mag = abs(net_charge)
            return f"{formula_ascii}{mag if mag != 1 else ''}{sign}", "computed"
        return formula_ascii, "computed"
    return "Unknown species", "computed"
