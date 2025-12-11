import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, RDConfig, rdFingerprintGenerator

# Add SA_Score module
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def compute_diversity_metrics(molecules, is_rdMol=False, radius=2, fp_size=2048):
    """
    Compute diversity metrics for a set of molecules using Morgan fingerprints.

    Parameters
    ----------
    molecules : iterable
        If is_rdMol is False: iterable of SMILES strings.
        If is_rdMol is True : iterable of RDKit Mol objects.
    is_rdMol : bool, optional
        Whether `molecules` already contains RDKit Mol objects.
    radius : int, optional
        Morgan fingerprint radius.
    fp_size : int, optional
        Bit vector size for the fingerprint.

    Returns
    -------
    dict
        {
            'n_valid_molecules': int,
            'avg_tanimoto_similarity': float,
            'avg_tanimoto_distance': float,
            'min_similarity': float,
            'max_similarity': float,
            'std_similarity': float,
            'pairwise_comparisons': int
        }
    """
    fps = []
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    for mol in molecules:
        if not is_rdMol:
            mol = Chem.MolFromSmiles(mol)
        if mol is None:
            continue
        fps.append(fp_gen.GetFingerprint(mol))

    n_valid = len(fps)
    if n_valid < 2:
        return {
            'n_valid_molecules': n_valid,
            'avg_tanimoto_similarity': np.nan,
            'avg_tanimoto_distance': np.nan,
            'min_similarity': np.nan,
            'max_similarity': np.nan,
            'std_similarity': np.nan,
            'pairwise_comparisons': 0,
        }

    similarities = []
    for i, fp1 in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp1, fps[i+1:])
        similarities.extend(sims)

    similarities = np.array(similarities)

    mean_sim = np.mean(similarities)

    return {
        'n_valid_molecules': n_valid,
        'avg_tanimoto_similarity': mean_sim,
        'avg_tanimoto_distance': 1.0 - mean_sim,
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'std_similarity': float(np.std(similarities)),
        'pairwise_comparisons': int(len(similarities)),
        'similarities': similarities.tolist(),
    }


def compute_sa(mol):
    """Compute SA score for a molecule"""
    return sascorer.calculateScore(mol)

def compute_qed(mol):
    """Compute QED for a molecule"""
    return QED.qed(mol)
    
    

# computing metrics
def compute_mol_metrics(df):
    """Compute molecule-level metrics (SA, QED, Mw, validity) for each molecule."""
    # Create RDKit mol objects, handle NA values
    if 'rdMol' not in df.columns:
        df['rdMol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)

    # Validity
    df['is_valid'] = df['rdMol'].notnull()
    
    # Filter to valid molecules for descriptor computation
    valid_df = df[df['is_valid']].copy()
    
    if len(valid_df) == 0:
        # Return df with NaN columns if no valid molecules
        for col in ['SA', 'QED', 'Mw']:
            df[col] = np.nan
        return df
    
    # Compute molecular descriptors
    valid_df['SA'] = valid_df['rdMol'].apply(compute_sa)
    valid_df['QED'] = valid_df['rdMol'].apply(compute_qed)
    valid_df['Mw'] = valid_df['rdMol'].apply(Descriptors.MolWt)
    valid_df['smiles_canonical'] = valid_df['rdMol'].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)
    
    # Merge back into original df (NaN for invalid rows)
    for col in ['SA', 'QED', 'Mw', 'smiles_canonical']:
        df[col] = np.nan
        df.loc[valid_df.index, col] = valid_df[col]
    
    return df


def compute_task_metrics(df):
    """Compute task-level summary metrics (validity, uniqueness, diversity) from molecule-level data."""
    n_total = len(df)
    n_valid = df['is_valid'].sum()
    validity = n_valid / n_total if n_total > 0 else np.nan
    
    # Filter to valid molecules
    valid_df = df[df['is_valid']].copy()
    
    if len(valid_df) == 0:
        return {
            'n_total': n_total,
            'n_valid': 0,
            'validity': 0.0,
            'n_unique': 0,
            'uniqueness': np.nan,
            'Objective_mean': np.nan,
            'Objective_std': np.nan,
            'SA_mean': np.nan,
            'SA_std': np.nan,
            'QED_mean': np.nan,
            'QED_std': np.nan,
            'Mw_mean': np.nan,
            'Mw_std': np.nan,
            'avg_tanimoto_similarity': np.nan,
            'diversity': np.nan,
            'std_similarity': np.nan,
            'pairwise_comparisons': 0,
        }, []
    
    # Uniqueness (based on canonical SMILES)
    valid_df['can_smiles'] = valid_df['rdMol'].apply(Chem.MolToSmiles)
    n_unique = valid_df['can_smiles'].nunique()
    uniqueness = n_unique / n_valid if n_valid > 0 else np.nan
    
    # Diversity metrics
    diversity_stats = compute_diversity_metrics(valid_df['rdMol'].tolist(), is_rdMol=True)
    
    # Summary statistics
    summary = {
        'n_total': n_total,
        'n_valid': int(n_valid),
        'validity': validity,
        'n_unique': int(n_unique),
        'uniqueness': uniqueness,
        'Objective_mean': valid_df['Objective'].mean(),
        'Objective_std': valid_df['Objective'].std(),
        'SA_mean': valid_df['SA'].mean(),
        'SA_std': valid_df['SA'].std(),
        'QED_mean': valid_df['QED'].mean(),
        'QED_std': valid_df['QED'].std(),
        'Mw_mean': valid_df['Mw'].mean(),
        'Mw_std': valid_df['Mw'].std(),
        'avg_tanimoto_similarity': diversity_stats['avg_tanimoto_similarity'],
        'diversity': diversity_stats['avg_tanimoto_distance'],
        'std_similarity': diversity_stats['std_similarity'],
        'pairwise_comparisons': diversity_stats['pairwise_comparisons'],
        # 'similarities': diversity_stats['similarities']
    }
    
    return summary, diversity_stats['similarities']



