# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional
import itertools
import re
import numpy as np
# from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose
# from language.folding_callbacks import FoldingResult
# from language.utilities import get_atomarray_in_residue_range


class EnergyTerm(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self, node, sequence: str) -> float:
        pass

############################
### NUCLEOTIDE FILTERING ###
############################

class ValidNucleotideCharacters(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, sequence: str) -> float:
        del node

        # return 1.0 if sequence only contains valid nucleotides (A,C,G,T), otherwise 0.0
        if re.search(r'[^ACGTacgt]', sequence):
            return 1.0 
        return 0.0

class GenomeLength(EnergyTerm):
    def __init__(self, target_length: int = 3000) -> None:
        super().__init__()
        self.target_length = target_length

    def compute(self, node, sequence: str) -> float:
        del node
        
        # return 1.0 if sequence length matches target length, otherwise 0.0
        if len(sequence) == self.target_length:
            return 0.0 
        return 1.0

class GCContent(EnergyTerm):
    def __init__(self, target_range: tuple = (30, 60)) -> None:
        super().__init__()
        self.min_gc = min(target_range)
        self.max_gc = max(target_range)
        
        # Validate range
        if self.min_gc < 0 or self.max_gc > 100:
            raise ValueError("GC content range must be between 0 and 100 percent.")

    def compute(self, node, sequence: str) -> float:
        del node
        
        seq = sequence.upper()
        seq_len = len(seq)
        
        # edge case
        if seq_len == 0:
            return 1.0
            
        # Count G and C nucleotides directly in one pass
        gc_count = 0
        for nt in seq:
            if nt in 'GC':
                gc_count += 1
                
        # Calculate GC percentage
        gc_percentage = (gc_count / seq_len) * 100
        
        # return 0.0 if GC content is within the desired range
        if self.min_gc <= gc_percentage <= self.max_gc:
            return 0.0
        else:
            # return a normalized score based on distance from acceptable range
            if gc_percentage < self.min_gc:
                deviation = (self.min_gc - gc_percentage) / self.min_gc
            else:
                deviation = (gc_percentage - self.max_gc) / (100 - self.max_gc)
            
            # Return a score that approaches 1 as deviation increases -- does this scoring make sense?
            return min(1.0, deviation)

class NucleotideHomopolymer(EnergyTerm):
    def __init__(self, max_length: int = 10) -> None:
        super().__init__()
        self.max_length = max_length

    def compute(self, node, sequence: str) -> float:
        del node
        
        sequence = sequence.upper()
        
        # Use a single regex pattern to find all homopolymers of any nucleotide
        # This matches any consecutive repeated nucleotide (A, C, G, or T)
        matches = re.findall(r'(A+|C+|G+|T+)', sequence)
        
        # Find the length of the longest homopolymer
        longest_homopolymer = max((len(match) for match in matches), default=0)
        
        # Return 0.0 if the longest homopolymer is within acceptable range
        if longest_homopolymer <= self.max_length:
            return 0.0
        else:
            # Use a logarithmic scale for scoring rather than assuming a fixed extreme value
            # This approach increases more slowly as homopolymer length grows
            # ln(excess/max_allowed)/ln(2) gives a score of 1.0 when excess length = 2*max_allowed

            # Valid scoring function?
            excess_length = longest_homopolymer - self.max_length
            log_ratio = np.log(1 + excess_length/self.max_length) / np.log(2)
            return min(1.0, log_ratio)

class DinucleotideFrequency(EnergyTerm):
    def __init__(self, freq_range: tuple = (0.03, 0.08)) -> None:
        super().__init__()
        self.min_freq = min(freq_range)
        self.max_freq = max(freq_range)

        # Precompute all dinucleotides
        self.dinucleotides = [''.join(pair) for pair in itertools.product('ACGT', repeat=2)]
        
    def compute(self, node, sequence: str) -> float:
        del node
        
        seq = sequence.upper()
        seq_len = len(seq)
        
        # Edge case
        if seq_len < 2:
            return 1.0
        

        dinucleotide_counts = {}
        total_count = 0
        for i in range(seq_len - 1):
            dinuc = seq[i:i+2]
            if all(nt in 'ACGT' for nt in dinuc):  # Only count valid dinucleotides
                dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
                total_count += 1
        
        # If no valid dinucleotides found
        if total_count == 0:
            return 1.0
            
        # Calculate frequencies and check if they're in range
        max_deviation = 0.0
        
        # Score based on deviation from target dinucleotide frequencies
        # Should we add deviations per dinucleotide instead?
        for dinuc in self.dinucleotides:
            freq = dinucleotide_counts.get(dinuc, 0) / total_count
            
            # Calculate deviation if outside acceptable range
            if freq < self.min_freq:
                deviation = (self.min_freq - freq) / self.min_freq
                max_deviation = max(max_deviation, deviation)
            elif freq > self.max_freq:
                deviation = (freq - self.max_freq) / (1.0 - self.max_freq)
                max_deviation = max(max_deviation, deviation)
        
        # Return max deviation
        return min(1.0, max_deviation)

class TetranucleotideUsage(EnergyTerm):
    def __init__(self, tetranucleotide: str = "GATC", tud_range: tuple = (0.8, 1.2)) -> None:
        super().__init__()
        self.tetranucleotide = tetranucleotide.upper()
        self.min_tud = min(tud_range)
        self.max_tud = max(tud_range)
        
        # Validate tetranucleotide input
        if len(self.tetranucleotide) != 4:
            raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")
    
    def compute(self, node, sequence: str) -> float:
        del node
        
        seq = sequence.upper()
        tetra = self.tetranucleotide
        
        # edge case
        total_bases = len(seq)
        if total_bases < 4:
            return 1.0
        
        # Calculate frequencies of each nucleotide in the whole sequence
        A_freq = seq.count("A") / total_bases
        C_freq = seq.count("C") / total_bases
        G_freq = seq.count("G") / total_bases
        T_freq = seq.count("T") / total_bases
        
        # Calculate tetranucleotide occurrences
        tetra_count = seq.count(tetra)
        
        # Calculate expected frequency using zero-order Markov method
        # There is a bug in the original code! 
        nucleotide_freqs = {'A': A_freq, 'C': C_freq, 'G': G_freq, 'T': T_freq}
        
        # Build expected frequency product based on nucleotides in the tetranucleotide
        tetra_expected_freq = 1.0
        for nt in tetra:
            if nt in nucleotide_freqs:
                tetra_expected_freq *= nucleotide_freqs[nt]
            else:
                # If invalid nucleotide, set to 0
                tetra_expected_freq = 0
                break
        tetra_expected_freq *= total_bases
        
        # Calculate TUD
        tetra_tud = tetra_count / tetra_expected_freq if tetra_expected_freq != 0 else 0
        
        # Score based on TUD range
        if self.min_tud <= tetra_tud <= self.max_tud:
            return 0.0
        else:
            # Calculate normalized deviation
            if tetra_tud < self.min_tud:
                deviation = (self.min_tud - tetra_tud) / self.min_tud
            else:
                deviation = (tetra_tud - self.max_tud) / self.max_tud
            

            # For all the above functions, should we take the min to ensure score is max 1.0?
            return min(1.0, deviation)
        

#####################
### ORF FILTERING ###
#####################



        
# class MaximizePTM(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         del node
#         return 1.0 - folding_result.ptm


# class MaximizePLDDT(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         del node
#         return 1.0 - folding_result.plddt


# class SymmetryRing(EnergyTerm):
#     def __init__(self, all_to_all_protomer_symmetry: bool = False) -> None:
#         super().__init__()
#         self.all_to_all_protomer_symmetry: bool = all_to_all_protomer_symmetry

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         protomer_nodes = node.get_children()
#         protomer_residue_ranges = [
#             protomer_node.get_residue_index_range() for protomer_node in protomer_nodes
#         ]

#         centers_of_mass = []
#         for start, end in protomer_residue_ranges:
#             backbone_coordinates = get_backbone_atoms(
#                 folding_result.atoms[
#                     np.logical_and(
#                         folding_result.atoms.res_id >= start,
#                         folding_result.atoms.res_id < end,
#                     )
#                 ]
#             ).coord
#             centers_of_mass.append(get_center_of_mass(backbone_coordinates))
#         centers_of_mass = np.vstack(centers_of_mass)

#         return (
#             float(np.std(pairwise_distances(centers_of_mass)))
#             if self.all_to_all_protomer_symmetry
#             else float(np.std(adjacent_distances(centers_of_mass)))
#         )


# def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
#     return atoms[
#         (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
#     ]


# def _is_Nx3(array: np.ndarray) -> bool:
#     return len(array.shape) == 2 and array.shape[1] == 3


# def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
#     assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#     return coordinates.mean(axis=0).reshape(1, 3)


# def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
#     assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#     m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
#     distance_matrix = np.linalg.norm(m, axis=-1)
#     return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


# def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
#     assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#     m = coordinates - np.roll(coordinates, shift=1, axis=0)
#     return np.linalg.norm(m, axis=-1)


# class MinimizeSurfaceHydrophobics(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         return hydrophobic_score(folding_result.atoms, start, end)


# _HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}


# def hydrophobic_score(
#     atom_array: AtomArray,
#     start_residue_index: Optional[int] = None,
#     end_residue_index: Optional[int] = None,
# ) -> float:
#     """
#     Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface
#     exposed. Typically, lower is better.
#     """

#     hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

#     if start_residue_index is None and end_residue_index is None:
#         selection_mask = np.ones_like(hydrophobic_mask)
#     else:
#         start_residue_index = 0 if start_residue_index is None else start_residue_index
#         end_residue_index = (
#             len(hydrophobic_mask) if end_residue_index is None else end_residue_index
#         )
#         selection_mask = np.array(
#             [
#                 i >= start_residue_index and i < end_residue_index
#                 for i in range(len(hydrophobic_mask))
#             ]
#         )

#     # TODO(scandido): Resolve the float/bool thing going on here.
#     hydrophobic_surf = np.logical_and(
#         selection_mask * hydrophobic_mask, sasa(atom_array)
#     )
#     # TODO(brianhie): Figure out how to handle divide-by-zero.
#     return sum(hydrophobic_surf) / sum(selection_mask * hydrophobic_mask)


# class MinimizeSurfaceExposure(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         return surface_ratio(folding_result.atoms, list(range(start, end)))


# class MaximizeSurfaceExposure(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         return 1.0 - surface_ratio(folding_result.atoms, list(range(start, end)))


# def surface_ratio(atom_array: AtomArray, residue_indices: List[int]) -> float:
#     """Computes ratio of atoms in specified ratios which are on the protein surface."""

#     residue_mask = np.array([res_id in residue_indices for res_id in atom_array.res_id])
#     surface = np.logical_and(residue_mask, sasa(atom_array))
#     return sum(surface) / sum(residue_mask)


# class MinimizeSurfaceExposure(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         return surface_ratio(folding_result.atoms, list(range(start, end)))


# class MaximizeSurfaceExposure(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         return 1.0 - surface_ratio(folding_result.atoms, list(range(start, end)))


# def surface_ratio(atom_array: AtomArray, residue_indices: List[int]) -> float:
#     """Computes ratio of atoms in specified ratios which are on the protein surface."""

#     residue_mask = np.array([res_id in residue_indices for res_id in atom_array.res_id])
#     surface = np.logical_and(residue_mask, sasa(atom_array))
#     return sum(surface) / sum(residue_mask)


# class MaximizeGlobularity(EnergyTerm):
#     def __init__(self) -> None:
#         super().__init__()

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         backbone = get_backbone_atoms(
#             folding_result.atoms[
#                 np.logical_and(
#                     folding_result.atoms.res_id >= start,
#                     folding_result.atoms.res_id < end,
#                 )
#             ]
#         ).coord

#         return float(np.std(distances_to_centroid(backbone)))


# def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
#     """
#     Computes the distances from each of the coordinates to the
#     centroid of all coordinates.
#     """
#     assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#     center_of_mass = get_center_of_mass(coordinates)
#     m = coordinates - center_of_mass
#     return np.linalg.norm(m, axis=-1)


# class MinimizeCRmsd(EnergyTerm):
#     def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
#         super().__init__()

#         self.template: AtomArray = template
#         self.backbone_only: bool = backbone_only
#         if self.backbone_only:
#             self.template = get_backbone_atoms(template)

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

#         if self.backbone_only:
#             atoms = get_backbone_atoms(atoms)

#         return crmsd(self.template, atoms)


# def crmsd(atom_array_a: AtomArray, atom_array_b: AtomArray) -> float:
#     # TODO(scandido): Add this back.
#     # atom_array_a = canonicalize_within_residue_atom_order(atom_array_a)
#     # atom_array_b = canonicalize_within_residue_atom_order(atom_array_b)
#     superimposed_atom_array_b_onto_a, _ = superimpose(atom_array_a, atom_array_b)
#     return float(rmsd(atom_array_a, superimposed_atom_array_b_onto_a).mean())


# class MinimizeDRmsd(EnergyTerm):
#     def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
#         super().__init__()

#         self.template: AtomArray = template
#         self.backbone_only: bool = backbone_only
#         if self.backbone_only:
#             self.template = get_backbone_atoms(template)

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

#         if self.backbone_only:
#             atoms = get_backbone_atoms(atoms)

#         return drmsd(self.template, atoms)


# def drmsd(atom_array_a: AtomArray, atom_array_b: AtomArray) -> float:
#     # TODO(scandido): Add this back.
#     # atom_array_a = canonicalize_within_residue_atom_order(atom_array_a)
#     # atom_array_b = canonicalize_within_residue_atom_order(atom_array_b)

#     dp = pairwise_distances(atom_array_a.coord)
#     dq = pairwise_distances(atom_array_b.coord)

#     return float(np.sqrt(((dp - dq) ** 2).mean()))


# def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
#     assert _is_Nx3(coordinates), "Coordinates must be Nx3."
#     m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
#     distance_matrix = np.linalg.norm(m, axis=-1)
#     return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


# class MatchSecondaryStructure(EnergyTerm):
#     def __init__(self, secondary_structure_element: str) -> None:
#         super().__init__()
#         self.secondary_structure_element = secondary_structure_element

#     def compute(self, node, folding_result: FoldingResult) -> float:
#         start, end = node.get_residue_index_range()

#         subprotein = folding_result.atoms[
#             np.logical_and(
#                 folding_result.atoms.res_id >= start,
#                 folding_result.atoms.res_id < end,
#             )
#         ]
#         sse = annotate_sse(subprotein)

#         return np.mean(sse != self.secondary_structure_element)
