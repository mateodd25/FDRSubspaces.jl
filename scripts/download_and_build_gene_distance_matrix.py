#!/usr/bin/env python3
import os
import wget
import tarfile
import glob
import numpy as np
from scipy.io import savemat
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO

def download_and_extract(url, download_path, extract_path):
    try:
        # Check if the folder already exists
        if not os.path.exists(extract_path):
            print("Downloading and extracting alignment files...")
            # Download the alignment file
            wget.download(url, download_path)
            # Open and extract the compressed file
            with tarfile.open(download_path) as compressed_file:
                compressed_file.extractall("data")
            print("\nDownload and extraction complete.")
            os.remove(download_path)
            print(f"Deleted the compressed file: {download_path}.")
        else:
            print("Folder already exists. Skipping download and extraction.")
    except Exception as e:
        print(f"An error occurred: {e}")

def lower_triangular_to_full(distance_matrix_list):
    # Determine the size of the matrix
    n = len(distance_matrix_list)
    # Initialize a square matrix of zeros
    full_matrix = np.zeros((n, n))
    # Fill in the lower and upper triangular parts of the matrix
    for i, row in enumerate(distance_matrix_list):
        for j, d_ij in enumerate(row):
            full_matrix[i, j] = d_ij
            if i != j:  # Avoid duplicating the diagonal
                full_matrix[j, i] = d_ij
    return full_matrix

def compute_and_save_distance_matrices(source_folder, target_folder, alignment_format="msf"):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    # Find all .msf files within the source folder recursively
    msf_files = glob.glob(os.path.join(source_folder, '**/*.msf'), recursive=True)
    # Initialize the DistanceCalculator with the 'identity' model
    calculator = DistanceCalculator('identity')
    for msf_file in msf_files:
        # Read the alignment from the .msf file
        alignment = AlignIO.read(msf_file, alignment_format)
        # Calculate the distance matrix
        distance_matrix = calculator.get_distance(alignment)
        if len(distance_matrix) < 100:
            continue
        # Convert the BioPython DistanceMatrix to a full symmetric NumPy array
        distance_matrix_np = lower_triangular_to_full(distance_matrix.matrix)
        # Construct the target .mat filename
        base_name = os.path.basename(msf_file)
        mat_filename = os.path.splitext(base_name)[0] + ".mat"
        mat_path = os.path.join(target_folder, mat_filename)
        # Save the distance matrix as a MATLAB .mat file
        savemat(mat_path, {'distance_matrix': distance_matrix_np})
        print(f"Distance matrix for {base_name} saved to {mat_path} successfully. Size = {distance_matrix_np.shape}")

def main():
    url = "http://www.lbgi.fr/balibase/BalibaseDownload/BAliBASE_R1-5.tar.gz"
    download_path = "data/BAliBASE_R1-5.tar.gz"
    extract_path = "data/bb3_release"
    target_folder = "data/mra_matrices"

    download_and_extract(url, download_path, extract_path)
    compute_and_save_distance_matrices(extract_path, target_folder)



if __name__ == "__main__":
    main()
