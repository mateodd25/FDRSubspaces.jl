# This script downloads and preprocess the data for the single-cell RNA experiment.
# It closely follows scanpy tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html.

from scipy.io import savemat
import numpy as np
import scanpy as sc

def download_and_extract(url, download_path, extract_path):
    try:
        if not os.path.exists(extract_path):
            print("Downloading and extracting files...")
            wget.download(url, download_path)
            with tarfile.open(download_path) as compressed_file:
                compressed_file.extractall("data")
            print("\nDownload and extraction complete.")
            os.remove(download_path)
            print(f"Deleted the compressed file: {download_path}.")
        else:
            print("Folder already exists. Skipping download and extraction.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_and_preprocess_data(data_path):
    adata = sc.read_10x_mtx('data/filtered_gene_bc_matrices/hg19/',  
                            var_names='gene_symbols',                
                            cache=True) 
    adata.var_names_make_unique()
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)
    return adata.X.T

def save_matlab_file(data, output_path):
        # Save the matrix as a MATLAB .mat file
        savemat(output_path, {'data': data})
        print(f"Data saved to {output_path} successfully. Size = {data.shape}")
 

def main():
    url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    download_path = "data/pbmc3k_filtered_gene_bc_matrices.tar.gz"    
    extract_path = "data/filtered_gene_bc_matrices/"
    data_path = "data/filtered_gene_bc_matrices/hg19/"
    output_path = "data/preprocessed_rna_data.mat"
    sc.settings.verbosity == 0

    download_and_extract(url, download_path, extract_path)
    data = load_and_preprocess_data(data_path)
    save_matlab_file(data, output_path)
    
    
if __name__ == "__main__":
    main()
