# This script downloads and preprocess the data for the single-cell RNA experiment.
# It closely follows scanpy tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html.

from scipy.io import savemat
import numpy as np
import scanpy as sc
import os
import requests
import tarfile


def download_and_extract(url, download_path, extract_path):
    try:
        if not os.path.exists(extract_path):
            print("Downloading and extracting files...")
            
            # Use requests with proper headers to avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("\nDownload complete. Extracting...")
            with tarfile.open(download_path) as compressed_file:
                compressed_file.extractall("data")
            print("Extraction complete.")
            os.remove(download_path)
            print(f"Deleted the compressed file: {download_path}.")
        else:
            print("Folder already exists. Skipping download and extraction.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def load_and_preprocess_data(data_path):
    adata = sc.read_10x_mtx(
        "data/filtered_gene_bc_matrices/hg19/", var_names="gene_symbols", cache=True
    )
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith(
        "MT-"
    )  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata, max_value=10)
    return adata.X.T


def save_matlab_file(data, output_path):
    # Save the matrix as a MATLAB .mat file
    savemat(output_path, {"data": data})
    print(f"Data saved to {output_path} successfully. Size = {data.shape}")


def main():
    # Try multiple URLs in case one is down
    urls = [
        "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
        "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
        "https://datasets.cellxgene.cziscience.com/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    ]
    
    download_path = "data/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    extract_path = "data/filtered_gene_bc_matrices/"
    data_path = "data/filtered_gene_bc_matrices/hg19/"
    output_path = "data/preprocessed_rna_data.mat"
    sc.settings.verbosity = 0  # Fix the verbosity setting
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Try downloading from different URLs
    success = False
    for i, url in enumerate(urls):
        print(f"Attempting download from URL {i+1}/{len(urls)}: {url}")
        try:
            download_and_extract(url, download_path, extract_path)
            success = True
            break
        except Exception as e:
            print(f"Failed with error: {e}")
            if i < len(urls) - 1:
                print("Trying next URL...")
            else:
                print("All URLs failed.")
    
    if not success:
        print("ERROR: Could not download data from any of the attempted URLs.")
        print("This might be due to network issues or server-side restrictions.")
        print("You can try:")
        print("1. Check your internet connection")
        print("2. Try running the script again later") 
        print("3. Manually download from: https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz")
        print("   and extract it to the data/ folder")
        return
        
    data = load_and_preprocess_data(data_path)
    save_matlab_file(data, output_path)


if __name__ == "__main__":
    main()
