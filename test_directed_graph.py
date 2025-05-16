# test_directed_graph.py
import shutil
import os
import sys
from pathlib import Path

# Attempt to set up import paths. This assumes the script might be run
# from the project root where 'salsaclrs' is a subdirectory, or 'salsaclrs' is installed.
try:
    # If run from workspace root, salsaclrs should be directly importable
    from salsaclrs.data import SALSACLRSDataset, SALSA_CLRS_DATASETS, er_probabilities
    from salsaclrs.sampler import Sampler # For type hinting or direct use if needed
except ImportError:
    # Fallback if the above doesn't work (e.g. script moved or specific CWD)

    print(f"Failed to perform initial import. Please ensure 'salsaclrs' package is in your PYTHONPATH, or run this script from a location where it can be found (e.g., SALSA-CLRS project root).")
    sys.exit(1)


def main():
    ALGORITHM = 'bfs'  # BFS typically uses ER graphs.
    SPLIT = 'val'      # 'val' split often has a simpler, fixed-size graph config.
    ROOT_DIR = './test_directed_dataset_temp_root' # Temporary directory for test data
    NUM_SAMPLES = 3    # Small number of samples for a quick test run.

    # Clean up any existing temporary directory from previous runs.
    if os.path.exists(ROOT_DIR):
        print(f"Cleaning up existing directory: {ROOT_DIR}")
        shutil.rmtree(ROOT_DIR)
    os.makedirs(ROOT_DIR, exist_ok=True)

    print("-------------------------------------------------------------------------")
    print(f"--- Test: Attempting to create a dataset with directed Erdos-Renyi graphs --- Targeting algorithm: '{ALGORITHM}', split: '{SPLIT}'")
    #print("The base Sampler class (in salsaclrs.sampler) contains an assertion:")
    #print("  `assert not directed, 'Directed graphs not supported yet.'`")
    #print("This assertion is intended to prevent the generation of directed graphs.")
    #print("\\nHowever, most specific sampler implementations (e.g., BfsSampler, DfsSampler)")
    #print("currently override the 'directed' parameter to False within their _sample_data methods,")
    #print("even if 'directed: True' is passed in graph_generator_kwargs.")
    #print("This test aims to demonstrate this behavior.\\n")

    # Use a base configuration for graph_generator_kwargs to ensure all necessary parameters (like 'n', 'p_range') are present.
    # For 'bfs' and 'val' split, the config is SALSA_CLRS_DATASETS["val"].
    # SALSA_CLRS_DATASETS["val"] = { "p_range": er_probabilities(16), "n": 16 }
    if SPLIT not in SALSA_CLRS_DATASETS or not isinstance(SALSA_CLRS_DATASETS[SPLIT], dict):
        print(f"Error: SALSA_CLRS_DATASETS['{SPLIT}'] is not a valid configuration dict.")
        print("Please check your salsaclrs.data.SALSA_CLRS_DATASETS definition.")
        # Fallback to a generic ER config if specific one is missing, for robustness of the script
        base_gg_kwargs = {"n": 16, "p_range": (0.1, 0.3)}
        print(f"Using fallback base_gg_kwargs: {base_gg_kwargs}")
    else:
        base_gg_kwargs = SALSA_CLRS_DATASETS[SPLIT].copy()


    # Now, create the modified kwargs for the attempt.
    attempted_gg_kwargs = base_gg_kwargs.copy()
    attempted_gg_kwargs['directed'] = True
    # Ensure 'k' is not present if we are forcing 'er' (it's a param for 'ws' graphs)
    # This makes the kwargs consistent for an ER graph generator.
    if 'k' in attempted_gg_kwargs:
        del attempted_gg_kwargs['k']

    print(f"Attempting to instantiate SALSACLRSDataset with these graph_generator_kwargs: {attempted_gg_kwargs}")
    print(f"A nickname is used to create a unique dataset path under: {os.path.join(ROOT_DIR, ALGORITHM, SPLIT, 'test_directed_er')}\\n")

    dataset_instance = None
    try:
        dataset_instance = SALSACLRSDataset(
            root=ROOT_DIR,
            split=SPLIT,
            algorithm=ALGORITHM,
            num_samples=NUM_SAMPLES,
            graph_generator="er",  # Explicitly specify Erdos-Renyi graph generator
            graph_generator_kwargs=attempted_gg_kwargs,
            nickname="test_directed_er" # Ensures a unique path for this test dataset
        )
        
        print("SALSACLRSDataset instance was created successfully.")
        # print("This indicates that the 'assert not directed' in Sampler._create_graph was NOT triggered during sampler initialization.")
        
        # Explanation for why the assertion was likely not triggered:
        if hasattr(dataset_instance, 'sampler') and dataset_instance.sampler is not None:
            actual_sampler_gg_kwargs = dataset_instance.sampler._graph_generator_kwargs
            print(f"  The internal Sampler was initialized with _graph_generator_kwargs: {actual_sampler_gg_kwargs}")
            if actual_sampler_gg_kwargs.get('directed') is True:
                print("    It correctly received 'directed: True' during its initialization, which is now respected.")
            else:
                # This case might occur if the default for a sampler is still False and not overridden by user.
                print("    It appears 'directed: True' was NOT part of the sampler's initial _graph_generator_kwargs, or it defaulted to False.")

        print("\\nNow, attempting to generate graph data by calling dataset.process()... _sample_data() will be called.")
        #print("During this step, the sampler's _sample_data() method will be called.")
        #print("The 'directed' parameter from graph_generator_kwargs should now be passed to _create_graph.")
        
        dataset_instance.process() # This forces data generation if not already present.
        
        print("dataset.process() completed.")
        # Try to get a sample to confirm.
        data_sample = dataset_instance.get(0)
        print(f"Successfully retrieved a data sample (e.g., data_sample.num_nodes: {data_sample.num_nodes}).")

        print("\\nInspecting the generated graph structure for directedness...")
        is_directed_graph = False
        if hasattr(data_sample, 'edge_index') and data_sample.edge_index is not None:
            edge_index = data_sample.edge_index
            if edge_index.shape[1] == 0:
                print("  Graph has no edges. Considered undirected by default.")
            else:
                # Check for asymmetry: if (u,v) exists, does (v,u) always exist?
                edges = set()
                for i in range(edge_index.shape[1]):
                    u, v = edge_index[0, i].item(), edge_index[1, i].item()
                    edges.add((u, v))
                
                found_asymmetric_edge = False
                for u, v in edges:
                    if (v, u) not in edges:
                        found_asymmetric_edge = True
                        print(f"  Found asymmetric edge: ({u}, {v}) exists, but ({v}, {u}) does not.")
                        break
                
                if found_asymmetric_edge:
                    is_directed_graph = True
                    print("  The graph appears to be DIRECTED based on its edge_index.")
                else:
                    print("  The graph appears to be UNDIRECTED (all edges are symmetric) or has a symmetric directed structure.")
        else:
            print("  Could not find 'edge_index' in the data sample to check for directedness.")


        print("\\nConclusion:")
        if is_directed_graph:
            print("The changes to the Sampler classes and the removal of the assertion appear to have successfully")
            print("allowed the generation of DIRECTED graphs when 'directed: True' is specified.")
        else:
            print("While no errors occurred, the generated graph does not appear to be directed based on edge_index inspection.")
            print("This could mean the graph generator (e.g., Erdos-Renyi with 'directed=True') still produced a symmetric graph,")
            print("or the graph was too small/sparse to exhibit clear asymmetry, or there's another subtlety.")
            print("Further investigation of the graph generation logic in 'networkx' or the sampler's _random_er_graph might be needed if directedness is strictly expected here.")

    except AssertionError as e: # This assertion should not be hit anymore
        if 'Directed graphs not supported yet' in str(e):
            print(f"\\nUNEXPECTED SUCCESS (for this test's premise): Caught the intended AssertionError: {e}")
            print("This implies that 'directed=True' *was* passed all the way to Sampler._create_graph.")
            print("This would only happen if the specific sampler's _sample_data method was modified OR")
            print("a new sampler type was used that does not override 'directed' to False.")
        else:
            print(f"\\nCaught an UNEXPECTED AssertionError: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\\nAn UNEXPECTED error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\\n--- Test finished ---")
        # Clean up the created temporary directory
        if os.path.exists(ROOT_DIR):
            print(f"Cleaning up test directory: {ROOT_DIR}")
            shutil.rmtree(ROOT_DIR)

if __name__ == '__main__':
    main() 