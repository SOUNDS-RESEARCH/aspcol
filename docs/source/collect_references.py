"""
Run this script to collect all references from the docs and write them out at the end of the front page of the documentation. 
"""

import pathlib
import os
import sys
import importlib
import numpy as np

def main():
    current_dir = pathlib.Path(__file__).parent
    package_dir = current_dir.parent.parent / "aspcol"
    output_file_path = current_dir / "references.rst"

    sys.path.append(str(package_dir))

    module_names = []
    for f in package_dir.iterdir():
        if f.suffix == ".py": #only search top level
            if f.stem != "__init__":
                module_names.append(f.stem)
    
    modules = [importlib.import_module(f"aspcol.{m}") for m in module_names]
    docs = [m.__doc__ for m in modules]
    
    all_refs = get_all_refs(docs)
    ids = get_ref_identifiers(all_refs)
    
    ref_list = extract_list_according_to_ids(all_refs, ids)
    ref_list = "".join(ref_list)

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"References\n----------\n{ref_list}")


def find_start_idxs(all_refs, ids):
    # Connect start idxs to IDs and put in dictionary
    # Put all start idxs (including duplicates) in a list
    id_start_idxs = {}
    all_start_idxs = []
    for ref_id in ids:
        search_start = 0
        while True:
            idx = all_refs.find(f"[{ref_id}]", search_start)
            if idx == -1:
                break
            search_start = idx + len(ref_id) + 2
            #str_to_check = str_to_check[idx + len(ref_id) + 2:]
            all_start_idxs.append(idx)
            if ref_id not in id_start_idxs:
                id_start_idxs[ref_id] = idx

    # Get end idxs for all start idxs
    # Manually correct the end idx of the last reference
    all_idxs = {}
    all_start_idxs = np.sort(all_start_idxs)
    for ref_id, start_idx in id_start_idxs.items():
        idx_diff = all_start_idxs - start_idx
        idx_diff[idx_diff <= 0] = int(1e8)
        end_idx = all_start_idxs[np.argmin(idx_diff)]

        if end_idx == 0:
            end_idx = len(all_refs)
        all_idxs[ref_id] = (start_idx, end_idx)

    return all_idxs
 

def extract_list_according_to_ids(all_refs, ids):
    all_idxs = find_start_idxs(all_refs, ids)
    all_idxs = dict(sorted(all_idxs.items()))
    #num_refs = len(ids)
    
    # ref_pos = [all_refs.find(ref_id) for ref_id in ids]
    # ref_pos.append(len(all_refs)+1) #to include the last reference
    # ref_pos = np.sort(ref_pos)
    # ref_pos -= 1 #to include the "[" in the reference
    ref_list = [all_refs[start_idx:end_idx] for ref_id, (start_idx, end_idx) in all_idxs.items()]
    
    #ref_list = [all_refs[ref_pos[i]:ref_pos[i+1]] for i in range(num_refs)]
    ref_list = [ref.replace("\n", "") for ref in ref_list]
    ref_list = [ref.strip() for ref in ref_list]
    ref_list = [f"{ref}\n\n" for ref in ref_list]
    return ref_list
            
def get_all_refs(docs):
    all_refs = []
    for txt in docs:
        split_txt = txt.split("References\n----------\n")
        if len(split_txt) > 1:
            references = split_txt[1]
            all_refs.append(references)
    full_ref_list = "\n".join(all_refs)
    return full_ref_list


def get_ref_identifiers(references):
    ids = []
    #references.split("[]")
    for str_part in references.split("["):
        if len(str_part) > 1:
            part_list = str_part.split("]")
            if len(part_list) > 1:
                id = part_list[0]
                if id != "link":
                    if id not in ids:
                        ids.append(id)
    return ids



if __name__ == "__main__":
    main()