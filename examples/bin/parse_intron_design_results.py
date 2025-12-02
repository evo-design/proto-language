from glob import glob
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List


def parse_fname_evo2(fname: str) -> List[Dict[str, Any]]:
    """
    Parse the TopK results from Evo 2. Multiple runs per file.
    """
    if 'min-blood-only' in fname:
        specificity_type = 'min_blood_only'
    elif 'max-brain-min-blood' in fname:
        specificity_type = 'max_brain_min_blood'
    else:
        raise ValueError(f'Cound not find specificity type in filename {fname}')
    
    data_dicts = []
    
    curr_intron = None
    with open(fname, 'r') as f:
        while line_content := f.readline():

            if line_content.strip().startswith('sequence (intron): '):
                curr_intron = line_content.split()[-1]
                line_content = f.readline().strip()
                assert line_content.strip().startswith('sequence (right_flank): ')

                data_dict = {
                    'intron': curr_intron,
                    'generator': 'evo2',
                    'specificity_type': specificity_type,
                }
                while (line_content := f.readline()).strip().startswith('Construct '):
                    formatted = line_content.strip().replace(':', '').replace('_', '-').replace(' ', '_')
                    key = '_'.join(formatted.split('_')[:-1])
                    value = formatted.split('_')[-1]
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    data_dict[key] = value

                data_dict['energy'] = sum([
                    value for key, value in data_dict.items()
                    if (key.startswith('Construct_') and 'donor' not in key and 'acceptor' not in key and isinstance(value, float))
                ])

                data_dicts.append(data_dict)

    return data_dicts


def parse_fname_mcmc(fname: str, starting_point: str) -> Dict[str, Any]:
    """
    Parse the MCMC results from various starting points. Single run per file.
    """
    if 'min-blood-only' in fname:
        specificity_type = 'min_blood_only'
    elif 'min-brain-max-blood' in fname:
        specificity_type = 'min_brain_max_blood'
    else:
        specificity_type = 'max_brain_min_blood'

    curr_intron = None
    curr_energy = None
    data_dict = None
    with open(fname, 'r') as f:
        f.seek(0, 2)  # Go to end of file.
        file_size = f.tell()
        bytes_to_read = min(file_size, 200 * 100)
        f.seek(max(0, file_size - bytes_to_read))
        
        while line_content := f.readline():

            if line_content.strip().startswith('Iteration'):
                curr_energy = float(line_content.split()[4].rstrip(','))

            elif line_content.strip().startswith('sequence (intron): '):
                curr_intron = line_content.split()[-1]
                line_content = f.readline().strip()
                assert line_content.strip().startswith('sequence (right_flank): ')

                data_dict = {
                    'intron': curr_intron,
                    'energy': curr_energy,
                    'generator': f'mcmc_{starting_point}',
                    'specificity_type': specificity_type,
                }
                while (line_content := f.readline()).strip().startswith('Construct '):
                    formatted = line_content.strip().replace(':', '').replace('_', '-').replace(' ', '_')
                    key = '_'.join(formatted.split('_')[:-1])
                    value = formatted.split('_')[-1]
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    data_dict[key] = value

    return data_dict


if __name__ == '__main__':
    log_fnames = glob('intron_design_multi-context_*/intron_design_*.log')

    data = []
    for fname in tqdm(log_fnames):
        if fname.startswith('intron_design_multi-context_evo2'):
            data_dicts = parse_fname_evo2(fname)
            data += data_dicts

        elif fname.startswith('intron_design_multi-context_random'):
            data_dict = parse_fname_mcmc(fname, 'random')
            if data_dict:
                data.append(data_dict)

        else:
            starting_point = fname.split('/')[0].split('_')[-1]
            data_dict = parse_fname_mcmc(fname, starting_point)
            if data_dict:
                data.append(data_dict)

    df = pd.DataFrame(data)
    df.to_csv('intron_design_results.tsv', index=False, sep='\t')
