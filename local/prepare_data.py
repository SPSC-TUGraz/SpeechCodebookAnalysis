# Author: Julian Linke (linke@tugraz.at)
# SPSC TU Graz (July 2023)

import os
import json
import argparse

def get_speaker_ids_and_lst_lines(corpus_dir):
    spk_ids = []
    lst_lines = []
    for spk in os.listdir(corpus_dir):
        spk_dir = os.path.join(corpus_dir, spk)
        if os.path.isdir(spk_dir):
            spk_ids.append(spk)
            for audio_file in os.listdir(spk_dir):
                if audio_file.endswith('.wav') or audio_file.endswith('.flac'):
                    uttID = audio_file.replace('.wav','').replace('.flac','')
                    audio_path = os.path.join(spk_dir, audio_file)
                    lst_line = f"{uttID} {audio_path}\n"
                    lst_lines.append(lst_line)
    return spk_ids, lst_lines

def process_DATA_directory(DATA_dir):
    spk_dict = {}
    lst_lines = []
    for corpus in os.listdir(DATA_dir):
        corpus_dir = os.path.join(DATA_dir, corpus)
        if os.path.isdir(corpus_dir):
            corpus_name = '_'.join(corpus.split('_')[1:])
            spk_ids, new_lst_lines = get_speaker_ids_and_lst_lines(corpus_dir)
            spk_dict[corpus_name] = spk_ids
            lst_lines.extend(new_lst_lines)
    return spk_dict, lst_lines

def write_output_files(output_lst_path, output_json_path, lst_lines, spk_dict):
    with open(output_lst_path, 'w') as f:
        f.writelines(lst_lines)

    with open(output_json_path, 'w') as f:
        json.dump(spk_dict, f, indent=4)

def main(output_lst_path, output_json_path, DATA_dir):
    spk_dict, lst_lines = process_DATA_directory(DATA_dir)
    #print(lst_lines)
    write_output_files(output_lst_path, output_json_path, lst_lines, spk_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate DATA.lst and DATA.json files.')
    parser.add_argument('--output_lst_path', required=True, help='Path to the output .lst file.')
    parser.add_argument('--output_json_path', required=True, help='Path to the output .json file.')
    parser.add_argument('--DATA_dir', required=True, help='Path to the DATA directory.')
    args = parser.parse_args()
    main(args.output_lst_path, args.output_json_path, args.DATA_dir)
