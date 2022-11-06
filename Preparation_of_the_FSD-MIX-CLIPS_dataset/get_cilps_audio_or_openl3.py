"""
-------------------------------File info-------------------------
% - File name: get_cilps_audio_or_openl3.py
% - Description:
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-05-24
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import os
from os import makedirs
from os.path import isfile, join, isdir
import pandas as pd
import pickle as pkl
import argparse

# import openl3
import soundfile as sf
from tqdm import tqdm


def get_data_and_filelists(annfile, audiofolder, savefolder, overwrite=False):
    # load pre-trained openl3 audio embedding model
    # embedding_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
    #                                                            embedding_size=512)

    filelist = {'data_path': [], 'labels': []}

    ann = pd.read_csv(annfile)

    for idx in tqdm(range(len(ann))):
        fname = ann['filename'][idx]
        start_time = ann['start_time'][idx]

        labels = [int(x) for x in ann['labels'][idx][1:-1].split(',')]  # convert string to list of int

        # -
        start_sample = int(start_time * 44100)
        # - fname e.g. soundscape_1667.wav  -->soundscape_1667_89082.pkl

        if args.data_type == 'audio':
            outfile = join(savefolder, fname.replace('.wav', '_' + str(start_sample) + '.wav'))
        # elif args.data_type == 'openl3':
        #     outfile = join(savefolder, fname.replace('.wav', '_' + str(start_sample) + '.pkl'))
        else:
            raise Exception('Incorrect data_type.')

        if not isfile(outfile) or overwrite:

            audio, sr = sf.read(join(audiofolder, fname))  # -
            audio = audio[start_sample:start_sample + 44100]  # -
            if args.data_type == 'audio':
                sf.write(outfile, audio, sr)  # -

            # elif args.data_type == 'openl3':
            #
            #     emb, ts = openl3.get_audio_embedding(audio, sr, model=embedding_model,
            #                                          center=False, verbose=0)  # -
            #     emb = emb.squeeze()  # - (1,512) --> (512,)
            #
            #     with open(outfile, 'wb') as f:
            #         pkl.dump(emb, f, protocol=pkl.HIGHEST_PROTOCOL)

            else:
                raise Exception('Incorrect data_type.')

            # -
            filelist['data_path'].append(outfile)
            filelist['labels'].append(labels)

    return filelist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annpath', type=str, required=True, help='path to FSD_MIX_CLIPS.annotations_revised folder')
    parser.add_argument('--audiopath', type=str, required=True, help='path to generated FSD_MIX_SED audio folder')
    parser.add_argument('--savepath', type=str, required=True, help='path to save the output samples)')
    parser.add_argument('--data_type', type=str, required=True, help='audio or openl3)')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    full_filelist_dir = savefolder = join(args.savepath, args.data_type, 'full_filelist')
    if not isdir(savefolder):
        makedirs(savefolder)

    # -
    for class_split in ['base', 'val', 'test']:
        if class_split == 'base':
            for data_split in ['train', 'val', 'test']:
                annfile = join(args.annpath, class_split + '_' + data_split + '.csv')  # -
                audiofolder = join(args.audiopath, class_split, data_split)  # -
                savefolder = join(args.savepath, args.data_type, class_split, data_split)
                if not isdir(savefolder):
                    makedirs(savefolder)

                filelist = get_data_and_filelists(annfile, audiofolder=audiofolder,
                                                  savefolder=savefolder, overwrite=False)
                filelist_dir = join(full_filelist_dir, class_split + '_' + data_split + '_filelist.pkl')
                with open(filelist_dir, 'wb') as f:
                    pkl.dump(filelist, f, protocol=pkl.HIGHEST_PROTOCOL)
                print(f'Done with {class_split}-{data_split}')
            print(f'Done with {class_split}.')
        else:
            annfile = join(args.annpath, 'novel_' + class_split + '.csv')
            audiofolder = join(args.audiopath, class_split)
            #
            savefolder = join(args.savepath, args.data_type, 'novel_' + class_split)
            if not isdir(savefolder):
                makedirs(savefolder)

            filelist = get_data_and_filelists(annfile, audiofolder=audiofolder,
                                              savefolder=savefolder, overwrite=False)

            filelist_dir = join(full_filelist_dir, 'novel_' + class_split + '_filelist.pkl')
            with open('novel_' + class_split + '_filelist.pkl', 'wb') as f:
                pkl.dump(filelist, f, protocol=pkl.HIGHEST_PROTOCOL)
            print(f'Done with {class_split}.')

    print(f'All Done.')
