''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

from dataset import collate_fn_test, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=False,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-sp', required=False,
                        help='Source sequence profiles to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    '''
    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    sp_instances = test_src_word_insts = read_instances_from_file(
        opt.sp,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])
    '''
    model = torch.load(opt.model)

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=model['settings']['src_word2idx'],
            tgt_word2idx=preprocess_data['dict']['tgt_word2idx'],
            src_insts=preprocess_data['valid']['src'],
            sp_insts=preprocess_data['valid']['sp']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn_test)

    translator = Translator(opt)

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join(
                        [test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write(pred_line.replace(' ', '').replace('</s>', '') + '\n')
    print('[Info] Finished.')


if __name__ == "__main__":
    main()
