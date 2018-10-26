from pathlib import Path
import sys, time
import torch
import torch.optim as optim
import numpy as np
import json
from utils.clevr import load_vocab, ClevrDataLoaderNumpy, ClevrDataLoaderH5
from tbd.module_net_binary import TbDNet
import h5py
from tbd.module_net import load_tbd_net
from utils.generate_programs import load_program_generator, generate_programs

vocab = load_vocab(Path('data/vocab.json'))

model_path = Path('./models')
program_generator_checkpoint = 'program_generator.pt'

tbd_net_checkpoint = './models/clevr-reg.pt'
# tbd_net = load_tbd_net(tbd_net_checkpoint, vocab, feature_dim=(512, 28, 28))
# tbd_net_checkpoint = './binary_example-13.pt'
tbd_net = load_tbd_net(tbd_net_checkpoint, vocab, feature_dim=(1024, 14, 14))
program_generator = load_program_generator(model_path / program_generator_checkpoint)
BATCH_SIZE = 64
val_loader_kwargs = {
    'question_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_questions_query_ending.h5'),
    # 'feature_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_features_28x28.h5'),
    'feature_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_features.h5'),
    'batch_size': BATCH_SIZE,
    'num_workers': 1,
    'shuffle': False
}

# generate_programs('/mnt/fileserver/shared/datasets/CLEVR_v1/data/test_questions_query_ending.h5', program_generator, 'data/test_set', BATCH_SIZE)
# quit()

loader = ClevrDataLoaderH5(**val_loader_kwargs)
dataset_size = loader.dataset_size
NBATCHES = int(dataset_size / BATCH_SIZE)

answers_str = {23: 'metal', 27: 'rubber', 22: 'large', 29: 'sphere', 28: 'small', 19: 'cylinder', 17: 'cube',
               30:'yellow', 26:'red', 25: 'purple', 21:'green', 20:'gray', 18:'cyan',16:'brown',15:'blue'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def map_ans_binary(answers, out_types):

    length = 0
    for types in out_types:
        length += len(types)

    ''' Map the answers from the format Justin Johnson et al. use to our own format. '''
    # ans_tensor = torch.LongTensor(length).zero_()
    ans_tensor = torch.FloatTensor(length).zero_()
    # ans_tensor.fill_(0)
    k=0
    for i in range(len(out_types)):
        for j in range(len(out_types[i])):
            # print(answers_str[answers[i].item()])
            if answers_str[answers[i].item()] in out_types[i][j]:
                ans_tensor[k] = 1
                # break
            k += 1

    return ans_tensor

torch.set_grad_enabled(False)
num_correct, num_samples, num_relevant = 0, 0, 0
ib = 0
for batch in loader:
    _, _, feats, answers, programs = batch
    feats = feats.to(device)
    programs = programs.to(device)

    outs, out_types = tbd_net.forward_binary(feats, programs)
    mapped_ans = map_ans_binary(answers, out_types)
    preds = torch.FloatTensor(mapped_ans.shape[0]).zero_()
    k = 0
    for f in outs:
        p = []
        for i in range(f.shape[0]):
            p.append(float(torch.sum(f[i]).to('cpu')))

        imax = np.argmax(np.array(p))
        preds[k + imax] = 1.

        k += f.shape[0]

    # outs = torch.sigmoid(outs)
    # preds = (outs.to('cpu') > 0.5).type(torch.FloatTensor)
    # preds = preds.view(outs.shape[0])
    acc = mapped_ans * preds
    batch_num_correct = float(acc.sum())
    num_correct += batch_num_correct
    batch_num_samples = float(preds.size(0))
    num_samples += batch_num_samples
    batch_num_relevant = float(mapped_ans.sum())
    num_relevant += batch_num_relevant

    ib += 1

    sys.stdout.write('\r \rBatch: {}/{} Recall: {:.4f}/{:.4f} - {:.4f}'.format(ib, NBATCHES,
                                                                                           batch_num_correct,
                                                                                           batch_num_relevant,
                                                                                           batch_num_correct / batch_num_relevant))


print('\nTotal accuracy: {} / {} ({}%) Recall: {}'.format(num_correct, num_samples, 100 * num_correct / num_samples,
                                                             num_correct / num_relevant))

