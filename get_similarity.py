from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from args import get_args
import random
import os
from youtube_dataloader import Youtube_DataLoader
from youcook_dataloader import Youcook_DataLoader
from model import Net
from metrics import compute_metrics, print_computed_metrics
from loss import MaxMarginRankingLoss
from gensim.models.keyedvectors import KeyedVectors
import pickle
from msrvtt_dataloader import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
from lsmdc_dataloader import LSMDC_DataLoader

args = get_args()
if args.verbose:
    print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

if not(args.youcook) and not(args.msrvtt) and not(args.lsmdc):
    print('Loading captions: {}'.format(args.caption_path))
    caption = pickle.load(open(args.caption_path, 'rb'))
    print('done')

print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')

if args.youcook:
    dataset = Youcook_DataLoader(
        data=args.youcook_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
elif args.msrvtt:
    dataset = MSRVTT_TrainDataLoader(
        csv_path=args.msrvtt_train_csv_path,
        json_path=args.msrvtt_train_json_path,
        features_path=args.msrvtt_train_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
elif args.lsmdc:
    dataset = LSMDC_DataLoader(
        csv_path=args.lsmdc_train_csv_path,
        features_path=args.lsmdc_train_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
else:
    dataset = Youtube_DataLoader(
        csv=args.train_csv,
        features_path=args.features_path_2D,
        features_path_3D=args.features_path_3D,
        caption=caption,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        we=we,
        we_dim=args.we_dim,
        n_pair=args.n_pair,
    )
dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=args.num_thread_reader,
    shuffle=False,
    batch_sampler=None,
    drop_last=True,
)
net = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    n_pair=args.n_pair,
    max_words=args.max_words,
    sentence_dim=args.sentence_dim,
)
net.train()
# Optimizers + Loss
loss_op = MaxMarginRankingLoss(
    margin=args.margin,
    negative_weighting=args.negative_weighting,
    batch_size=args.batch_size,
    n_pair=args.n_pair,
    hard_negative_rate=args.hard_negative_rate,
)

net.cuda()
loss_op.cuda()

if args.pretrain_path != '':
    net.load_checkpoint(args.pretrain_path)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.verbose:
    print('Starting training loop ...')

def TrainOneBatch(model, opt, data, save_dir):
    text = data['text'].cuda()
    video = data['video'].cuda()
    se = data['se']
    print(data['video_id'])
    video = video.view(-1, video.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])

    sim_matrix = model(video, text)
    to_save = {
        'video_id': data['video_id'],
        'starts': data['se'][0][0],
        'ends': data['se'][1][0],
        'sim_matrix': sim_matrix
    }
    fp = os.path.join(save_dir, str(data['video_id'][0])+".pkl")
    pickle.dump(to_save, open(fp, 'wb'))
    print("saved", fp)


sd = '/checkpoint/bkorbar/HowTo100M/similarity'
os.makedirs(sd, exist_ok=True)

print("DATA LEN", len(dataset))
for i_batch, sample_batch in enumerate(dataloader):
    batch_loss = TrainOneBatch(net, optimizer, sample_batch, sd)

