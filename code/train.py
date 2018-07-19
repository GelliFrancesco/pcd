from model import VggModel
from torch.nn import MarginRankingLoss
import torch.optim as optim
from input_pipeline import BrandDataset
from transform import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle
import datetime
import os
from utils import build_brand_list
from test import test_ranking


def persist_model(model, path):
    torch.save(model.state_dict(), path)
    return


def train(gpu=None):

    # Loading post image features
    with open(post_map_path, 'r') as f:
        code_list = pickle.load(f)
    image_features = np.load(feature_path)

    brands = build_brand_list(data_path + 'brand_list.csv')
    brand_list = brands['username'].tolist()

    model = VggModel(len(brand_list))

    # Initializing PyTorch Dataloader
    db = BrandDataset(training_path, code_list, image_features, brand_list)
    dataloader = DataLoader(db, batch_size=256, shuffle=True, num_workers=0)

    loss_function = MarginRankingLoss(margin=0.3)
    if gpu:
        model.cuda(gpu)
        loss_function.cuda(gpu)

    optimizer_rel = optim.Adadelta(model.parameters(), lr=1)

    for epoch in range(20):
        for i_batch, sample_batched in enumerate(dataloader):

            model.zero_grad()
            image_pos = Variable(sample_batched['image_p'])
            image_neg = Variable(sample_batched['image_n'])
            brand = Variable(sample_batched['brand'])
            ones = Variable(torch.ones(image_pos.size()[0], 1))

            if gpu:
                image_pos.cuda(gpu)
                image_neg.cuda(gpu)
                brand.cuda(gpu)
                ones.cuda(gpu)

            # Forwarding the network for positive and negative samples
            out_pos = model({'image': image_pos, 'brand': brand})
            out_neg = model({'image': image_neg, 'brand': brand})

            loss = loss_function(out_pos, out_neg, ones)
            loss.backward()
            optimizer_rel.step()

            # Computing evaluation metrics on testing/validation set
            if (i_batch % eval_freq == 0) & (i_batch > 0):
                model.eval()
                test = test_ranking(model, testing_path, code_list, image_features, brands, gpu)
                model.train()

                persist_model(model, experiment_folder + '/vgg_model.dat')
                print 'Epoch:', epoch, 'batch', i_batch, \
                    'Tr_Loss:', loss.item(), \
                    'Testing MedR:', test[0], \
                    'Testing AUC:', test[1], \
                    'Testing cAUC:', test[2], \
                    'Testing NDCG@10:', test[3], \
                    'Testing NDCG@50:', test[4]
            else:
                print 'Epoch:', epoch, 'batch', i_batch, 'Tr_Loss:', loss.item()
        persist_model(model, experiment_folder + '/vgg_model_ep_' + str(epoch) + '.dat')

    # Performing final evaluation
    model.eval()
    test = test_ranking(model, testing_path, code_list, image_features, brands, gpu)
    model.train()
    persist_model(model, model_path)
    print 'Final Result: ', \
        'MedR:', test[0], \
        'AUC:', test[1], \
        'cAUC:', test[2], \
        'NDCG@10:', test[3], \
        'NDCG@50:', test[4]
    return


if __name__ == '__main__':

    data_path = '../data/'
    model_path = '../model/vgg_model.dat'
    training_path = data_path + 'training/neg_samples.csv'
    testing_path = data_path + 'testing/posts.csv'

    post_map_path = data_path + 'features/map_list.pickle'
    feature_path = data_path + 'features/features.npy'

    experiment_time = str(datetime.datetime.now()).replace(' ', '_')
    experiment_folder = 'log/' + experiment_time

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        print 'Experiment log folder is: ' + experiment_folder
        os.makedirs(experiment_folder + '/losses')

    eval_freq = 2000  # How often evaluation metrics are to be performed during training
    # torch.manual_seed(10)
    # torch.cuda.manual_seed(10)

    train()
