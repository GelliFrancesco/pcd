from ndcg import ndcg_at_k
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
import torch
import pickle
from utils import build_brand_list, load_model, unique
from model import VggModelAspects, VggModelTruncated
from sklearn.metrics.pairwise import cosine_similarity


def test_ranking(model, testing_path, code_list, image_features, brands, gpu=None):
    brand_list = brands['username'].tolist()
    data = pd.read_csv(testing_path).values
    test_brands = unique(data[:, 0])
    test_posts = unique(data[:, 1])
    brand_dict = {d[1]: d[0] for d in data}

    asp_model = VggModelAspects(model.brand_embeddings, model.aspects_embeddings).eval()
    model_truncated = VggModelTruncated(model.fc1, model.fc2).eval()

    brand_ids = Variable(torch.LongTensor([brand_list.index(el) for el in test_brands]), volatile=True)
    image_features = Variable(torch.from_numpy(image_features[[code_list[el] for el in test_posts], :]), volatile=True)

    if gpu:
        asp_model.cuda(gpu)
        model_truncated.cuda(gpu)
        brand_ids.cuda(gpu)
        image_features.cuda(gpu)

    # Computing Aspect Features
    aspects = asp_model(brand_ids)
    aspects = aspects.permute((1, 0, 2)).mean(0)

    # Computing Post Features
    posts = model_truncated(image_features)

    aspects = aspects.data
    posts = posts.data
    if gpu:
        aspects.cpu()
        posts.cpu()
    aspects = aspects.numpy()
    posts = posts.numpy()

    #Computing similarity scores
    scores = cosine_similarity(aspects, posts)

    queries = []
    pbar = tqdm(total=len(scores))

    verticals = {b['username']: b['vertical'] for index, b in brands.iterrows()}
    for p in range(scores.shape[0]):

        # Computing evaluation metrics for a brand
        predictions = [(test_posts[j], scores[p, j], brand_dict[test_posts[j]]) for j in range(scores.shape[1])]
        s_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        pos = [v[1] for v in s_predictions if brand_list[p] == v[-1]]
        neg = [v[1] for v in s_predictions if brand_list[p] != v[-1]]
        comp = [v[1] for v in s_predictions if (brand_list[p] != v[-1]) & (verticals[brand_list[p]] == verticals[v[-1]])]

        sum = np.sum([len([el for el in neg if e > el]) for e in pos])
        rank_of_first_pos = zip(*s_predictions)[-1].index(brand_list[p])
        queries.append((rank_of_first_pos,
                        float(sum) / (len(pos) * len(neg)),
                        float(np.sum([len([el for el in comp if e > el]) for e in pos])) / (len(pos) * len(comp)),
                        ndcg_at_k([1 if brand_list[p] == v[-1] else 0 for v in s_predictions], 10),
                        ndcg_at_k([1 if brand_list[p] == v[-1] else 0 for v in s_predictions], 50)))
        pbar.update(1)

    pbar.close()
    queries = zip(*queries)
    return (
        np.median(queries[0]),   # MedR
        np.average(queries[1]),  # AUC
        np.average(queries[2]),  # cAUC
        np.average(queries[3]),  # NDCG@10
        np.average(queries[4])   # NDCG@50
    )


if __name__ == '__main__':

    data_path = '../data/'
    testing_path = data_path + 'testing/posts.csv'

    brands = build_brand_list(data_path + 'brand_list.csv')

    post_map_path = data_path + 'features/map_list.pickle'
    feature_path = data_path + 'features/features.npy'
    with open(post_map_path, 'r') as f:
        code_list = pickle.load(f)
    image_features = np.load(feature_path)

    model_path = '../model/vgg_model.dat'
    model = load_model(model_path, len(brands))
    model.eval()

    ranking_metrics = test_ranking(model, testing_path, code_list, image_features, brands)
    print 'MedR:', ranking_metrics[0]
    print 'AUC:', ranking_metrics[1]
    print 'cAUC:', ranking_metrics[2]
    print 'NDCG@10:', ranking_metrics[3]
    print 'NDCG@50:', ranking_metrics[4]
