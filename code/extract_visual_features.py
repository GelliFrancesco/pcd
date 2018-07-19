import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from input_pipeline import FeatureExtraction
from torchvision import transforms
from transform import *
from torch.autograd import Variable
from tqdm import tqdm


batch_size = 256
model = models.vgg16_bn(pretrained=True)

# Removing the last fully-connected layer
new_classifier = nn.Sequential(list(model.classifier.children())[0])
model.classifier = new_classifier

for param in model.parameters():
    param.requires_grad = False

model.eval()


def feature_extraction():
    if gpu:
        model.cuda(gpu)

    db = FeatureExtraction(training_path, testing_path, image_folder_path, post_map_path, transform=transforms.Compose([
                                                   Rescale(256), RandomCrop(224), ToTensor()]))

    dataloader = DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=16)

    feature_matrix = np.zeros((len(db.input_points), 4096))
    pbar = tqdm(total=len(db.input_points))

    for i_batch, sample_batched in enumerate(dataloader):
        batch_input = Variable(sample_batched['image'])
        if gpu:
            batch_input.cuda(gpu)

        out = model(batch_input)
        out = out.float().data
        if gpu:
            out = out.cpu()

        feature_matrix[i_batch*batch_size:min((i_batch+1)*batch_size, feature_matrix.shape[0]), :] = out.numpy()
        pbar.update(batch_size)
    pbar.close()

    with open(feature_path, 'w') as w:
        np.save(w, feature_matrix)


if __name__ == '__main__':
    gpu = None

    data_path = '../data/'
    training_path = data_path + 'training/posts.csv'
    testing_path = data_path + 'testing/posts.csv'
    image_folder_path = '<image dataset directory>'

    post_map_path = data_path + 'features/map_list.pickle'
    feature_path = data_path + 'features/features.npy'

    feature_extraction()


