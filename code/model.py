import torch
import torch.nn as nn
from torch.autograd import Function


class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(0.0001)
        grad_input += grad_output
        return grad_input


class VggModel(nn.Module):

    def __init__(self, num_brands):
        super(VggModel, self).__init__()
        self.num_aspects = 2000
        self.embedding_size = 1024
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, self.embedding_size)
        self.fc_b = nn.Linear(self.num_aspects, self.num_aspects)
        self.l_relu = nn.LeakyReLU()
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.brand_embeddings = nn.Embedding(num_brands, self.num_aspects)
        self.aspects_embeddings = nn.Parameter(torch.randn(self.num_aspects, self.embedding_size), requires_grad=True)

    def forward(self, data):
        im_ft = self.fc2(self.l_relu(self.fc1(data['image'])))
        brand_weights = self.brand_embeddings(data['brand'])
        brand_weights = L1Penalty.apply(brand_weights)
        w_aspects = torch.mul(brand_weights.view(im_ft.shape[0], self.num_aspects, 1).expand(im_ft.shape[0], self.num_aspects, im_ft.shape[1]), self.aspects_embeddings.view(1, self.num_aspects, im_ft.shape[1]).expand(im_ft.shape[0], self.num_aspects, im_ft.shape[1]))
        prod = torch.bmm(w_aspects, im_ft.view(im_ft.shape[0], im_ft.shape[1], 1))
        prod = self.dropout(prod)
        return prod.mean(1)


class VggModelTruncated(nn.Module):

    def __init__(self, fc1, fc2):
        super(VggModelTruncated, self).__init__()

        self.fc1 = fc1
        self.fc2 = fc2
        self.l_relu = nn.LeakyReLU()

    def forward(self, data):
        return self.fc2(self.l_relu(self.fc1(data)))


class VggModelAspects(nn.Module):

    def __init__(self, brand_embeddings, aspect_embeddings):
        super(VggModelAspects, self).__init__()

        self.brand_embeddings = brand_embeddings
        self.aspects_embeddings = aspect_embeddings
        self.num_aspects = self.aspects_embeddings.shape[0]

    def forward(self, data):
        brand_weights = self.brand_embeddings(data)
        return torch.mul(brand_weights.view(data.shape[0], self.num_aspects, 1).expand(data.shape[0], self.num_aspects, self.aspects_embeddings.shape[1]),
                         self.aspects_embeddings.view(1, self.num_aspects, self.aspects_embeddings.shape[1]).expand(data.shape[0], self.num_aspects, self.aspects_embeddings.shape[1]))
