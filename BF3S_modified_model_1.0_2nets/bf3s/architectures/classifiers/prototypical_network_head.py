import torch
import torch.nn as nn

import bf3s.architectures.classifiers.utils as cutils


def L2SquareDist(A, B, average=True):
    # input A must be:  [nB x Na x nC]
    # input B must be:  [nB x Nb x nC]
    # output C will be: [nB x Na x Nb]
    assert A.dim() == 3
    assert B.dim() == 3
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2)
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)

    # AB = A * B = [nB x Na x nC] * [nB x nC x Nb] = [nB x Na x Nb]
    AB = torch.bmm(A, B.transpose(1, 2))

    AA = (A * A).sum(dim=2, keepdim=True).view(nB, Na, 1)  # [nB x Na x 1]
    BB = (B * B).sum(dim=2, keepdim=True).view(nB, 1, Nb)  # [nB x 1 x Nb]
    # l2squaredist = A*A + B*B - 2 * A * B
    dist = AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB
    if average:
        dist = dist / nC

    return dist


class PrototypicalNetworkHead(nn.Module):
    def __init__(self, opt):
        super().__init__()
        scale_cls = opt["scale_cls"] if ("scale_cls" in opt) else 1.0
        learn_scale = opt["learn_scale"] if ("learn_scale" in opt) else True
        self.global_pooling = opt["global_pooling"] if ("global_pooling" in opt) else False
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=learn_scale
        )
    # def forward(self, features_test, base_ids=None, features_train=None, labels_train=None):
    def forward(self, features_test_1, features_train_1, features_test_2, features_train_2, labels_train):
        """Recognize novel categories based on the Prototypical Nets approach.

        Classify the test examples (i.e., `features_test`) using the available
        training examples (i.e., `features_test` and `labels_train`) using the
        Prototypical Nets approach.

        Args:
            features_test: A 3D tensor with shape
                [batch_size x num_test_examples x num_channels] that represents
                the test features of each training episode in the batch.
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the train features of each training episode in the batch.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the train labels (encoded as 1-hot vectors) of each training
                episode in the batch.

        Return:
            scores_cls: A 3D tensor with shape
                [batch_size x num_test_examples x nKnovel] that represents the
                classification scores of the test feature vectors for the
                nKnovel novel categories.
        """
        # if features_train_2 is not None:
        if features_train_1.dim() == 5:
            features_train_1 = cutils.preprocess_5D_features(features_train_1, self.global_pooling)
        if features_train_2.dim() == 5:
            features_train_2 = cutils.preprocess_5D_features(features_train_2, self.global_pooling)
        if features_test_1.dim() == 5:
            features_test_1 = cutils.preprocess_5D_features(features_test_1, self.global_pooling)
        if features_test_2.dim() == 5:
            features_test_2 = cutils.preprocess_5D_features(features_test_2, self.global_pooling)
        # print("proto net labels train test", labels_train)
        features_train = torch.cat([features_train_1, features_train_2], dim=2)
        features_test = torch.cat([features_test_1, features_test_2], dim = 2)
        # else:
        #   if features_train_1.dim() == 5:
        #       features_train_1 = cutils.preprocess_5D_features(features_train_1, self.global_pooling)
        #   if features_test_1.dim() == 5:
        #       features_test_1 = cutils.preprocess_5D_features(features_test_1, self.global_pooling)
        #   # print("proto net labels train test", labels_train)
        #   features_train = features_train_1
        #   features_test = features_test_1
        assert features_train.dim() == 3
        assert labels_train.dim() == 3
        assert features_test.dim() == 3
        assert features_train.size(0) == labels_train.size(0)
        assert features_train.size(0) == features_test.size(0)
        assert features_train.size(1) == labels_train.size(1)
        assert features_train.size(2) == features_test.size(2)

        # ************************* Compute Prototypes **************************
        labels_train_transposed = labels_train.transpose(1, 2)
        # Batch matrix multiplication:
        #   prototypes = labels_train_transposed * features_train ==>
        #   [batch_size x nKnovel x num_channels] =
        #       [batch_size x nKnovel x num_train_examples] *
        #       [batch_size * num_train_examples * num_channels]
        prototypes = torch.bmm(labels_train_transposed, features_train)
        # Divide with the number of examples per novel category.
        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        # ***********************************************************************
        scores_cls = -self.scale_cls * L2SquareDist(features_test, prototypes)
        # distances = L2SquareDist(features_test, prototypes)
        # print("distances", distances)
        # y_pred = (-distances).softmax(dim=1)
        # print("test prototypical network head", scores_cls)
        return scores_cls
        # print("y_pred", y_pred)
        # print("y_pred_size", y_pred.size())
        # return y_pred
    


def create_model(opt):
    return PrototypicalNetworkHead(opt)
