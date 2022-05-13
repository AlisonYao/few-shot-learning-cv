import math
# from msilib import knownbits
import random

import numpy as np
import torch
import torchnet as tnt

class FewShotDataloader:
    def __init__(
            self,
            dataset,
            nKnovel=5,
            nKbase=-1,
            n_exemplars=1,
            n_test_novel=15 * 5,
            n_test_base=15 * 5,
            batch_size=1,
            num_workers=4,
            epoch_size=2000,
    ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (
            self.dataset.num_cats_base
            if (self.phase == "train" or self.phase == "trainval")
            else self.dataset.num_cats_novel
        )

        # assert 0 <= nKnovel <= max_possible_nKnovel
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase

        self.nKbase = nKbase
        self.n_exemplars = n_exemplars
        self.n_test_novel = n_test_novel
        self.n_test_base = n_test_base
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == "test") or (self.phase == "val")

    def sample_image_ids_from(self, cat_id, sample_size=1):

        assert cat_id in self.dataset.label2ind.keys()
        assert len(self.dataset.label2ind[cat_id]) >= sample_size
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sample_categories(self, cat_set, sample_size=1):

        if cat_set == "base":
            labelIds = self.dataset.labelIds_base
        elif cat_set == "novel":
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError(f"Not recognized category set {cat_set}")

        assert len(labelIds) >= sample_size
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        if nKbase > 0:
            Kbase = sorted(self.sample_categories("base", nKbase))
            return Kbase
        else:
            Knovel = sorted(self.sample_categories("novel", nKnovel))
            return Knovel

    def sample_test_and_train_examples_modified(self, categories, n_query_total, n_support):
        QuerySet = []
        SupportSet = []
        n_query = n_query_total // len(categories)

        if len(categories) > 0:

            for idx in range(len(categories)):
                img_ids = self.sample_image_ids_from(categories[idx], sample_size=(n_query + n_support))
                img_support = img_ids[:n_support]
                img_query = img_ids[n_support:]

                QuerySet += [(img_id, idx) for img_id in img_query]
                SupportSet += [(img_id, idx) for img_id in img_support]

        assert len(QuerySet) == n_query_total

        return QuerySet, SupportSet


    def sample_episode_modified(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        n_test_novel = self.n_test_novel
        n_test_base = self.n_test_base
        n_exemplars = self.n_exemplars

        categories = self.sample_base_and_novel_categories(nKbase, nKnovel)  # 64 和 5 都是id 都在suppory 和 query之外
        if nKnovel == 0:
            QuerySet, SupportSet = self.sample_test_and_train_examples_modified(categories, n_test_base,
                                                                                     n_exemplars)  # 从base class的要用来test的id
            return SupportSet, QuerySet, categories, nKbase
        else:
            QuerySet, SupportSet = self.sample_test_and_train_examples_modified(categories, n_test_novel,
                                                                                       n_exemplars)  # 从base class的要用来test的id
            return SupportSet, QuerySet, categories, nKnovel


    def create_examples_tensor_data(self, examples):

        images = torch.stack([self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        # print("labels dataloader test",labels)
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(_):

            SupportSet, QuerySet, Kall, nKbase = self.sample_episode_modified()
            Xe, Ye = self.create_examples_tensor_data(SupportSet)
            Xt, Yt = self.create_examples_tensor_data(QuerySet)
            Kall = torch.LongTensor(Kall)

            return Xe, Ye, Xt, Yt, Kall, nKbase  # Xe 和Ye evaluation

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=list(range(self.epoch_size)), load=load_function
        )
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True),
        )

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size

