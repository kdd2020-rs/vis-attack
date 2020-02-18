import os
import numpy as np

from dataset import RecSysDataset, read_item_data, read_user_item_data
from models import VBPR, DeepStyle, ImageModel

import numbers
import random

import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from PIL import Image

import pickle, json, time

from argparse import ArgumentParser

from numpy.linalg import norm
import scipy.stats

from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def to_img(x):
    x = x.squeeze(0).type(torch.DoubleTensor)
    x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    x = np.transpose(x*255, (1, 2, 0))
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

from sklearn import linear_model, svm

def solve(W, y):
    # x, _, _, _ = np.linalg.lstsq(W, y, rcond=None)
    x = linear_model.LinearRegression().fit(W, y).coef_
    return x

def pad_resize(im, desired_size=224):
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    return new_im

class TorchPCA:
    def __init__(self, weight, bias):
        self.W = torch.tensor(weight.T, dtype=torch.float32)
        self.b = torch.tensor(bias, dtype=torch.float32)

    def transform(self, X):
        return torch.mm(X-self.b, self.W)

    @staticmethod
    def get_pca(X, n_components):

        from sklearn.decomposition import PCA

        filename = f"../data/pca_{n_components}.pkl"

        if not os.path.exists(filename):
            pca = PCA(n_components=n_components)
            pca.fit(X)

            W, b = pca.components_, pca.mean_

            with open(filename, "bw") as f:
                pickle.dump((W, b), f) 
        else:
            with open(filename, "br") as f:
                W, b = pickle.load(f)
        
        return TorchPCA(W, b)

class JsonLooger():
    import json
    
    def __init__(self, logfile):
        self.logfile = logfile
        
    def __enter__(self):
        self.fd = open(self.logfile, "w")
        return self
    
    def __exit__(self, type, value, traceback):
        self.fd.close()
        
    def log(self, d):
        json.dump(d, self.fd)
        self.fd.write("\n")

class Experimentation:
    def __init__(self, model, dataset_name):

        dataset_path = f"../data/dataset/{dataset_name}"

        items_data, items_idx = read_item_data(dataset_path)
        user_items, _ = read_user_item_data(dataset_path)

        self.model = model
        self.image_model = ImageModel('resnet50')
        self.user_items = user_items
        self.items_data = items_data
        self.items_idx = items_idx

        self.n_users = model.n_users
        self.n_items = model.n_items

        self.image_folder = dataset_path + "/images"

    def search(self, u, n=10):
        res = self.model.score(u)
        res = res.argsort(dim=0, descending=True).squeeze().numpy()
        return res[:n]

    def load_image(self, i):
        name = self.items_data[i][2]
        filename = os.path.join(self.image_folder, name)
        img = Image.open(filename)
        return pad_resize(img)

    def get_image_features(self, img):
        return self.image_model.get_features(img)/60

    def get_rank(self, u_scores, score):
        return int(np.searchsorted(-u_scores, -score)) + 1

    def run_wb_single_user(self, args, logger):

        u = args.user
        steps = args.steps
        epsilon = int(args.epsilon * 255)

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1].item()

        print(f"Run single user, white-box, for user {u} from rank {from_rank}")

        logger.log({
            'user': u,
            'item': t,
            'rank': from_rank,
        })

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        img = self.load_image(t)
        img_M = np.array(img)

        for step in range(steps):
            t_img_tran = self.image_model.transform(img)
            t_img_tran = torch.unsqueeze(t_img_tran, 0)
            t_img_var = nn.Parameter(t_img_tran)
            t_feat = self.get_image_features(t_img_var)

            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([u], [t], t_feat)
            loss.backward(retain_graph=True)

            grad = t_img_var.grad.data
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, rank={rank}")

            logger.log({
                'step': step,
                'rank': rank,
            })

            if rank == 1:
                break
            
        return img

    def run_wb_single_user_o(self, args, logger):

        u = args.user
        steps = args.steps
        epsilon = args.epsilon

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1]

        print(f"Run single user, white-box, for user {u} from rank {from_rank}")

        t_img = self.load_image(t)

        t_img_tran = self.image_model.transform(t_img)
        t_img_tran = torch.unsqueeze(t_img_tran, 0)
        t_img_var = nn.Parameter(t_img_tran)
        t_feat = self.get_image_features(t_img_var)

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        for step in range(steps):
            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([u], [t], t_feat)
            score = -loss.item()
            rank = self.get_rank(u_scores, score)
            print(f"Step: {step+1}, Rank: {rank}")

            logger.log({
                'step': step+1,
                'rank': rank,
            })

            if rank == 1:
                break

            loss.backward(retain_graph=True)
            grad = torch.sign(t_img_var.grad.data)
            adversarial = t_img_var.data - epsilon * grad
            t_img_var.data = adversarial
            t_feat = self.get_image_features(t_img_var)

        return Image.fromarray(to_img(adversarial))

    def bb_attack(
            self, u, t, img, img_M, ori_img_M, n_examples, n_features, u_scores, 
            do_pca, pca_fc, by_rank, rank_distribution, epsilon, eps, attack="ifgsm"
        ):
        img_t = self.image_model.transform(img)
        img_t = img_t.unsqueeze(0)
        var_t = nn.Parameter(img_t)
        feat_t = self.get_image_features(var_t)

        y_t = self.model.score_user_item(u, t, feat_t.detach()).item()

        rank = self.get_rank(u_scores, y_t)

        if do_pca:
            pca_t = pca_fc.transform(feat_t)

        def score_rank(rank):
            quantile = 1-rank/(self.n_items+1)
            if rank_distribution == "normal":
                return scipy.stats.norm.ppf(quantile)
            return quantile

        if by_rank:
            s_t = score_rank(rank)

        W = np.zeros((n_examples, n_features), dtype=np.float32)
        y = np.zeros(n_examples, dtype=np.float32)

        noise = 7

        for i in tqdm(range(n_examples)):
            d = np.random.choice(range(-noise, noise+1), size=img_M.shape)
            img_i = np.clip(img_M.astype(np.int) + d, 0, 255).astype(np.uint8)
            img_i = Image.fromarray(img_i)

            img_i = self.image_model.transform(img_i)
            feat_i = self.get_image_features(img_i.unsqueeze(0))
            if do_pca:
                pca_i = pca_fc.transform(feat_i)

            y_i = self.model.score_user_item(u, t, feat_i).item()

            if by_rank:
                s_i = score_rank(self.get_rank(u_scores, y_i))

            diff = pca_i-pca_t if do_pca else feat_i-feat_t

            W[i] = diff.detach().numpy()
            y[i] = s_t-s_i if by_rank else y_t-y_i

        x = solve(W, y)
        
        if do_pca:
            pca_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)
        else:
            feat_t.backward(torch.tensor(x).view(1, n_features), retain_graph=True)

        grad = var_t.grad

        # fgsm, ifgsm, pgd, first
        if attack == "first":
            grad = grad.squeeze(0).numpy().transpose((1, 2, 0))
            grad /= norm(grad)
            grad = np.clip(grad*255, -epsilon, epsilon)
            img_M = np.clip(img_M.astype(np.int) - grad, 0, 255).astype(np.uint8)
        elif attack == "ifgsm":
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)
        elif attack == "pgd":
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255)
            eta = np.clip(img_M - ori_img_M, -eps, eps)
            img_M = np.clip(ori_img_M+eta, 0, 255).astype(np.uint8)

        return img_M
        
    def run_bb_single_user(self, args, logger):
        u = args.user
        do_pca = args.do_pca
        by_rank = args.by_rank
        rank_distribution = "uniform"
        if by_rank:
            rank_distribution = args.rank_distribution
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        n_examples = args.examples

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1].item()

        print(f"Run single user, black-box, for user {u} from rank {from_rank}")

        logger.log({
            'user': u,
            'item': t,
            'rank': from_rank,
        })

        backup = self.model.F[t].unsqueeze(0).clone().detach()

        n_features = backup.shape[1]

        pca_fc = None
        if do_pca:
            n_components = args.n_components
            pca_fc = TorchPCA.get_pca(self.model.F.numpy(), n_components)
            n_features = n_components
        
        img = self.load_image(t)
        img_M = np.array(img)

        if not os.path.exists(f"images/{t}"):
            os.makedirs(f"images/{t}")
        img.save(f"images/{t}/original.jpeg", "JPEG")

        ori_img_M = img_M
        eps = 10

        for step in range(steps):

            img_M = self.bb_attack(
                u, t, img, img_M, ori_img_M, n_examples, n_features, u_scores, 
                do_pca, pca_fc, by_rank, rank_distribution, epsilon, eps
            )

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))
            score = self.model.score_user_item(u, t, feat_t.detach()).item()
            rank = self.get_rank(u_scores, score)

            print(f"Step: {step+1}, rank={rank}")

            logger.log({
                'step': step,
                'rank': rank,
            })

            img.save(f"images/{t}/step_{step}.jpeg", "JPEG")

            if rank == 1:
                break
    
        return img

    
    def run_bb_segment_attack(self, args, logger):
        do_pca = args.do_pca
        by_rank = args.by_rank
        if by_rank:
            rank_distribution = args.rank_distribution
        steps = args.steps
        epsilon = int(args.epsilon * 255)
        n_examples = args.examples
        i = args.item

        while True:
            users = [u for u, items in enumerate(self.user_items) if i in items]
            if len(users) >= 20:
                break
            i += 1

        random_users = random.sample(range(self.model.n_users), k=100)

        args.user = user = self.model.add_fake_user_by_item(i)

        user_scores = self.model.score(user).detach().numpy().squeeze()
        user_scores.sort()
        user_scores = np.flip(user_scores)
        
        t = random.choice(range(self.model.n_items))

        print(f"Segment bb experiment for user={user}, item={t}")
        logger.log({
            'user': user,
            'target_item': t,
            'seed_item': i,
        })

        users_scores = {}
        for us in [users, random_users]:
            ranks = []
            for u in us:
                u_scores = self.model.score(u).detach().numpy().squeeze()
                u_scores.sort()
                u_scores = np.flip(u_scores)
                users_scores[u] = u_scores

                score = self.model.score_user_item(u, t).item()
                rank = self.get_rank(u_scores, score)

                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

        backup = self.model.F[t].unsqueeze(0).clone().detach()

        n_features = backup.shape[1]

        pca_fc = None
        if do_pca:
            n_components = args.n_components
            pca_fc = TorchPCA.get_pca(self.model.F.numpy(), n_components)
            n_features = n_components

        img = self.load_image(t)
        img_M = np.array(img)
        ori_img_M = img_M

        eps = 10

        for step in range(steps):
            print(f"Step: {step}")

            img_M = self.bb_attack(
                user, t, img, img_M, ori_img_M, n_examples, n_features, user_scores, 
                do_pca, pca_fc, by_rank, rank_distribution, epsilon, eps
            )

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0)).detach()

            for us in [users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = self.model.score_user_item(u, t, feat_t).item()
                    rank = self.get_rank(u_scores, score)

                    ranks.append(rank)

                logger.log({
                    'step': step+1,
                    'rank': ranks,
                })
                ranks = np.array(ranks)
                print((ranks <= 20).mean())

        return img


    def run_wb_segment_attack(self, args, logger):

        steps = args.steps
        epsilon = args.epsilon
        i = args.item

        while True:
            users = [u for u, items in enumerate(self.user_items) if i in items]
            if len(users) >= 20:
                break
            i += 1

        random_users = random.sample(range(self.model.n_users), k=100)

        args.user = user = self.model.add_fake_user_by_item(i)
        
        t = random.choice(range(self.model.n_items))

        print(f"Segment wb experiment for user={user}, item={t}")
        logger.log({
            'user': user,
            'target_item': t,
            'seed_item': i,
        })

        users_scores = {}
        for us in [users, random_users]:
            ranks = []
            for u in us:
                u_scores = self.model.score(u).detach().numpy().squeeze()
                u_scores.sort()
                u_scores = np.flip(u_scores)
                users_scores[u] = u_scores

                score = self.model.score_user_item(u, t).item()
                rank = self.get_rank(u_scores, score)

                ranks.append(rank)

            logger.log({
                'step': 0,
                'rank': ranks,
            })
            ranks = np.array(ranks)

        img = self.load_image(t)
        img_M = np.array(img)

        for step in range(steps):
            print(f"Step: {step}")

            t_img_tran = self.image_model.transform(img)
            t_img_tran = torch.unsqueeze(t_img_tran, 0)
            t_img_var = nn.Parameter(t_img_tran)
            t_feat = self.get_image_features(t_img_var)

            zero_gradients(t_img_var)
            loss = self.model.pointwise_forward([user], [t], t_feat)
            loss.backward(retain_graph=True)

            grad = t_img_var.grad.data
            grad = grad.sign().squeeze(0).numpy().transpose((1, 2, 0))
            img_M = np.clip(img_M.astype(np.int) - epsilon * grad, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_M)

            img_t = self.image_model.transform(img)
            feat_t = self.get_image_features(img_t.unsqueeze(0))

            for us in [users, random_users]:
                ranks = []
                for u in us:
                    u_scores = users_scores[u]
                    score = -self.model.pointwise_forward([u], [t], feat_t).item()
                    rank = self.get_rank(u_scores, score)

                    ranks.append(rank)

                logger.log({
                    'step': step+1,
                    'rank': ranks,
                })

                ranks = np.array(ranks)
                # print(ranks)
                # print(ranks.mean())
                print((ranks <= 20).mean())

        return img


    def run_baseline(self, args, logger):
        u = args.user

        u_scores = self.model.score(u).detach().numpy().squeeze()
        u_scores.sort()
        u_scores = np.flip(u_scores)

        if hasattr(args, 'item') and args.item is not None:
            t = args.item
            result = self.search(u, -1)
            from_rank = np.where(result==t)[0][0].item() + 1
        else:
            from_rank = args.from_rank
            result = self.search(u, from_rank)
            t = result[from_rank-1].item()

        top = self.search(u, 1)[0].item()

        t_img = self.load_image(t)
        top_img = self.load_image(top)

        print(f"Run single user baseline, for user {u} from rank {from_rank}")

        logger.log({
            'user': u,
            'item': t,
            'rank': from_rank,
        })

        img_M = np.array(t_img)
        top_M = np.array(top_img)

        img_M = np.clip(img_M.astype(np.float) - 0.07*top_M.astype(np.float), 0, 255).astype(np.uint8)

        img = Image.fromarray(img_M)

        img_t = self.image_model.transform(img)
        feat_t = self.get_image_features(img_t.unsqueeze(0))
        score = self.model.score_user_item(u, t, feat_t.detach()).item()
        rank = self.get_rank(u_scores, score)

        print(f"Rank: {rank}")

        logger.log({
            'step': 0,
            'rank': rank,
        })

        return img


    def run(self, name, args):

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # ts = time.strftime("%y%m%d-%H%M%S")
        logfile = f"{name}.log"

        with JsonLooger(logfile) as logger:
            logger.log({k:v for k,v in vars(args).items() if "__" not in k})

            if args.experiment == "single_user":
                if args.blackbox > 0:
                    return self.run_bb_single_user(args, logger)
                else:
                    return self.run_wb_single_user(args, logger)

            elif args.experiment == "segment":
                if args.blackbox > 0:
                    return self.run_bb_segment_attack(args, logger)
                else:
                    return self.run_wb_segment_attack(args, logger)

            elif args.experiment == "baseline":
                return self.run_baseline(args, logger)

if __name__ == '__main__':
    parser = ArgumentParser(description="Experiments")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--experiment', type=str, default="single_user")
    parser.add_argument('--blackbox', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--epsilon', type=int, default=1/255)
    parser.add_argument('--user', type=int, default=9)
    parser.add_argument('--item', type=int, default=328407)
    parser.add_argument('--from-rank', type=int, default=100000)
    parser.add_argument('--do-pca', type=int, default=0)
    parser.add_argument('--by-rank', type=int, default=1)
    parser.add_argument('--n-components', type=int, default=150)
    parser.add_argument('--rank-distribution', type=str, default='uniform')
    parser.add_argument('--examples', type=int, default=8)

    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k2', type=int, default=10)
    parser.add_argument('--algorithm', type=str, default='vbpr') # vbpr, deepstyle
    parser.add_argument('--experiment_name', type=str, default='exp_defualt')

    args = parser.parse_args()

    dataset_name = "Clothing_Shoes_and_Jewelry"

    dataset = RecSysDataset(dataset_name)

    if args.algorithm == "vbpr":
        model = VBPR(
            dataset.n_users, dataset.n_items, dataset.corpus.image_features, 
            args.k, args.k2)

    elif args.algorithm == "deepstyle":
        model = DeepStyle(
            dataset.n_users, dataset.n_items, dataset.n_categories, 
            dataset.corpus.image_features, dataset.corpus.item_category, args.k)

    model.load(f'../data/dataset/{dataset_name}/models/{args.algorithm}_resnet50.pth')

    print(args)

    exp = Experimentation(model, dataset_name)

    exp.run(args.experiment_name, args)

    
    

        