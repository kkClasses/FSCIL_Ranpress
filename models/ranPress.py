import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,  SimpleVitNet, SimpleCosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# Tune the model at first session with ViT adapter, and then perform RanPAC.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.proto_list = []
        self.ridge = 0
        self.U, self.S, self.V = [], [], []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        print("start time", self.start_event)
        self.tim = [];

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        self.start_event.record()
        model = model.eval()
        embedding_list = []
        label_list = []
        cur_proto_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_list = np.unique(self.train_dataset.labels)

        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self.proto_list.append(proto)
            cur_proto_list.append(proto)

        if self.args['teen'] and self._cur_task > 0:
            base_protos = torch.stack(self.proto_list[:args['init_cls']])
            softmax_t = 16
            alpha = 0.9  # 0.9 for CUB200, Aircrafts, Cars and 0.75 for CIFAR100
            cur_proto_list = torch.stack(cur_proto_list).detach().cpu()
            weights = torch.mm(cur_proto_list, base_protos.T) * softmax_t
            norm_weights = torch.softmax(weights, dim=1)
            delta_protos = torch.matmul(norm_weights, base_protos)
            updated_protos = alpha * cur_proto_list + (1 - alpha) * delta_protos

            for idd, class_index in enumerate(class_list):
                # print(idd, class_index, updated_protos.shape, curInd)
                self.proto_list[class_index] = updated_protos[idd]
        # .............................................................
        if self.args["ranpac"] and (self.args["loRP"] == False):
            Y = target2onehot(label_list, self.args["nb_classes"])
            Features_h = F.relu(embedding_list @ self.W_rand.cpu())
            if self._cur_task == 0:
                if args['ranPress']:
                    self.W_reduced = self.reducedSVD(self.W_rand.cpu())
                    Features_h = F.relu(embedding_list @ self.W_reduced.cpu())
                self.ridge = self.optimise_ridge_parameter(Features_h, Y)
            # print(torch.mean(self.W_rand), embedding_list.shape, Features_h.shape)
            Q1 = Features_h.T @ Y;
            G1 = Features_h.T @ Features_h

            self.Q = self.Q + Q1
            self.G = self.G + G1

            Wo = torch.linalg.solve(self.G + self.ridge * torch.eye(self.G.size(dim=0)),
                                    self.Q).T  # better numerical stability than .invv
            print(self.Q.shape, self.G.shape, self.W_reduced.shape, Wo.shape)
            self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0], :].to(self._device)
            # if self._cur_task == 8:
            #   print((Wo[0:self._network.fc.weight.shape[0],:].cpu().numpy()).shape)
            #   np.savetxt("cub_proto_ranPress.csv",Wo[0:self._network.fc.weight.shape[0],:].cpu().numpy(), delimiter=",")
        elif self.args["loRP"]:
            # 1. Lift Features: (N, D) @ (D, M) -> (N, M)
            H_curr = F.relu(embedding_list.to(self._device) @ self.W_rand)
            # 2. Prepare Labels (N, C)
            Y_t = target2onehot(label_list, self.args["nb_classes"]).to(self._device)
            # 3. Update Label-Feature Product (Q = Y^T @ H)
            # Note: In your W = Y H^T notation, Y is (C, N) and H^T is (N, M)
            # Result Q: (C, M)
            current_Q = Y_t.T @ H_curr  # Q1 = Features_h.T @ Y ;
            self.Q = self.Q.to(self._device) + current_Q

            # 4. Update the Low-Rank Basis (U)
            r_target = self.args.get("rank", H_curr.shape[1] * 5 // 10)
            self.Sigma_1, self.U_1 = self.loranpac_svd_update(
                H_curr, r_target, self.Sigma_1, self.U_1)

            self.Sigma_1 = self.Sigma_1.to(self._device)  # (r, r)
            self.U_1 = self.U_1.to(self._device)  # (M, r)

            # 5. Solve for Classifier Weights (W)
            # Formula: W = Q @ (U @ Sigma^-2 @ U^T)
            ridge_lambda = self.ridge if self.ridge > 0 else 1e-4
            s_diag = torch.diag(self.Sigma_1)
            # Sigma^-2 with Ridge regularization
            inv_sigma_sq = torch.diag(1.0 / (s_diag ** 2 + ridge_lambda))
            # Efficient reconstruction: (C, M) @ (M, r) @ (r, r) @ (r, M) -> (C, M)
            QU = self.Q @ self.U_1
            Wo = (QU @ inv_sigma_sq) @ self.U_1.T
            # 6. Set Weights
            self._network.fc.weight.data = Wo[:self._total_classes, :].to(self._device)
        else:
            for idd, class_index in enumerate(class_list):
                self._network.fc.weight.data[class_index] = self.proto_list[class_index]
        # Wait for the GPU to catch up and finish
        # self.end_event.record()
        # torch.cuda.synchronize()
        # execution_time = self.start_event.elapsed_time(self.end_event) / 1000
        # self.tim.append(execution_time);
        # allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        # print("task end:", self._cur_task, self.tim, allocated_mem)

        return model

    def compute_svd(self, cur_protos):
        # Perform SVD (reduced form)
        U, S, V = torch.svd(cur_protos, some=True)  # U (n x r), S (r), V (d x r)
        print("Singular values=", U.shape, S.shape, V.shape)
        return U, S, V

    def reducedSVD(self, W):
        U, S, V = self.compute_svd(W)
        rank = 720;  # N = (self._network.fc.in_features -rank +r_)//10
        # if self._cur_task == 0:
        wb = U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T
        return wb

    def loranpac_svd_update1(self, H_current, target_rank, Sigma_past=None, U_past=None):
        """
        H_current: (N, M) from the dataloader
        U_past: (M, r) the feature basis from previous tasks
        Sigma_past: (r, r) the singular values from previous tasks
        """
        # H_t: (M, N) - We decompose the feature space
        H_t = H_current.T
        if U_past is None or Sigma_past is None:
            # Task 1: Standard Economy SVD
            U, S, _ = torch.linalg.svd(H_t, full_matrices=False)
        else:
            # --- Algorithm 2: Orthogonalization Step ---
            # 1. Project H_t onto the existing basis U_past
            # m: (r, N)
            m = U_past.T @ H_t
            # 2. Compute the component of H_t orthogonal to U_past
            # p: (M, N)
            p = H_t - U_past @ m
            # 3. QR Decomposition to get the new orthogonal directions (P)
            # P: (M, N), R_p: (N, N)
            P, R_p = torch.linalg.qr(p, mode='reduced')
            # 4. Construct the small "Core" Matrix B_t
            # B_t = [Sigma_past, m]  -> Shape: (r + N, r + N)
            #       [0,         R_p]
            top_row = torch.cat([Sigma_past, m], dim=1)
            bottom_row = torch.cat([torch.zeros(R_p.shape[0], Sigma_past.shape[1]).to(H_t.device), R_p], dim=1)
            B_t = torch.cat([top_row, bottom_row], dim=0)
            # 5. SVD on the small core matrix B_t (much faster than SVD on M)
            # U_tilde: (r+N, r+N), S: (r+N)
            U_tilde, S, _ = torch.linalg.svd(B_t, full_matrices=False)
            # 6. Rotate the combined basis into the new singular space
            # [U_past, P] is (M, r+N) @ U_tilde (r+N, r+N) -> U: (M, r+N)
            U = torch.cat([U_past, P], dim=1) @ U_tilde
        # --- Truncation ---
        r = min(target_rank, S.size(0))
        Sigma_new = torch.diag(S[:r])
        U_new = U[:, :r]
        return Sigma_new, U_new

    def loranpac_svd_update(self, H_current, target_rank, Sigma_past=None, U_past=None):
        """
        H_current: (N, M) from the dataloader
        U_past: (M, r) the feature basis from previous tasks
        """
        # Transpose input to match your convention: H is (M, N)
        H_t = H_current.T

        if U_past is None or Sigma_past is None:
            # Task 1: H_t (M, N) = U (M, r) @ Sigma (r, r) @ V^T (r, N)
            U, S, _ = torch.linalg.svd(H_t, full_matrices=False)
        else:
            # Task t: B = [U_past @ Sigma_past, H_t] -> Shape: M x (r + N)
            # This preserves the feature basis U
            past_summary = U_past @ Sigma_past
            B_t = torch.cat([past_summary, H_t], dim=1)
            U, S, _ = torch.linalg.svd(B_t, full_matrices=False)

        # Truncate to target rank r
        r = min(target_rank, S.size(0))
        Sigma_new = torch.diag(S[:r])
        U_new = U[:, :r]
        return Sigma_new, U_new

    def setup_RP(self):
        self.base_proto = self._network.fc.weight[:self.args['init_cls'], :].cpu().detach()
        M = self.args['M']
        self._network.fc.weight = nn.Parameter(
            torch.Tensor(self._network.fc.out_features, M).to(self._device)).requires_grad_(
            False)  # num classes in task x M
        self._network.RP_dim = M
        self.W_rand = torch.randn(self._network.fc.in_features, M).to(self._device)
        self.W_reduced = torch.zeros(self._network.fc.in_features, M).to(self._device)
        self._network.W_rand = self.W_rand
        self._network.W_reduced = self.W_reduced
        self.U_1 = None;
        self.Sigma_1 = None;
        self.V_1 = None;
        self.Q = torch.zeros(M, self.args["nb_classes"])
        if self.args["loRP"]:
            self.Q = torch.zeros(self.args["nb_classes"], M)  # for lowRP
            torch.nn.init.orthogonal_(self.W_rand)  # for lowRP

        self.G = torch.zeros(M, M)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(1, 9)
        # ridges = np.arange(1000, 10**5, 10000)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)),
                                    Q_val).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda = ', ridge)
        return ridge

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.to(self._device)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            self.shot = self.args["shot"]
        else:
            self.shot = None

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", shot=self.shot, )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", shot=self.shot, )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)

        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
            if not self.args['resume']:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
                self.save_checkpoint(
                    "weights/{}_{}_{}_{}".format(self.args["dataset"], self.args["model_name"], self.args["init_cls"],
                                                 self.args["increment"]))
                self._network.to(self._device)
            else:
                self._network.load_state_dict(torch.load(
                    "weights/{}_{}_{}_{}_{}.pkl".format(self.args["dataset"], self.args["model_name"],
                                                        self.args["init_cls"], self.args["increment"], self._cur_task))[
                                                  "model_state_dict"])
                self._network.to(self._device)
        else:
            pass
        if self._cur_task == 0 and self.args["ranpac"]:
            self.setup_RP()
        self.replace_fc(train_loader_for_protonet, self._network, self.args)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
