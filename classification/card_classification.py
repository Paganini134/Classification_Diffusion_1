import logging
import time
import gc
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats import ttest_rel
from tqdm import tqdm
from data_loader import *
from ema import EMA
from model import *
from pretraining.encoder import Model as AuxCls
from pretraining.resnet import ResNet18
from utils import *
from diffusion_utils import *
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
plt.style.use('ggplot')
import os
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import pandas as pd
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.archi=config.model.arch
        self.folder=config.training.new_folder
        print("what device is being used")
        print(device)

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.embed=config.model.embeddings
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        #error aux_cls gives the cond_pred_model or the guidance classifier
        if config.diffusion.apply_aux_cls:
            if config.data.dataset == "gaussian_mixture":
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(1, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1)
                ).to(self.device)
            # elif config.data.dataset == "MNIST" and config.model.arch == "simple":
            elif config.data.dataset =="MED_DATA":
                efficientnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                efficientnet_model.classifier[1] = nn.Linear(in_features=1280, out_features=(config.data.num_classes))
                efficientnet_model.features[0][0] = nn.Conv2d(
                in_channels=3,  # Keep input as RGB (3 channels)
                out_channels=32,  # Keep the same number of output channels
                kernel_size=7,  # Increase kernel size from 3x3 to 7x7
                stride=1,  # Reduce stride to prevent excessive downsampling
                padding=3,  # Adjust padding to maintain spatial dimensions
                bias=False  #error try a true bias
            )
            elif config.data.dataset =="Vehicle":
                loaded_model = models.resnet18(pretrained=False)
                num_ftrs = loaded_model.fc.in_features
                loaded_model.fc = nn.Linear(num_ftrs, 1)
                loaded_model = loaded_model.to(device)
                self.cond_pred_model=loaded_model

                # Load state dict with strict=False to avoid key mismatches (if any)
        else:
            pass

        # scaling temperature for NLL and ECE computation
        self.tuned_scale_T = None

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if self.config.model.arch == "simple" or \
                (self.config.model.arch == "linear" and self.config.data.dataset == "MNIST"):
            x = torch.flatten(x, 1)
        y_pred = self.cond_pred_model(x)
        return y_pred

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            # logging.info("\nEvaluating test Minibatch {}...\n".format(step))
            # minibatch_start = time.time()
            x_batch, y_labels_batch = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            y_pred_prob = self.compute_guiding_prediction(
                x_batch.to(self.device)).softmax(dim=1)  # (batch_size, n_classes)
            y_pred_label = torch.argmax(y_pred_prob, 1, keepdim=True).cpu().detach().numpy()  # (batch_size, 1)
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label == y_labels_batch  # (batch_size, 1)
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred = self.compute_guiding_prediction(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)

        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
        logging.info("\nBefore training, the guidance classifier accuracy on the test set is {:.8f}.\n\n".format(
            y_acc_aux_model))

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.CrossEntropyLoss()
        brier_score = nn.MSELoss()

        # apply an auxiliary optimizer for the guidance classifier
        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            efficient_net_weights_path="/home/iotlab/Desktop/CARD-MAIN/classification/resnet18_finetuned_weights.pth"
            aux_states = torch.load(efficient_net_weights_path, map_location=self.device)
            self.cond_pred_model.load_state_dict(aux_states, strict=True)  # No ['state_dict'] needed

            self.cond_pred_model.eval()
            # save auxiliary model after pre-training
            aux_states = [
                self.cond_pred_model.state_dict(),
                aux_optimizer.state_dict(),
            ]

            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            # report accuracy on both training and test set for the pre-trained auxiliary classifier

            y_acc_aux_model = self.evaluate_guidance_model(train_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                y_acc_aux_model))

            y_acc_aux_model = self.evaluate_guidance_model(test_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the test set is {:.8f}.\n".format(
                y_acc_aux_model))

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = 0.0
            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")
            #x

            # ... (other necessary imports and function definitions)

            # Assuming these lists are defined at the beginning of training:
            epoch_list = []
            accuracy_list = []
            rmse_list = []
            cross_entropy_list = []
            nll_list = []
            noise_loss_list = []  # New: Store noise reconstruction loss

            max_accuracy = 0.0  # Track best accuracy
            step = 0  # Make sure step is defined

            # Training loop
            for epoch in range(0, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                vari = 0

                for i, feature_label_set in enumerate(train_loader):
                    if config.data.dataset == "gaussian_mixture":
                        x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                    else:
                        x_batch, y_labels_batch = feature_label_set
                        y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)

                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)

                    n = x_batch.size(0)
                    x_unflat_batch = x_batch.to(self.device)

                    if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                        x_batch = torch.flatten(x_batch, 1)

                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    x_batch = x_batch.to(self.device)
                    y_0_hat_batch = self.compute_guiding_prediction(x_unflat_batch).softmax(dim=1)
                    y_T_mean = y_0_hat_batch if not config.diffusion.noise_prior else torch.zeros_like(y_0_hat_batch)
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)

                    y_t_batch = q_sample(y_0_batch, y_T_mean, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    output = model(x_batch, y_t_batch, t, y_0_hat_batch)

                    # Compute Noise Loss (Reconstruction Loss of Diffusion Model)
                    noise_loss = (e - output).square().mean()

                    loss = noise_loss  # Start with noise loss as the base

                    loss0 = torch.tensor([0.0], device=self.device)
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t, self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                        loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0  # Add Cross Entropy loss to the total loss

                    if tb_logger is not None:
                        tb_logger.add_scalar("loss", loss.item(), global_step=step)
                        tb_logger.add_scalar("noise_loss", noise_loss.item(), global_step=step)  # Log noise loss

                    if epoch % self.config.training.logging_freq == 0 and vari == 0:
                        logging.info(f"Epoch: {epoch}, Step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, Data time: {data_time / (i + 1)}")
                        vari = 1

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                    optimizer.step()

                    if self.config.model.ema:
                        ema_helper.update(model)

                    data_start = time.time()

                # ======= VALIDATION AFTER EVERY FEW EPOCHS =======
                if epoch % self.config.training.validation_freq == 0 or epoch + 1 == self.config.training.n_epochs:
                    model.eval()
                    self.cond_pred_model.eval()
                    acc_avg = 0.0
                    loss_mse_total = 0.0
                    loss_ce_total = 0.0
                    loss_nll_total = 0.0
                    noise_loss_total = 0.0  

                    for test_batch_idx, (images, target) in enumerate(train_loader):
                        images_unflat = images.to(self.device)
                        if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                            images = torch.flatten(images, 1)

                        images = images.to(self.device)
                        target = target.to(self.device)

                        with torch.no_grad():
                            target_pred = self.compute_guiding_prediction(images_unflat).softmax(dim=1)
                            y_T_mean = target_pred if not config.diffusion.noise_prior else torch.zeros_like(target_pred)

                            label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                                    self.num_timesteps, self.alphas,
                                                    self.one_minus_alphas_bar_sqrt,
                                                    only_last_sample=True)

                            # Compute accuracy
                            acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()

                            # Compute loss functions
                            num_classes = label_t_0.shape[1]
                            target_one_hot = F.one_hot(target, num_classes=num_classes).float()

                            loss_mse = (F.mse_loss(label_t_0, target_one_hot))
                            loss_ce = F.cross_entropy(label_t_0, target)
                            loss_nll = F.nll_loss(F.log_softmax(label_t_0, dim=1), target)
                            noise_loss_validation = (label_t_0 - target_one_hot).square().mean()  # Compute noise loss for validation

                            loss_mse_total += loss_mse.item()
                            loss_ce_total += loss_ce.item()
                            loss_nll_total += loss_nll.item()
                            noise_loss_total += noise_loss_validation.item()

                    acc_avg /= (test_batch_idx + 1)
                    loss_mse_avg = loss_mse_total / (test_batch_idx + 1)
                    loss_ce_avg = loss_ce_total / (test_batch_idx + 1)
                    loss_nll_avg = loss_nll_total / (test_batch_idx + 1)
                    noise_loss_avg = noise_loss_total / (test_batch_idx + 1)

                    # Store results
                    accuracy_list.append(acc_avg)
                    epoch_list.append(epoch)
                    rmse_list.append(loss_mse_avg)
                    cross_entropy_list.append(loss_ce_avg)
                    nll_list.append(loss_nll_avg)
                    noise_loss_list.append(noise_loss_avg)  # Store noise loss

                    # Log results
                    if acc_avg > max_accuracy:
                        logging.info(f"Update best accuracy at Epoch {epoch}.")
                    max_accuracy = max(max_accuracy, acc_avg)

                    if tb_logger is not None:
                        tb_logger.add_scalar('accuracy', acc_avg, global_step=step)

                    logging.info(f"Epoch: {epoch}, Step: {step}, Average Accuracy: {acc_avg:.4f}, Max Accuracy: {max_accuracy:.2f}%, RMSE Loss: {loss_mse_avg:.4f}, CE Loss: {loss_ce_avg:.4f}, NLL Loss: {loss_nll_avg:.4f}, Noise Loss: {noise_loss_avg:.4f}")

            # ======= AFTER TRAINING IS COMPLETE: SAVE THE METRICS TO CSV =======
            # Define file paths


            # Define paths
            save_path = f"/home/iotlab/Desktop/CARD-MAIN/diffusion_model_plots/{self.folder}"
            os.makedirs(save_path, exist_ok=True)

            # Paths to stored results for other models
            other_models_paths = {
                "resnet18": os.path.join(save_path, "resnet18_epoch_metrics.csv"),
                "resnet50": os.path.join(save_path, "resnet50_epoch_metrics.csv"),
                "googlenet": os.path.join(save_path, "googlenet_epoch_metrics.csv"),
            }

            # Path for diffusion model CSV
            diffusion_model_path = os.path.join(save_path, "diffusion_model_epoch_metrics.csv")

            # ✅ Step 1: Save Diffusion Model Metrics as a CSV
            df_diffusion = pd.DataFrame({
                "Epoch": epoch_list,
                "Accuracy": accuracy_list,
                "MSE": rmse_list,
                "CrossEntropy": cross_entropy_list,
                "NLL": nll_list,
                "NoiseLoss": noise_loss_list
            })

            df_diffusion.to_csv(diffusion_model_path, index=False)
            print(f"\n==== Diffusion Model CSV Saved at {diffusion_model_path} ====")

            # ✅ Step 2: Load Other Models' CSVs & Align Epochs
            def load_other_models():
                other_models_data = {}
                for model_name, path in other_models_paths.items():
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        # Trim other models' data to match the diffusion model epochs
                        max_epoch = df_diffusion["Epoch"].max()
                        df = df[df["Epoch"] <= max_epoch]
                        other_models_data[model_name] = df
                    else:
                        print(f"Warning: File not found - {path}")
                return other_models_data

            other_models_data = load_other_models()

            def plot_metric(metric_name, y_label, zoomed=False, zoom_ylim=None):
                # Define marker mapping for models:
                # Diffusion Model: circle ('o')
                # ResNet18: triangle up ('^')
                # All other models: default to square ('s')
                marker_mapping = {
                    "Diffusion Model": "o",
                    "ResNet18": "^"
                }
                default_marker = "s"
                
                # Use a clean style and set figure size
                plt.style.use('seaborn-darkgrid')
                plt.figure(figsize=(10, 5))
                
                # For the diffusion model, get epochs and values and compute marker indices
                epochs = df_diffusion["Epoch"]
                values = df_diffusion[metric_name]
                markevery = df_diffusion.index[df_diffusion["Epoch"] % 5 == 0].tolist()
                plt.plot(epochs, values, label="Diffusion Model",
                        linewidth=2, marker=marker_mapping["Diffusion Model"], markevery=markevery)
                
                # Annotate diffusion model markers with an offset and background box
                for idx in markevery:
                    epoch_val = epochs.iloc[idx]
                    value_val = values.iloc[idx]
                    plt.annotate(f"{value_val:.2f}", (epoch_val, value_val),
                                textcoords="offset points", xytext=(0, 5),
                                ha='center', fontsize=8, color='black',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                # Ensure the last epoch is annotated if not already
                last_idx = df_diffusion.index[-1]
                if last_idx not in markevery:
                    epoch_val = epochs.iloc[last_idx]
                    value_val = values.iloc[last_idx]
                    plt.annotate(f"{value_val:.2f}", (epoch_val, value_val),
                                textcoords="offset points", xytext=(0, 5),
                                ha='center', fontsize=8, color='black',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                
                # Plot other models with the same approach
                for model_name, df in other_models_data.items():
                    epochs_other = df["Epoch"]
                    if metric_name == "Accuracy" and model_name != "Diffusion Model":
                        # Multiply by 100 if needed
                        values_other = 100 * df[metric_name]
                    else:
                        values_other = df[metric_name]
                    markevery_other = df.index[df["Epoch"] % 5 == 0].tolist()
                    marker = marker_mapping.get(model_name, default_marker)
                    plt.plot(epochs_other, values_other, label=model_name,
                            linewidth=2, marker=marker, markevery=markevery_other)
                    # Annotate each marker with its value using an offset
                    for idx in markevery_other:
                        epoch_val = epochs_other.iloc[idx]
                        value_val = values_other.iloc[idx]
                        plt.annotate(f"{value_val:.2f}", (epoch_val, value_val),
                                    textcoords="offset points", xytext=(0, 5),
                                    ha='center', fontsize=8, color='black',
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                    # Ensure the last epoch is annotated if not already
                    last_idx_other = df.index[-1]
                    if last_idx_other not in markevery_other:
                        epoch_val = epochs_other.iloc[last_idx_other]
                        value_val = values_other.iloc[last_idx_other]
                        plt.annotate(f"{value_val:.2f}", (epoch_val, value_val),
                                    textcoords="offset points", xytext=(0, 5),
                                    ha='center', fontsize=8, color='black',
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                
                # Labeling: For Accuracy, change y-label to "Training Accuracy"
                plt.xlabel("Epochs", fontsize=12)
                if metric_name == "Accuracy":
                    plt.ylabel("Training Accuracy", fontsize=12)
                else:
                    plt.ylabel(y_label, fontsize=12)
                
                # Do not set a title as per requirements
                # plt.title(f"{y_label} over Epochs", fontsize=14)
                
                # Set x-axis ticks every 5 epochs
                plt.xticks(np.arange(0, epochs.max() + 1, 5))
                plt.legend(fontsize=10)
                plt.grid(True)
                
                # Adjust y-axis limits for zoomed plots and set file extension to PDF
                if zoomed:
                    if zoom_ylim is not None:
                        plt.ylim(*zoom_ylim)
                        plot_filename = f"{metric_name}_zoomed_{zoom_ylim[0]}_{zoom_ylim[1]}_plot.pdf"
                    else:
                        plt.ylim(0, 3)
                        plot_filename = f"{metric_name}_zoomed_plot.pdf"
                else:
                    plot_filename = f"{metric_name}_comparison_plot.pdf"
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, plot_filename))
                plt.close()

            # Plot standard graphs
            plot_metric("Accuracy", "Accuracy")
            plot_metric("MSE", "MSE")
            plot_metric("CrossEntropy", "Cross Entropy Loss")
            plot_metric("NLL", "Negative Log Likelihood (NLL)")

            # Plot additional zoomed-in graphs (for MSE, CrossEntropy, and NLL)
            plot_metric("MSE", "MSE", zoomed=True)
            plot_metric("CrossEntropy", "Cross Entropy Loss", zoomed=True)
            plot_metric("NLL", "Negative Log Likelihood (NLL)", zoomed=True)

            # Plot an additional zoomed-in MSE graph with y-axis between 0 and 1
            plot_metric("MSE", "MSE", zoomed=True, zoom_ylim=(0, 1))

            print("\n==== All Plots Saved Successfully ====")

            # ---------------------------
            # Evaluation over test_loader
            # ---------------------------
            acc_avg = 0.0
            loss_mse_total, loss_ce_total, loss_nll_total, noise_loss_total = 0.0, 0.0, 0.0, 0.0
            precision_total, recall_total, f1_total = 0.0, 0.0, 0.0
            num_batches = 0

            # Assuming test_loader is defined, as well as model, self.device, config, etc.
            for test_batch_idx, (images, target) in enumerate(test_loader):
                images_unflat = images.to(self.device)
                if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                    images = torch.flatten(images, 1)
                
                images = images.to(self.device)
                target = target.to(self.device)

                with torch.no_grad():
                    # Compute model prediction (using your guiding function and p_sample_loop)
                    target_pred = self.compute_guiding_prediction(images_unflat).softmax(dim=1)
                    y_T_mean = target_pred if not config.diffusion.noise_prior else torch.zeros_like(target_pred)
                    
                    label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                            self.num_timesteps, self.alphas,
                                            self.one_minus_alphas_bar_sqrt,
                                            only_last_sample=True)

                    # Compute accuracy (assuming accuracy() returns a tuple and the first element is accuracy)
                    acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                    
                    # Convert predictions to class indices for other metrics
                    pred_labels = label_t_0.argmax(dim=1).cpu().numpy()
                    true_labels = target.cpu().numpy()

                    # Investigate the predicted and true labels for this batch:
                    print(f"Batch {test_batch_idx}: Unique predicted classes: {np.unique(pred_labels)}")
                    print(f"Batch {test_batch_idx}: Unique true classes: {np.unique(true_labels)}")

                    # Compute additional metrics using sklearn
                    precision_total += precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                    recall_total    += recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                    f1_total        += f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
                    
                    # Compute loss functions
                    num_classes = label_t_0.shape[1]
                    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
                    
                    loss_mse = F.mse_loss(label_t_0, target_one_hot)  # using MSE loss (optionally take sqrt if desired)
                    loss_ce   = F.cross_entropy(label_t_0, target)
                    loss_nll  = F.nll_loss(F.log_softmax(label_t_0, dim=1), target)
                    noise_loss_validation = (label_t_0 - target_one_hot).square().mean()
                    
                    loss_mse_total  += loss_mse.item()
                    loss_ce_total    += loss_ce.item()
                    loss_nll_total   += loss_nll.item()
                    noise_loss_total += noise_loss_validation.item()
                    
                    num_batches += 1


            # Compute averages over all batches
            acc_avg       /= num_batches
            precision_avg = precision_total / num_batches
            recall_avg    = recall_total    / num_batches
            f1_avg        = f1_total        / num_batches
            loss_mse_avg  = loss_mse_total / num_batches
            loss_ce_avg    = loss_ce_total   / num_batches
            loss_nll_avg   = loss_nll_total  / num_batches
            noise_loss_avg = noise_loss_total/ num_batches


            new_results = {
                "Model": "Diffusion", 
                "Accuracy": acc_avg,
                "Precision": precision_avg,
                "Recall": recall_avg,
                "F1 Score": f1_avg,
                "MSE": loss_mse_avg,
                "CrossEntropy Loss": loss_ce_avg,
                "NLL": loss_nll_avg,
                "Noise Loss":noise_loss_avg
            }

            new_df = pd.DataFrame([new_results])

            # Define the path to the CSV file with previous model results
            csv_path = f"/home/iotlab/Desktop/CARD-MAIN/diffusion_model_plots/{self.folder}/model_comparison_results.csv"

            # Append the new results to the CSV file
            if os.path.exists(csv_path):
                # Load the existing CSV into a DataFrame
                df_existing = pd.read_csv(csv_path)
                # Concatenate the new results
                df_updated = pd.concat([df_existing, new_df], ignore_index=True)
                # Save the updated DataFrame back to CSV
                df_updated.to_csv(csv_path, index=False)
                print(f"Appended Diffusion results to {csv_path}")
            else:
                # If the CSV does not exist, create it with the new results
                new_df.to_csv(csv_path, index=False)
                print(f"Created new CSV file with Diffusion results at {csv_path}")


            # ---------------------------
            # Combine with Existing Comparison Results Table
            # ---------------------------
            # Path to the existing CSV file
            existing_csv_path = f"/home/iotlab/Desktop/CARD-MAIN/diffusion_model_plots/{save_path}/model_comparison_results.csv"

            if os.path.exists(existing_csv_path):
                existing_df = pd.read_csv(existing_csv_path)
                # Option 1: If the model is already present and you wish to update it,
                # you can remove any previous row for the same model. Otherwise, simply append.
                existing_df = existing_df[existing_df["Model"] != new_results["Model"]]
                
                # Append the new row to the existing DataFrame
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            # Save the new combined table to a CSV file.
            # You can overwrite the existing file or create a new file (here we create a new file):
            combined_csv_path = "/home/iotlab/Desktop/CARD-MAIN/all_models_comparison_results.csv"
            combined_df.to_csv(combined_csv_path, index=False)

            print(f"Combined metrics table saved at {combined_csv_path}")
            print("file_1 done")

'''
if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
    aux_states = [
        self.cond_pred_model.state_dict(),
        aux_optimizer.state_dict(),
    ]
    torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
    # report training set accuracy if applied joint training
    y_acc_aux_model = self.evaluate_guidance_model(train_loader)
    logging.info("After joint-training, guidance classifier accuracy on the training set is {:.8f}.".format(
        y_acc_aux_model))
    # report test set accuracy if applied joint training
    y_acc_aux_model = self.evaluate_guidance_model(test_loader)
    logging.info("After joint-training, guidance classifier accuracy on the test set is {:.8f}.".format(
        y_acc_aux_model))
'''