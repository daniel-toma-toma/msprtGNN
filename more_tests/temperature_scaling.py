import torch
from torch import nn, optim
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
import sequential_test

@torch.no_grad()
def singleton_dataset_from_list(model, dataset, device):
    out = dataset
    logits_list = []
    labels_list = []
    for data in tqdm(out):
        data=data.to(device)
        logits = model(data, return_logits=True)
        label = data.y
        logits_list.append(logits)
        labels_list.append(label)
    logits = torch.cat(logits_list).squeeze()
    labels = torch.cat(labels_list).squeeze()
    return labels, logits

@torch.no_grad()
def get_full_trace_data(model, valid_dataset, device):
    logits_list = []
    labels_list = []
    for data in valid_dataset:
        data = data.to(device)
        logits = model(data, return_logits=True)
        label = data.y
        logits_list.append(logits)
        labels_list.append(label)
        logits = torch.cat(logits_list).squeeze()
        labels = torch.cat(labels_list).squeeze()
    return labels, logits

@torch.no_grad()
def get_data(model, valid_dataset, device):
    logits_list = []
    labels_list = []
    for data in valid_dataset:
        data = data.to(device)
        depth = 0
        for subgraph in sequential_test.add_nodes_by_posting_time(data):
            logits = model(subgraph, return_logits=True)
            label = data.y
            logits_list.append(logits)
            labels_list.append(label)
            depth += 1
            if depth > 20:
                break
        logits = torch.cat(logits_list).squeeze()
        labels = torch.cat(labels_list).squeeze()
    return labels, logits

def singleton_graphs_from_data(data, node_indices=None):
    N = data.num_nodes
    node_indices = range(N) if node_indices is None else node_indices
    singletons = []
    for i in node_indices:
        d = Data()
        if hasattr(data, 'x') and data.x is not None:
            d.x = data.x[i].unsqueeze(0)  # [1, F]
        if hasattr(data, 'y') and data.y is not None:
            yi = data.y[i] if data.y.size(0) == N else data.y  # keep graph-level y as-is
            d.y = yi if yi.dim() else yi.unsqueeze(0)
        if hasattr(data, 'pos') and data.pos is not None:
            d.pos = data.pos[i].unsqueeze(0)
        # any other per-node attributes you'd like to carry over:
        for key, val in data.items():
            if key in {'x', 'y', 'pos', 'edge_index', 'edge_attr', 'adj_t', 'batch'}:
                continue
            # If this attribute is node-level (first dim == N), slice it:
            if isinstance(val, torch.Tensor) and val.size(0) == N:
                d[key] = val[i].unsqueeze(0)
        d.edge_index = torch.empty((2, 0), dtype=torch.long)
        d.num_nodes = 1
        singletons.append(d)
    return singletons


def temperature_scaling_test(orig_model, valid_loader, device):
    scaled_model = ModelWithTemperature(orig_model)
    scaled_model.set_temperature(valid_loader, device)
    return scaled_model

def get_ece_from_logits(logits, labels, num_classes, rel_diag_name):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss(num_classes=num_classes, n_bins=15).cuda()
    reliabilty_diag_criterion = _ECELoss(num_classes=num_classes, n_bins=10).cuda()
    nll = nll_criterion(logits, labels).item()
    ece = ece_criterion(logits, labels).item()
    reliabilty_diag_criterion.draw_reliability_graph(logits, labels, rel_diag_name)
    return nll, ece

def get_ece(model, dataset, device, rel_diag_name, labels=None, logits=None):
    model.eval()
    labels, logits = get_full_trace_data(model, dataset, device)
    nll, ece = get_ece_from_logits(logits, labels, model.num_classes, rel_diag_name)
    return ece, nll

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.num_classes = model.num_classes
        self.T = nn.Parameter(torch.tensor(float(1.0)))

    def forward(self, input, return_logits=False):
        logits = self.model(input, return_logits=return_logits)
        return self.temperature_scale(logits)

    def reset(self):
        pass

    def temperature_scale(self, logits):
        return logits / self.T

    def set_temperature(self, valid_dataset, device):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss(num_classes=self.model.num_classes, n_bins=15).cuda()
        labels, logits = get_full_trace_data(self, valid_dataset, device)

        training_mode = self.training        # Prevent batchnorm statistics from changing
        self.eval()
        before_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        before_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f (validation dataset)' % (before_temperature_nll, before_temperature_ece))
        optimizer = optim.LBFGS([self.T], lr=0.01, max_iter=1000, line_search_fn="strong_wolfe")
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('After temperature - NLL: %.3f, ECE: %.3f, optimal temperature: %.3f (validation dataset)' % (after_temperature_nll, after_temperature_ece, self.T.detach().item()))
        self.train(training_mode)
        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15, num_classes=4):
        super(_ECELoss, self).__init__()
        self.n_bins = n_bins
        self.num_classes = num_classes
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]

    def draw_reliability_graph(self, logits, labels, name):
        """Prints the full LaTeX document (generated inside _make_reliability_latex)."""
        latex_code = self._make_reliability_latex(logits, labels, name)
        print(latex_code)

    def _make_reliability_latex(self, ece, accs, cfs, bin_centers, width, name):
        """Returns a *complete* LaTeX document including preamble + reliability diagram."""
        safe_name = str(name).replace('_', r'\_')

        lines = []

        # -----------------------------------------------------------
        # LaTeX PREAMBLE **included directly in this function**
        # -----------------------------------------------------------
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage[margin=1in]{geometry}")
        lines.append(r"\usepackage{float} % for H placement")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage{tikz}")
        lines.append(r"\usepackage{pgfplots}")
        lines.append(r"\pgfplotsset{compat=newest}")
        lines.append("")
        lines.append(r"\begin{document}")
        lines.append("")

        # -----------------------------------------------------------
        # FIGURE ENVIRONMENT
        # -----------------------------------------------------------
        lines.append(r"\begin{figure}[H]")
        lines.append(r"  \centering")
        lines.append(r"  \begin{tikzpicture}")
        lines.append(r"    \begin{axis}[")
        lines.append(r"      width=8cm,")
        lines.append(r"      height=8cm,")
        lines.append(r"      xmin=0, xmax=1.05,")
        lines.append(r"      ymin=0, ymax=1,")
        lines.append(r"      grid=both,")
        lines.append(r"      xlabel={Confidence},")
        lines.append(r"      ylabel={Accuracy},")
        lines.append(r"      axis equal image,")
        lines.append(r"      legend style={at={(0.02,0.98)},anchor=north west,legend columns=1,/tikz/every even column/.append style={column sep=0.5cm}},")
        lines.append(r"     legend image code/.code={\draw [#1] (0cm,-0.1cm) rectangle (0.2cm,0.25cm); },    ]")

        # Confidence bars
        lines.append(r"    % Confidence histogram")
        lines.append(r"    \addplot[ybar,fill=blue,draw=black] coordinates {")
        for x, y in zip(bin_centers, cfs):
            lines.append(f"      ({x:.4f},{y:.4f})")
        lines.append(r"    };")

        # Accuracy bars
        lines.append(r"    % Accuracy histogram")
        lines.append(r"    \addplot[ybar,fill=red,draw=black,fill opacity=0.3,pattern=north east lines] coordinates {")
        for x, y in zip(bin_centers, accs):
            lines.append(f"      ({x:.4f},{y:.4f})")
        lines.append(r"    };")

        # Perfect calibration line
        lines.append(r"    % Perfect calibration line")
        lines.append(r"    \addplot[dashed] coordinates {(0,0) (1,1)};")

        lines.append(rf" \node[fill=white, draw=black, rounded corners=2pt, above] at (axis cs:0.2,0.7) {{ECE={ece.item():.4f}}}; \legend{{Outputs, Accuracy}}")
        lines.append(r"    \legend{Confidence,Accuracy}")
        lines.append(r"    \end{axis}")
        lines.append(r"  \end{tikzpicture}")
        lines.append(rf"  \caption{{Reliability diagram for {safe_name}}}")
        lines.append(r"  \label{fig:reliability}")
        lines.append(r"\end{figure}")

        # -----------------------------------------------------------
        # END DOCUMENT
        # -----------------------------------------------------------
        lines.append("")
        lines.append(r"\end{document}")

        return "\n".join(lines)

    def draw_reliability_graph(self, logits, labels, name):
        ece, accs, cfs = self.ece_calc(logits, labels)
        bin_centers = self.bin_lowers + (self.bin_uppers - self.bin_lowers) / 2
        width = self.bin_uppers - self.bin_lowers
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed')
        plt.bar(bin_centers, cfs, width=width, alpha=1, edgecolor='black', color='b')
        plt.bar(bin_centers, accs, width=width, alpha=0.3, edgecolor='black', color='r', hatch='\\')
        plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('./tmp/calibrated_network_' + name + '.png', bbox_inches='tight')
        latex_code = self._make_reliability_latex(ece, accs, cfs, bin_centers, width, name)
        with open('./tmp/calibrated_network_' + name + '.tex', "w") as f:
            f.write(latex_code)

    def ece_calc(self, logits, labels):
        softmaxes = torch.softmax(logits, dim=1)
        one_hot =  F.one_hot(labels, num_classes=self.num_classes).float()
        confidences = softmaxes.view(-1)
        accuracies = one_hot.view(-1)
        #confidences, predictions = torch.max(softmaxes, 1)
        #accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        accs = []
        cfs = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                accs += [accuracy_in_bin]
                cfs += [avg_confidence_in_bin]
            else:
                accs += [torch.tensor(0).float()]
                cfs += [torch.tensor(0).float()]
        accs = torch.stack(accs).detach().numpy()
        cfs = torch.stack(cfs).detach().numpy()
        return ece, accs, cfs

    def forward(self, logits, labels):
        ece, accs, cfs = self.ece_calc(logits, labels)
        return ece
