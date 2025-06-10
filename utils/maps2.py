from matplotlib import pyplot
import math
import torch
from transformers.models.clip.modeling_clip import CLIPAttention
import torch.nn.functional as F

class Attn_Hooker:
    def __init__(self, return_probs=False):
        self.outputs = []
        self.return_probs = return_probs
    # how to get module_in for CLIP? And hidden states?
    def __call__(self, module, module_in, module_out): # case of multiple positional inputs? It's ok, is a list of items. check attn module inputs
        # self.-->module.
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = module.q_proj(hidden_states) * module.scale
        key_states = module._shape(module.k_proj(hidden_states), -1, bsz)
        value_states = module._shape(module.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * module.num_heads, -1, module.head_dim)
        query_states = module._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * module.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * module.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=module.dropout, training=module.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * module.num_heads, tgt_len, module.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, module.num_heads, tgt_len, module.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, module.num_heads, tgt_len, module.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = module.out_proj(attn_output)
        
        if self.return_probs:
            self.outputs.append(attn_probs)
        else:
            self.outputs.append(attn_output)
    
    def clear(self):
        self.outputs = []

class CLIPAttentionHooker:
    def __init__(self, seq_len, attn_mask=None, casual_attn_mask=None):
        self.all_heat_maps = []
        self.attn_mask = attn_mask
        self.causal_attention_mask = casual_attn_mask
        self.seq_len = seq_len
        self.latent_hw = 4096
    
    def __call__(
            self,
            attn: CLIPAttention,
            hidden_states,
            encoder_hidden_states=None
    ):
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = attn.q_proj(hidden_states) * self.scale
        key_states = attn._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = attn._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * attn.num_heads, -1,attn.head_dim)
        query_states = attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz, attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn._weights.size()}"
            )

        # apply the causal_attention_mask first
        if self.causal_attention_mask is not None:
            if self.causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {self.causal_attention_mask.size()}"
                )
            attn_weights =  self.n_weights.view(attn.num_heads, tgt_len, src_len) + self.causal_attention_mask
            attn_weights = self.n_weights.view(bsz, attn.num_heads, tgt_len, src_len)

        if self.attention_mask is not None:
            if self.attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {self.attention_mask.size()}"
                )
            attn_weights =attn.n_weights.view(bsz, attn.num_heads, tgt_len, src_len) + self.attention_mask
            attn_weights = attn_weights.view(bsz * attn.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = torch.nn.functional.dropout(attn_weights, p=attn.dropout, training=attn.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * attn.num_heads, tgt_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, tgt_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, attn.num_heads, tgt_len, attn.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = attn.out_proj(attn_output)
        heat_map = self._unravel_attn(attn_output)

        self.all_heat_maps.append(heat_map)
    
    def clear(self):
        self.all_heat_maps = []

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        #with auto_autocast(dtype=torch.float32):
        for map_ in x:
            map_ = map_.view(map_.size(0), h, w)
            map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
            maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        # with auto_autocast(dtype=torch.float32):
        for (factor, layer, head), heat_map in heat_maps:
            if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                heat_map = heat_map.unsqueeze(1)
                # The clamping fixes undershoot.
                all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

        try:
            maps = torch.stack(all_merges, dim=0)
        except RuntimeError:
            if head_idx is not None or layer_idx is not None:
                raise RuntimeError('No heat maps found for the given parameters.')
            else:
                raise RuntimeError('No heat maps found. Did you forget to call `with trace(..)` during generation?')

        maps = maps.mean(0)[:, 0]
        maps = maps[:self.sq_len]  # 1 for SOS and 1 for padding

        if normalize:
            maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return maps

def visualize_features(feature_maps, save_path):
    nc = feature_maps.shape[1] # nchannels in convlayers, nheads in attn layer
    h = math.sqrt(nc)
    exp = int(math.log2(h))
    if 2**exp != h: exp = exp+1
    h = 2**exp
    w = nc//h
    ix = 1
    for _ in range(h):
        for _ in range(w):
        # specify subplot and turn of axis
            ax = pyplot.subplot(h, w, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, ix-1, :, :], cmap='gray')
            ix += 1
    # show the figure
    pyplot.savefig(save_path)

# maybe need seaborn
def visualize_features_attn(feature_maps_list, nb, save_path):
    h = len(feature_maps_list) # nlayers*bs
    ix=1
    for nl, feature_maps in enumerate(feature_maps_list):
        feature_maps = feature_maps.detach().cpu().numpy()
        bs = feature_maps.shape[0] 
        w = feature_maps.shape[1] # nheads
        for na in range(w):
        # specify subplot and turn of axis
            ax = pyplot.subplot(h*bs, w, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[nb, na, :, :], cmap='gray')
            ix += 1
    # show the figure
    pyplot.savefig(save_path)

def plot_attention_maps(attentions, idx, save_path, mode="avg"):
    """_summary_

    Args:
        attentions (_type_): _description_
        idx (_type_): _description_
        save_path (_type_): _description_
    """
    attn_maps = attentions.detach().cpu().numpy() # use m[idx] in case of batch_size
    if mode == "avg":
        global_map = attn_maps.mean((0,1)) # careful
        fig, ax = plt.subplots(1,1,figsize=(3,3))
        ax.imshow(global_map, origin='lower', vmin=0)
        plt.savefig(save_path)
    
    else:
        num_heads = attn_maps[0].shape[0]
        num_layers = attn_maps.shape[0]
        seq_len = attn_maps[0].shape[-1]
        fig_size = 4 if num_heads == 1 else 3
        fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
        if num_layers == 1:
            ax = [ax]
        if num_heads == 1:
            ax = [[a] for a in ax]
        for row in range(num_layers):
            for column in range(num_heads):
                ax[row][column].imshow(attn_maps[row, column], origin='lower', vmin=0)
                # use this for PTEs explanability
                # ax[row][column].set_xticks(list(range(seq_len)))
                # ax[row][column].set_xticklabels(input_data.tolist())
                # ax[row][column].set_yticks(list(range(seq_len)))
                # ax[row][column].set_yticklabels(input_data.tolist())
                ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
        fig.subplots_adjust(hspace=0.5)
        plt.savefig(save_path)


import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Rectangle
import itertools


def plot_grid_query_pix(width, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.set_xticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.set_aspect(1)
    ax.set_yticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, alpha=0.5)

    # query pixel
    querry_pix = Rectangle(xy=(-0.5,-0.5),
                          width=1,
                          height=1,
                          edgecolor="black",
                          fc='None',
                          lw=2)

    ax.add_patch(querry_pix);

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)
    ax.set_aspect("equal")

def plot_attention_layer(attention_probs, width, ax=None):
    """Plot the 2D attention probabilities of all heads on an image
    of layer layer_idx
    """
    if ax is None:
        fig, ax = plt.subplots()

    contours = np.array([0.9, 0.5])
    linestyles = [":", "-"]
    flat_colors = ["#3498db", "#f1c40f", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#34495e", "#1abc9c", "#95a5a6"]

    if ax is None:
        fig, ax = plt.subplots()

    shape = attention_probs.shape
    # remove batch size if present
    if len(shape) == 6:
        shape = shape[1:]
    num_heads, height, width= shape

    attention_at_center = attention_probs[width // 2, height // 2]
    attention_at_center = attention_at_center.detach().cpu().numpy()

#     compute integral of distribution for thresholding
    n = 1000
    t = np.linspace(0, attention_at_center.max(), n)
    integral = ((attention_at_center >= t[:, None, None, None]) * attention_at_center).sum(
        axis=(-1, -2)
    )

    # plot_grid_query_pix(width - 2, ax)

    for h, color in zip(range(num_heads), itertools.cycle(flat_colors)):
        f = interpolate.interp1d(integral[:, h], t, fill_value=(1, 0), bounds_error=False)
        t_contours = f(contours)

        # remove duplicate contours if any
        keep_contour = np.concatenate([np.array([True]), np.diff(t_contours) > 0])
        t_contours = t_contours[keep_contour]

        for t_contour, linestyle in zip(t_contours, linestyles):
            ax.contour(
                np.arange(-width // 2, width // 2) + 1,
                np.arange(-height // 2, height // 2) + 1,
                attention_at_center[h],
                [t_contour],
                extent=[- width // 2, width // 2 + 1, - height // 2, height // 2 + 1],
                colors=color,
                linestyles=linestyle
            )

    return ax