import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPAttention
from utils.maps import visualize_features


def encode_text_word_embedding(text_encoder: CLIPTextModel,
                               input_ids: torch.Tensor,
                               word_embeddings: torch.Tensor = None,
                               num_vstar: int=1,
                               n_retrieved: torch.Tensor = None,
                               return_pte: bool = False,
                               masking_strategy = None, # TODO add to args
                               plot_attn = False,
                               ): #-> BaseModelOutputWithPooling:
    """
    Encode text by replacing the '$' with the PTEs extracted with the
    inversion adapter.
    Heavily based on hugginface implementation of CLIP.
    """
    # 259 is the index of '$' in the vocabulary
    existing_indexes = (input_ids == 259).nonzero(as_tuple=True)[0]
    existing_indexes = existing_indexes.unique() # tells in which elements of the batch do the substitution

    if len(existing_indexes) > 0:  # if there are '$' in the text
        _, counts = torch.unique((input_ids == 259).nonzero(as_tuple=True)[0], return_counts=True)

        cum_sum = torch.cat((torch.zeros(1, device=input_ids.device).int(), torch.cumsum(counts, dim=0)[:-1]))

        # get the index of the first '$' in each sentence. Different only if different number of text tokens in batch items
        first_vstar_indexes = (input_ids == 259).nonzero()[cum_sum][:, 1]
        # find repetition index of $ symbol
        if type(word_embeddings) == list: # if n_retrieved is not None:
            rep_idx = []
            for i, pos_idx in enumerate(existing_indexes):
                nr = n_retrieved[pos_idx]
                rep_idx.append(first_vstar_indexes[i] + torch.arange(0, nr*num_vstar).to(first_vstar_indexes.device).to(first_vstar_indexes.dtype))
            # if type(word_embeddings) != list:
            #    rep_idx = torch.cat(rep_idx)

        elif n_retrieved is not None:
            assert word_embeddings.shape[0] == torch.sum(n_retrieved, dim=0)
            filtered_retrieved = n_retrieved[existing_indexes.cpu()]
            start_indexes_wemb = torch.cumsum(n_retrieved, dim=0)[existing_indexes.cpu()] - filtered_retrieved
            # start_indexes_wemb = start_indexes_wemb.repeat_interleave(filtered_retrieved)
            existing_indexes_wemb = torch.cat([start_indexes_wemb[i] + torch.arange(nr) for i, nr in enumerate(filtered_retrieved)]) # assumes cpu device for all
            word_embeddings = word_embeddings[existing_indexes_wemb]
                
        else:
            rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_vstar)])
        
        if type(word_embeddings) == list:
            word_embeddings = [image.to(input_ids.device) for image in word_embeddings]
        else:
            word_embeddings = word_embeddings.to(input_ids.device)

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    seq_length = input_ids.shape[-1]
    position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]
    input_embeds = text_encoder.text_model.embeddings.token_embedding(
        input_ids)

    if len(existing_indexes) > 0:

        if type(word_embeddings) == list:
            assert len(word_embeddings) == input_embeds.shape[0]
            # Example: rep_idx = [Tensor(11...42), ..., Tensor(11...26)] vector of PTEs positions
            for i, pos_idx in enumerate(existing_indexes):
                input_embeds[pos_idx, rep_idx[i]] = \
                    word_embeddings[pos_idx].to(input_embeds.dtype)  # replace the '$' with the PTEs
                
        elif n_retrieved is not None:
            assert existing_indexes_wemb.shape[0] == torch.sum(filtered_retrieved, dim=0)
            # n_retrieved = num_vstar * torch.Tensor(n_retrieved).to(int)
            # rep_idx determines the PTEs positions in the flattened 1-D version (bs*seq_len) of input_embeds
            rep_idx = torch.cat([seq_length*pos_idx + first_vstar_indexes[i] + \
                                  torch.arange(0, n_retrieved[pos_idx]*num_vstar).to(first_vstar_indexes.device).to(first_vstar_indexes.dtype) for i, pos_idx in enumerate(existing_indexes)])
            if len(word_embeddings.shape) == 2:
                word_embeddings = word_embeddings.unsqueeze(1)
            # numerical example: bs = 5, seq_length=77, num_vstar = 16, hidden_size = 1024, tot_retrieved = sum(1 3 2 1 3) = 10
            # flatten input_embeds into (5*77) x 1024, assign (10*16) x 1024 PTEs to the positions specified at rep_idx
            input_embeds.view(input_shape[0]*seq_length,-1)[rep_idx] = \
                word_embeddings.to(input_embeds.dtype).view(num_vstar*word_embeddings.shape[0], -1)
            
        else:
            assert word_embeddings.shape[0] == input_embeds.shape[0]
            if len(word_embeddings.shape) == 2:
                word_embeddings = word_embeddings.unsqueeze(1)
            input_embeds[torch.arange(input_embeds.shape[0]).repeat_interleave(
                num_vstar).reshape(input_embeds.shape[0], num_vstar)[existing_indexes.cpu()], rep_idx.T] = \
                word_embeddings.to(input_embeds.dtype)[existing_indexes]  # replace the '$' with the PTEs 
            # n_retrieved[existing_indexes], torch.Tensor(n_retrieved).to(int)

    position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
    hidden_states = input_embeds + position_embeddings

    if return_pte:
        return hidden_states
    
    # ENCODING AND VISUALIZATION PART

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # https://huggingface.co/transformers/v4.8.0/model_doc/clip.html
    attention_mask = None
    if masking_strategy == "txt2img":
        attention_mask = torch.zeros((bsz, seq_len)).to(rep_idx.device)
        attention_mask[:,rep_idx.transpose(0,1)] = 1 # is the correct choice for img-txt disentaglement?
    # breakpoint()
    
    # visualize self-attention. Goal: output both attention and output, no need 2 fw passes please
    class Hooker:
        def __init__(self):
            self.outputs = []
        
        def __call__(self, module, module_in, module_out): # case of multiple positional inputs? It's ok, is a list of items. check attn module inputs
            self.outputs.append(module_out)
        
        def clear(self):
            self.outputs = []

    hook_handles = []
    hooker = Hooker()
    if plot_attn:

        for layer in text_encoder.text_model.encoder.modules(): # check way of iterating used in clip text encoder
            if isinstance(layer, CLIPAttention):
                # breakpoint()
                handle = layer.register_forward_hook(hooker) # also register_module_...
                hook_handles.append(handle)

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    )
    for i, (hook, feats) in enumerate(zip(hook_handles, hooker.outputs)):
        visualize_features(feats[0], f"prova/feats_{i}.png")
        hook.remove()
        # print save_output.outputs, better if done inside class, not here

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(
        last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0],
                     device=last_hidden_state.device),
        input_ids.to(dtype=torch.int,
                     device=last_hidden_state.device).argmax(dim=-1),
    ]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    ).last_hidden_state

if __name__ == "__main__":
    we_type = "tensor"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nr_per_item = [3, 1, 2]
    hidden_size, num_vstar = 1024, 16
    tot_pte = num_vstar * sum(nr_per_item)
    category = ["dresses" for nr in nr_per_item]
    category_text = {
        'dresses': 'a dress',
        'upper_body': 'an upper body garment',
        'lower_body': 'a lower body garment',

    }
    text = [f'a photo of a model wearing {category_text[category[n]]} made of {" $ " * num_vstar * n_retrieved}'
            for n, n_retrieved in enumerate(nr_per_item)]
    bsz = len(nr_per_item)
    bsz_tot = sum(nr_per_item)
    uncond_mask_text = [False] * bsz
    uncond_mask_text[1] = True
    text = [t if not uncond_mask_text[i] else "" for i, t in enumerate(text)]
    print(text)
    add_vector = torch.arange(sum(nr_per_item))
    if we_type == "list":
        word_embeddings = [torch.zeros(nr*num_vstar, 1024) for nr in nr_per_item]
    else:
        word_embeddings = torch.arange(bsz_tot).view(bsz_tot, 1, 1).expand(bsz_tot, 16, 1024)

    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer", cache_dir="/work/CucchiaraYOOX2019/RAG/base_checkpoints")
    text_encoder=None
    tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                                truncation=True, return_tensors="pt").input_ids
    text_encoder = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder", cache_dir="/work/CucchiaraYOOX2019/RAG/base_checkpoints")
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)
    tokenized_text = tokenized_text.to(device)
    nr_tensor = torch.Tensor(nr_per_item).to(tokenized_text.dtype)
    encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                    word_embeddings,
                                                    num_vstar = num_vstar, n_retrieved=nr_tensor)

