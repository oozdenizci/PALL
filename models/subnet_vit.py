import math
from models.base import *
from .vit import ScaledDotProductAttention
from .subnet_layers import SubnetConv2d, SubnetLinear, SubnetClassifier


__all__ = ['SubnetVisionTransformer', 'subnet_vit_t_8', 'subnet_vit_t_16']


class maskedSequentialViT(nn.Sequential):
    def forward(self, *inputs):
        x = inputs[0]
        mask = inputs[1]
        mode = inputs[2]
        for module in self._modules.values():
            if isinstance(module, SubnetEncoderBlock):
                x = module(x, mask, mode)
            else:
                x = module(x)
        return x


class SubnetMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k=None, d_v=None, p_dropout=0.0, sparsity=0.0, name=""):
        super(SubnetMultiHeadAttention, self).__init__()
        self.name = name
        self.n_head = n_head
        self.d_k = d_k if d_k is not None else d_model // n_head
        self.d_v = d_v if d_v is not None else d_model // n_head
        self.W_Q = SubnetLinear(d_model, self.d_k * n_head, bias=False, sparsity=sparsity)
        self.W_K = SubnetLinear(d_model, self.d_k * n_head, bias=False, sparsity=sparsity)
        self.W_V = SubnetLinear(d_model, self.d_v * n_head, bias=False, sparsity=sparsity)
        self.sdpa = ScaledDotProductAttention(p_dropout)
        self.fc = SubnetLinear(self.d_v * n_head, d_model, bias=False, sparsity=sparsity)

    def forward(self, Q, K, V, mask, mode='train'):
        """
            inputs:
                Q: (batch_size, seq_len1, d_model)
                K: (batch_size, seq_len2, d_model)
                V: (batch_size, seq_len2, d_model]
                mask: (batch_size, seq_len1, seq_len2)
            outputs:
                output: (batch_size, seq_len1, d_model)
                attn: (batch_size, n_heads, seq_len1, seq_len2)
        """
        batch_size = Q.shape[0]

        # Q -> (batch_size, seq_len1, d_k * n_head) -> (batch_size, seq_len1, n_head, d_k) -> (batch_size, n_head, seq_len1, d_k)
        name = self.name + ".mha.W_Q"
        Q = self.W_Q(Q, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)
        Q = Q.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # K -> (batch_size, seq_len2, d_k * n_head) -> (batch_size, seq_len2, n_head, d_k) -> (batch_size, n_head, seq_len2, d_k)
        name = self.name + ".mha.W_K"
        K = self.W_K(K, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)
        K = K.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # V -> (batch_size, seq_len2, d_v * n_head) -> (batch_size, seq_len2, n_head, d_v) -> (batch_size, n_head, seq_len2, d_v)
        name = self.name + ".mha.W_V"
        V = self.W_V(V, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)
        V = V.view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # mask -> (batch_size, 1, seq_len1, seq_len2) -> (batch_size, n_heads, seq_len1, seq_len2)
        context, attn = self.sdpa(Q, K, V)

        # (batch_size, seq_len1, n_heads * d_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        name = self.name + ".mha.fc"
        output = self.fc(context, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)

        return output, attn


class SubnetEncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_params=True, sparsity=0.0, name=""):
        super().__init__()
        self.name = name
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=norm_params, bias=norm_params)
        self.mha = SubnetMultiHeadAttention(hidden_dim, num_heads, p_dropout=attention_dropout, sparsity=sparsity, name=name)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=norm_params, bias=norm_params)
        self.mlp_lin1 = SubnetLinear(hidden_dim, mlp_dim, bias=True, sparsity=sparsity)
        self.mlp_act = torch.nn.GELU()
        self.mlp_lin2 = SubnetLinear(mlp_dim, hidden_dim, bias=True, sparsity=sparsity)

    def forward(self, input, mask, mode='train'):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.mha(x, x, x, mask, mode)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        name = self.name + ".mlp_lin1"
        y = self.mlp_lin1(y, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)
        y = self.dropout(self.mlp_act(y))
        name = self.name + ".mlp_lin2"
        y = self.mlp_lin2(y, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)
        return x + self.dropout(y)


class SubnetVisionTransformer(BaseModel):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.1,
                 attention_dropout=0.1, num_classes=200, norm_params=True, n_tasks=1, sparsity=0.0):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_params = norm_params

        seq_length = (image_size // patch_size) ** 2

        self.conv1 = SubnetConv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size,
                                  bias=True, sparsity=sparsity)

        # Add a class token
        self.class_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(n_tasks)])
        seq_length += 1
        self.seq_length = seq_length

        # Transformer Encoder
        # Note that batch_size is on the first dim because we have batch_first=True in nn.MultiAttention() by default
        self.pos_embeddings = nn.ParameterList([nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)) for _ in range(n_tasks)])
        self.d_out = nn.Dropout(dropout)
        layers = []
        for i in range(num_layers):
            layers.append(SubnetEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_params,
                sparsity,
                f"encoder_layers.{i}",
            ))
        self.encoder_layers = maskedSequentialViT(*layers)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=norm_params, bias=norm_params)

        self.classifier = SubnetClassifier(hidden_dim, num_classes, n_tasks, bias=False)

        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def _process_input(self, x, mask, mode):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, task=-1, mask=None, mode="train", returnt='out'):
        if mask is None:
            mask = self.none_masks

        # Reshape and permute the input tensor
        x = self._process_input(x, mask, mode)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_tokens[task].expand(n, -1, -1)   # if task==-1, use whatever, doesn't matter
        x = torch.cat([batch_class_token, x], dim=1)

        # Forward pass from the encoder
        x = x + self.pos_embeddings[task]   # if task==-1, use whatever, doesn't matter
        x = self.ln(self.encoder_layers(self.d_out(x), mask, mode))

        # Classifier "token" as used by standard language architectures
        feature = x[:, 0]

        if returnt == 'features':
            return feature

        out = self.classifier(feature, task)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return out, feature

        raise NotImplementedError("Unknown return type")

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)
                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
            elif isinstance(module, SubnetClassifier):
                task_mask[name + '.weight'] = module.wired_masks[task_id].detach().clone().to(module.weight.device)
        return task_mask

    def reinit_scores(self):
        # print("reinitializing scores")
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                module.init_mask_parameters()

    def reinit_weights(self, combined_masks):
        # print("reinitializing weights")
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d) or isinstance(module, SubnetClassifier):
                if combined_masks != {}:
                    module.weight.data = torch.where(combined_masks[name + '.weight'], module.weight,
                                                     nn.init.kaiming_uniform_(torch.zeros_like(module.weight), a=math.sqrt(5)))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in)
                        module.bias.data = torch.where(combined_masks[name + '.bias'], module.bias,
                                                       nn.init.uniform_(torch.zeros_like(module.bias), -bound, bound))
                else:
                    module.weight.data = nn.init.kaiming_uniform_(torch.zeros_like(module.weight), a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in)
                        module.bias.data = nn.init.uniform_(torch.zeros_like(module.bias), -bound, bound)


def subnet_vit_t_8(num_classes, norm_params=False, n_tasks=1, sparsity=0.0):
    return SubnetVisionTransformer(image_size=64, patch_size=8, num_layers=12, num_heads=3, hidden_dim=192,
                                   mlp_dim=768, num_classes=num_classes, norm_params=norm_params,
                                   n_tasks=n_tasks, sparsity=sparsity)


def subnet_vit_t_16(num_classes, norm_params=False, n_tasks=1, sparsity=0.0):
    return SubnetVisionTransformer(image_size=64, patch_size=16, num_layers=12, num_heads=3, hidden_dim=192,
                                   mlp_dim=768, num_classes=num_classes, norm_params=norm_params,
                                   n_tasks=n_tasks, sparsity=sparsity)
