import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch import Tensor
from mamba_ssm.modules.spd import *
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def FM(A, B, a, n):
    '''
    Compute the Weighted Fréchet mean
    '''
    return torch.add((1.0 - a) * A, a * B)

def add_small_diagonal(matrix, epsilon=1e-6):
    """
    Add a small value to the diagonal elements of the matrix.
    """
    return matrix + torch.eye(matrix.size(0), device=matrix.device) * epsilon

# Apply perturbation

def NUS(W_root, A, a_num, tot, n=1):
    '''
    Compute the weighted average on the M -> Y
    '''
    W = torch.pow(W_root, 2)
    if a_num == 1:
        return (W[0] / tot) * A
    else:
        # result = torch.squeeze(A[:, :1, :, :]) * (W[0] / tot)
        result = A[:, 0, :, :] * (W[0] / tot)
        for i in range(1, A.shape[1]):
            # result = result + torch.squeeze(A[:, i : i + 1, :, :]) * (W[i] / tot)
            result = result + A[:, i, :, :] * (W[i] / tot)
        return result

def MatrixExp(B, l, n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''

    Result = torch.eye(n, device=B.device)
    B = (B + B.T)/2 
    B = add_small_diagonal(B)
    matrix_to_invert = torch.subtract(Result, B)
    # Result = torch.eye(n, device=B.device).unsqueeze(0).repeat(B.shape[0], 1, 1)
    return torch.matmul(torch.inverse(matrix_to_invert), torch.add(Result, B))
    # return torch.matmul(torch.inverse(torch.subtract(Result, B)), torch.add(Result, B))
    # return torch.matmul(torch.linalg.inv(Result - B), Result + B)

def Translation_batch(A, B, n, batch_size):
    '''
    input the matrix A and vector B
    change B to be SO
    like [[0 ,  1, 2]
          [-1,  0, 3]
          [-2, -3, 0]]
    return B * A * B.T
    '''
    power_matrix = 5
    B = torch.reshape(B, [1, -1])

    line_B = [torch.zeros([1, n], device=A.device)]
    for i in range(n - 1):
        temp_line = torch.cat(
            [B[:1, i : 2 * i + 1], torch.zeros([1, n - i - 1], device=A.device)], axis=1
        )
        line_B.append(temp_line)

    lower_triangle = torch.cat(line_B, axis=0)

    B_matrix = torch.subtract(lower_triangle, lower_triangle.T)
    B_matrix = MatrixExp(B_matrix, power_matrix, n)

    B_matrix = torch.unsqueeze(B_matrix, 0).unsqueeze(0)  #  [n, n] -> [1, 1, n, n]
    
    B_matrix = B_matrix.repeat(batch_size, 1, 1, 1)  # [batch_size, 1, n, n]

    A = torch.einsum('tbmn,b1mn->tbmn', A, B_matrix)
    Tresult = torch.einsum('tbmn,b1nm->tbmn', A, B_matrix.permute(0, 1, 3, 2))

    return Tresult

def Translation(A, B, n, batch_size):

    '''
    input the matrix A and vector B
    change B to be SO
    like [[0 ,  1, 2]
          [-1,  0, 3]
          [-2, -3, 0]]
    return B * A * B.T
    '''
    power_matrix = 5
    B = torch.reshape(B, [1, -1])
    
    # lower_triangel = fill_triangular(B)
    line_B = [torch.zeros([1, n], device=A.device)]
    for i in range(n - 1):
        temp_line = torch.cat(
            [B[:1, i : 2 * i + 1], torch.zeros([1, n - i - 1], device=A.device)], axis=1
        )
        line_B.append(temp_line)

    lower_triangel = torch.cat(line_B, axis=0)

    B_matrix = torch.subtract(lower_triangel, lower_triangel.T)
    # print("B1_matrix",B_matrix.shape)
    B_matrix = (B_matrix + B_matrix.T) / 2
    B_matrix = add_small_diagonal(B_matrix)
    B_matrix = MatrixExp(B_matrix, power_matrix, n)
    B_matrix = torch.unsqueeze(B_matrix, 0).repeat([batch_size, 1, 1])
    # print("B_matrix",B_matrix.shape)
    # print("A_matrix",A.shape)
    Tresult = torch.matmul(B_matrix, A)  # B * A

    Tresult = torch.matmul(Tresult, B_matrix.permute([0, 2, 1]))  # B * A * B.T
    return Tresult

def Chol_de(A, n):
    '''
    input matrix A and it's size n
    decomponent by Cholesky
    return a vector with size n*(n+1)/2
    '''
    # A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
    # A = tf.cond(
    #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) ,
    #     lambda: A,
    #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
    # L = tf.cholesky(A)

    L = A
    result = L[:, :1, :1]
    for i in range(1, n):
        j = i
        result = torch.cat([result, L[:, i : i + 1, : j + 1]], axis=2)

    result = torch.reshape(result, [-1, n * (n + 1) // 2])
    return result


def Chol_com(l, n, batch_size):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    lower_triangle_ = torch.unsqueeze(
        torch.cat([l[:, :1], torch.zeros((batch_size, n - 1))], axis=1),
        1,
    )
    for i in range(1, n):
        lower_triangle_ = torch.cat(
            [
                lower_triangle_,
                torch.unsqueeze(
                    torch.cat(
                        [
                            l[:, i * (i + 1) // 2 : i * (i + 1) // 2 + i + 1],
                            torch.zeros((batch_size, n - i - 1)),
                        ],
                        axis=1,
                    ),
                    1,
                ),
            ],
            axis=1,
        )

    lower_triangle_ = torch.add(
        lower_triangle_,
        torch.unsqueeze(torch.eye(n) * 1e-2, axis=0).repeat([batch_size, 1, 1]),
    )
    result = torch.matmul(lower_triangle_, lower_triangle_.transpose())
    return result

class GeoMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=5,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        time_point=39,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model*self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # manifold conv, activate, and nn.Linear 
        self.SPDconv = SPDConv(time_point,self.d_inner,d_conv)
        self.SPDattention = SPDAttention(time_point,d_conv)
        self.SPDconv2 = SPDConv(self.d_inner,2*self.d_inner,d_conv)
        self.SPDact = SPDActivate(activate_func='exp')
        self.norm=Normalize(p='fro', dim=(-2, -1))
        self.activation = "silu"
        self.act = nn.SiLU()
        self.WR_root = torch.nn.Parameter(torch.Tensor(1))
        self.Wt_root = torch.nn.Parameter(torch.Tensor(1))
        self.Wphi_root = torch.nn.Parameter(torch.Tensor(1))
        self.Ws_root = torch.nn.Parameter(torch.Tensor(1))

        self.Br = torch.nn.Parameter(torch.Tensor((self.d_inner-self.d_conv+1) * ((self.d_inner-self.d_conv+1) - 1) // 2, 1))
        self.Bt = torch.nn.Parameter(torch.Tensor((self.d_inner-self.d_conv+1) * ((self.d_inner-self.d_conv+1) - 1) // 2, 1))
        self.By = torch.nn.Parameter(torch.Tensor((self.d_inner-self.d_conv+1) * ((self.d_inner-self.d_conv+1) - 1) // 2, 1))
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # print("dt_init",dt.shape)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear((self.d_model-self.d_conv+1)*(self.d_model-self.d_conv+1), self.d_model, bias=bias, **factory_kwargs)
        # self.out_proj = nn.Linear((self.d_model)*(self.d_model), self.d_model*2, bias=bias, **factory_kwargs)
        # self.out_proj2 = nn.Linear(self.d_model*2, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim, dim2 = hidden_states.shape

        conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out
        # print("weight_dimension",(self.in_proj.weight).shape)
        # We do matmul and transpose BLH -> HBL at the same time (BHL)
        # xz = rearrange(
        #     self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        #     "d (b l) -> b d l",
        #     l=seqlen,
        # ) # b d l
        b, l, d1, d2 = hidden_states.shape
        reshaped_weight = rearrange(self.in_proj.weight, 'o (d1 d2) -> o d1 d2', d1=d1, d2=d2)  # (output_dim, d1, d2)
        # hidden_states_reshaped = hidden_states.view(b, l, d1 * d2)
        xz = torch.einsum('ode, blde -> bol', reshaped_weight, hidden_states) 
        # xz = xz.permute(0, 2, 1) 
    #     xz = rearrange(
    #     torch.einsum('o d1 d2, b l d1 d2 -> b o l', reshaped_weight, hidden_states),
    #     'b o l -> b l o'
    # )  # b l o -> b o l
    #     xz = rearrange(xz, 'b d l -> b l d') 
        # manifold version
        # xz =  rearrange(hidden_states,'b l d1 d2 -> b d1 d2 l')
        xz = hidden_states 
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x = xz
            z = xz
            # print("x_dimension",x.shape)#[16,720,39]
            # print("z_dimension",z.shape)#[16,720,39]
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                # x = self.act(self.conv1d(x)[..., :seqlen])
                # print("x1_dimension",x.shape)#([16, 720, 39])
                ##need to revise to manifold
                # print(x.shape)
                output, attention = self.SPDattention(x)
                x = output[:, :seqlen, :, :]
                x = self.SPDact(self.SPDconv(x)[:, :seqlen,:,:])
                x = self.norm(x)
                # print("x1spd1_dimension",x.shape)#([16, 39, 357, 357])
                # x = self.SPDact(self.SPDconv(x)[:, :seqlen,:,:])
                # x = self.norm(x)
                # x = self.SPDact(self.SPDconv(x)[:, :seqlen,:,:])
                # x = self.norm(x)
                # print("x1spd2_dimension",x.shape)#([16, 39, 360, 360])
                # print("x1spd_dimension",x.shape)#([16, 39, 360, 360])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                # print("x2_dimension",x.shape) #no
            # print("x",x[1,1,:,:])
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d) ([624, 39])self.x_proj一个nn.linear操作
            # manifold
            x = rearrange(x,'b l d1 d2 -> l b d1 d2')
#             x1 = x[1,:,:,:]
#             Yt = NUS(
#             self.WR_root,
#             x,
#             1,
#             torch.sum(torch.pow(self.WR_root, 2)) + (1e-10),
#             (dim2-self.d_conv+1),
#         )
#             x_dbl = FM(
#             x,
#             Yt,
#             torch.pow(self.Wt_root, 2)
#             / (
#                 torch.sum(
#                     torch.pow(self.Wt_root, 2)
#                     + torch.pow(self.Wphi_root, 2)
#                 )
#                 + (1e-10)
#             ),
#             (dim2-self.d_conv+1),
#         )
#             # print("x_dbl",x_dbl[1,1,:,:])
#             # print("x_dblspdweight_dimension",x_dbl.shape)
#             # print("Btspdweight_dimension",(self.Bt).shape)
#             results =[]
#             # for t in range(x_dbl.shape[0]):
#             #     x_dbl2 = x_dbl[t,:,:,:]
#             #     result_t = Translation(x_dbl2, self.Bt, (dim2-self.d_conv+1), batch)
#             #     results.append(result_t)
#             x_dbl2 = torch.stack(results, dim=0)
#             # print("x_dbl2",x_dbl2[1,1,:,:])
#             # print("x_dblavage_dimension",x_dbl2.shape) 
#             X_dbl2 = rearrange(x_dbl2,'b l d1 d2 -> (b l) d1 d2')
#             # dt, B, C = torch.split(x_dbl2, [self.dt_rank, self.d_state, self.d_state], dim=0)
#             dt = X_dbl2[:, :self.dt_rank, :self.dt_rank]
#             B = X_dbl2[:, self.dt_rank:self.dt_rank+self.d_state, self.dt_rank:self.dt_rank+self.d_state]
#             C = X_dbl2[:, -self.d_state:, -self.d_state:]
#             # print("dt_dimension",dt.shape)#([624, 23])
#             # print("B_dimension",B.shape)#([624, 8])
#             # print("C_dimension",C.shape)#([624, 8])
#             # print("dtt_dimension",dt.t().shape)
#             # dt = self.dt_proj.weight @ dt.t()
#             Yt2 = NUS(
#             self.WR_root,
#             dt,
#             1,
#             torch.sum(torch.pow(self.WR_root, 2)) + (1e-10),
#             (self.dt_rank),
#         )
#             dt = FM(
#             dt,
#             Yt2,
#             torch.pow(self.Wt_root, 2)
#             / (
#                 torch.sum(
#                     torch.pow(self.Wt_root, 2)
#                     + torch.pow(self.Wphi_root, 2)
#                 )
#                 + (1e-10)
#             ),
#             (self.dt_rank),
#         )
#             # print("dt2_dimension",dt.shape) #([720, 624])
#             dt = rearrange(dt, "(b l) d1 d2 -> b d1 d2 l", l=seqlen) #([16, 720, 39])
#             # print("dt3_dimension",dt.shape) #([16, 720, 39])
#             B = rearrange(B, "(b l) dstate1  dstate2-> b dstate1  dstate2 l", l=seqlen).contiguous()
#             # print("B2_dimension",B.shape)#([16, 8, 39])
#             C = rearrange(C, "(b l) dstate1  dstate2 -> b dstate1  dstate2 l", l=seqlen).contiguous()
#             # print("C2_dimension",C.shape)#([16, 8, 39])
#             assert self.activation in ["silu", "swish"]
#             # y = selective_scan_fn(
#             #     x,
#             #     dt,
#             #     A,
#             #     B,
#             #     C,
#             #     self.D.float(),
#             #     z=z,
#             #     delta_bias=self.dt_proj.bias.float(),
#             #     delta_softplus=True,
#             #     return_last_state=ssm_state is not None,
#             # )
#             y = torch.zeros((batch, self.dt_rank+2*self.d_state, self.dt_rank+2*self.d_state, l), device=dt.device)
# #             y = torch.cat([
# #     torch.cat([dt, B, C], dim=1),
# #     torch.cat([dt, B, C], dim=2)
# # ], dim=2)
#             y[:, :self.dt_rank, :self.dt_rank, :] = dt
#             y[:, self.dt_rank:(self.dt_rank+self.d_state), self.dt_rank:(self.dt_rank+self.d_state), :] = B
#             y[:, (self.dt_rank+self.d_state):, (self.dt_rank+self.d_state):, :] = C
#             # print("y_dimension",y.shape)#([16, 720, 39])
#             if ssm_state is not None:
#                 y, last_state = y
#                 # print("y2_dimension",y.shape)
#                 # print("last_state_dimension",last_state.shape)
#                 ssm_state.copy_(last_state)
#             y = rearrange(y, "b d1 d2 l -> l b d1 d2")
#             # print("y3_dimension",y.shape)#([16, 39, 720])
#             # out = self.out_proj(y)
#             Yt3 = NUS(
#             self.Ws_root,
#             y,
#             1,
#             torch.sum(torch.pow(self.Ws_root, 2)) + (1e-10),
#             (self.dt_rank),
#         )
#             y = FM(
#             y,
#             Yt3,
#             torch.pow(self.Wt_root, 2)
#             / (
#                 torch.sum(
#                     torch.pow(self.Wt_root, 2)
#                     + torch.pow(self.Wphi_root, 2)
#                 )
#                 + (1e-10)
#             ),
#             (self.dt_rank),
#         )
#             results =[]
#             for t in range(y.shape[0]):
#                 x_dbl2 = y[t,:,:,:]
#                 result_t = Translation(x_dbl2, self.Bt, l, batch)
#                 results.append(result_t)
#             out = torch.stack(results, dim=0)
#             # print("out",out[1,1,:,:])
#             # print("out2_dimension",out.shape)#([16, 39, 360])
#             # reshaped_tensor = out.view(-1, l, l)
#             # tangent_space_matrices =torch.stack([SPDTangentSpace(mat.unsqueeze(0)) for mat in reshaped_tensor])
#             # out =tangent_space_matrices.view(l, batch, l, l)
#             lower_triangular = torch.tril(out)
#             out = lower_triangular.view(lower_triangular.shape[0], lower_triangular.shape[1], -1)
#             # out = self.out_proj(out)
#             # print("x",x.shape)
            lower_triangular2 = torch.tril(x)
            out2 = lower_triangular2.view(lower_triangular2.shape[0], lower_triangular2.shape[1], -1)
            out = self.out_proj(out2)
            # out = self.out_proj2(out)
            out = rearrange(out, "l b d -> b l d")
            # print("out2",out[1,:,:])
            # print("out22_dimension",out.shape)#([16, 39, 360])
        return out, attention

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
