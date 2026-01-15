"""Meta Connector."""

import torch
from torch import nn
from torch.nn import functional as func


class Expert(nn.Module):
    """Position-wise Feed-Forward Networks.

    This consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_expert: int) -> None:
        """Initialize the Expert layer.

        Args:
            d_model (int): The dimension of the model.
            d_expert (int): The dimension of the expert.
        """
        super().__init__()
        self.d_model = d_model
        self.d_expert = d_expert

        # Linear transformation y = xW+b
        self.fc1 = nn.Linear(self.d_model, self.d_expert, bias=True)
        self.fc2 = nn.Linear(self.d_expert, self.d_model, bias=True)
        self.fc3 = nn.Linear(self.d_model, self.d_expert, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights of the Expert layer."""
        # init weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Expert layer.

        Args:
            embedding (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        inner_emb = func.silu(self.fc1(embedding)) * self.fc3(embedding)
        return self.fc2(inner_emb)


class MetaAttentionBlock(nn.Module):
    """Meta Attention Block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.05) -> None:
        """Initialize the Meta Attention Block.

        Args:
            d_model (int): The dimension of the embeddings.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Dropout layers for residual connections
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.mlp = Expert(d_model, d_model * 4)

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for attention and MLP blocks."""
        # Initialize MultiheadAttention weights
        # The in_proj_weight contains Q, K, V concatenated
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)

        # Initialize the output projection of MultiheadAttention
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MetaAttentionBlock.

        Args:
            embedding (torch.Tensor): The input embeddings. Shape: (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: The output embeddings. Shape: (batch_size, seq_len, hidden_dim).
        """
        # apply layer normalization to embeddings
        normed_embedding = self.input_norm(embedding)

        # apply attention and skip connection
        attn_output, _ = self.attn(normed_embedding, normed_embedding, normed_embedding, need_weights=False)
        embedding = embedding + self.dropout_attn(attn_output)

        output = self.mlp(self.output_norm(embedding))
        return embedding + self.dropout_ffn(output)


class MetaConnector(nn.Module):
    """Meta Connector."""

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int = 4096,
        hidden_dim: int = 2048,
        output_dim: int = 768,
        n_stacks: int = 8,
        heads: int = 16,
        *,
        use_input_proj: bool = True,
    ) -> None:
        """Initialize the MetaConnector.

        Args:
            input_dim (int): The dimension of the input.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            n_stacks (int): The number of attention layers.
            heads (int): The number of attention heads.
            use_input_proj (bool): Whether to use input projection.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.use_input_proj = use_input_proj

        # Input Projection
        if use_input_proj:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            hidden_dim = input_dim

        # Attention blocks
        self.attn_blocks = nn.Sequential(*[MetaAttentionBlock(hidden_dim, heads) for _ in range(n_stacks)])

        # Output Projection
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        # Initialize input_proj
        if self.use_input_proj:
            nn.init.xavier_uniform_(self.input_proj.weight)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)

        # Initialize output_proj
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, meta_queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the Meta Connector.

        Args:
            hidden_states (torch.Tensor): The input hidden states. Shape: (batch_size, seq_len, hidden_dim).
            meta_queries (torch.Tensor): The input meta queries. Shape: (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: The output embeddings. Shape: (batch_size, seq_len, hidden_dim).
        """
        hidden_states = hidden_states + meta_queries
        if self.use_input_proj:
            hidden_states = self.input_proj(hidden_states)

        # apply attention blocks
        for attn_block in self.attn_blocks:
            hidden_states = attn_block(hidden_states)

        # apply output projection
        return self.output_proj(hidden_states)
