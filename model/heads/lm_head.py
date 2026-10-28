    def forward(
        self,
        input_ids: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"EmbeddingLayer expected torch.Tensor, got {type(input_ids)}"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, seq_len); got {tuple(input_ids.shape)}"
            )

        seq_len = input_ids.size(1)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(
            seq_len=seq_len,
            offset=offset,
            device=input_ids.device,
        )

        return self.dropout(token_embeddings + position_embeddings)
