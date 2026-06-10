//! Granite transformer block: Llama-style pre-norm residual flow with a
//! `residual_multiplier` applied to each sublayer output (muP-style), and an
//! MLP that loads either standard `mlp.{gate,up,down}_proj` weights
//! (Granite 3.x) or the fused `shared_mlp.input_linear`/`output_linear`
//! layout (dense GraniteMoeHybrid, e.g. granite-4.0-1b).
//!
//! The attention softmax scale (`attention_multiplier`) is handled inside
//! [`CausalSelfAttention`] via `Config.attn_scale`.

use anyhow::Result;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use crate::models::common::{CausalSelfAttention, MLP};
use async_trait::async_trait;

/// A Granite transformer block.
#[derive(Debug, Clone)]
pub struct GraniteBlock {
    name: String,
    rms_1_weight: Tensor,
    rms_2_weight: Tensor,
    rms_eps: f32,
    /// Granite `residual_multiplier` (1.0 when absent).
    residual_scale: f64,
    attn: CausalSelfAttention,
    mlp: MLP,
}

impl std::fmt::Display for GraniteBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local, granite)", &self.name)
    }
}

#[async_trait]
impl Forwarder for GraniteBlock {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, ctx.backend.clone())?;
        let mlp = if cfg.granite_shared_mlp {
            MLP::load_granite_shared(vb.pp("shared_mlp"), cfg, ctx.backend.clone())?
        } else {
            MLP::load(vb.pp("mlp"), cfg, ctx.backend.clone())?
        };

        let rms_1_weight = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let rms_2_weight = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;

        Ok(Box::new(Self {
            name,
            rms_1_weight,
            rms_2_weight,
            rms_eps: cfg.rms_norm_eps as f32,
            residual_scale: cfg.residual_scale.unwrap_or(1.0),
            attn,
            mlp,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let residual = x;
        let x = ctx
            .backend
            .rms_norm(x, &self.rms_1_weight, self.rms_eps)
            .map_err(|e| anyhow!("rms_1: {e}"))?;
        let x = self
            .attn
            .forward(
                &x,
                index_pos,
                block_idx,
                ctx.cache.as_mut().expect("No cache specified"),
            )
            .map_err(|e| anyhow!("attention: {e}"))?;
        let x = ((x * self.residual_scale)? + residual)
            .map_err(|e| anyhow!("attn residual: {e}"))?;

        let residual = &x;
        let x = ctx
            .backend
            .rms_norm(&x, &self.rms_2_weight, self.rms_eps)
            .map_err(|e| anyhow!("rms_2: {e}"))?;
        let x = self.mlp.forward(&x).map_err(|e| anyhow!("mlp: {e}"))?;
        let x = ((x * self.residual_scale)? + residual)
            .map_err(|e| anyhow!("mlp residual: {e}"))?;

        Ok(x)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}
